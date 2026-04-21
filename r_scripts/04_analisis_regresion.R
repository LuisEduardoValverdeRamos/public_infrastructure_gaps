# =============================================================================
# GFP — Análisis de Regresión Logit + AME
# Pipeline automatizado: Contrata · AD · ARCC
# =============================================================================

suppressPackageStartupMessages({
  library(rio)
  library(dplyr)
  library(tidyr)
  library(margins)
  library(pROC)
  library(ggplot2)
  library(scales)
  library(broom)
  library(stringr)
  library(knitr)
})

# =============================================================================
# 0. CONFIGURACIÓN
# =============================================================================

CONFIGS <- list(
  list(modalidad = "contrata", sampling = "nrs", top_n = 10),
  list(modalidad = "ad",       sampling = "o",   top_n = 10),
  list(modalidad = "arcc",     sampling = "o",   top_n = 10)
)

BASE_DIR <- "C:/15_GFP"
DIR_LOGS <- file.path(BASE_DIR, "outputs/regresiones")
dir.create(DIR_LOGS, showWarnings = FALSE, recursive = TRUE)

LOG_FILE <- file.path(DIR_LOGS, paste0("analysis_log_",
                       format(Sys.time(), "%Y%m%d_%H%M%S"), ".txt"))

# =============================================================================
# 1. LOGGER
# =============================================================================

log_con <- file(LOG_FILE, open = "wt", encoding = "UTF-8")

log <- function(..., level = "INFO") {
  msg <- paste0("[", format(Sys.time(), "%H:%M:%S"), "] [", level, "] ", ...)
  cat(msg, "\n")                       # consola
  cat(msg, "\n", file = log_con)       # archivo
}

log_section <- function(title) {
  sep <- paste(rep("=", 70), collapse = "")
  cat("\n", sep, "\n", title, "\n", sep, "\n\n", file = log_con)
  cat("\n", sep, "\n", title, "\n", sep, "\n\n")
}

log_table <- function(df, caption = "") {
  if (nchar(caption) > 0) log(caption)
  txt <- capture.output(print(df, row.names = FALSE))
  for (l in txt) cat("    ", l, "\n", file = log_con)
  for (l in txt) cat("    ", l, "\n")
}

on.exit({ log("Log cerrado."); close(log_con) }, add = TRUE)

log_section("GFP — ANÁLISIS DE REGRESIÓN LOGIT")
log(paste0("Inicio      : ", format(Sys.time(), "%Y-%m-%d %H:%M:%S")))
log(paste0("R version   : ", R.version$version.string))
log(paste0("Log file    : ", LOG_FILE))
log(paste0("Modalidades : ", paste(sapply(CONFIGS, `[[`, "modalidad"), collapse = ", ")))

# =============================================================================
# 2. FUNCIONES AUXILIARES
# =============================================================================

# ── Tema ggplot2 ─────────────────────────────────────────────────────────────
theme_gfp <- function(base_size = 12) {
  theme_minimal(base_size = base_size) %+replace%
    theme(
      plot.title       = element_text(face = "bold", size = base_size + 2, hjust = 0),
      plot.subtitle    = element_text(color = "gray40", size = base_size - 1, hjust = 0),
      plot.caption     = element_text(color = "gray55", size = 9, hjust = 1),
      axis.title       = element_text(color = "gray30", size = base_size - 1),
      axis.text        = element_text(color = "gray20"),
      panel.grid.minor  = element_blank(),
      panel.grid.major  = element_line(color = "gray92"),
      legend.position  = "bottom",
      legend.title     = element_blank(),
      strip.text       = element_text(face = "bold")
    )
}

COLORES <- c("Positivo sig." = "#2c7bb6",
             "Negativo sig." = "#d7191c",
             "n.s."          = "#bdbdbd")

# ── Acortar etiquetas ─────────────────────────────────────────────────────────
shorten_label <- function(x, max_len = 28) {
  x <- str_replace(x,  "^Region_",              "reg: ")
  x <- str_replace(x,  "^naturaleza_obra_",      "nat: ")
  x <- str_replace(x,  "^modalidad_ejecucion_",  "mod: ")
  x <- str_replace(x,  "^tipo_obra_nivel(\\d)_", "t\\1: ")
  x <- str_replace(x,  "^tipo_obra_full_[^_]+_[^_]+_", "tf: ")
  x <- str_replace_all(x, "_", " ")
  str_trunc(x, max_len, ellipsis = "...")
}

# ── Cargar datos ──────────────────────────────────────────────────────────────
load_modality <- function(cfg) {
  path_data <- file.path(BASE_DIR, "data/processed", cfg$modalidad,
                         paste0("1_data_", cfg$modalidad, ".xlsx"))
  path_shap <- file.path(BASE_DIR, "outputs/shap", cfg$modalidad,
                         paste0("shap_values_xgb_", cfg$sampling, ".xlsx"))

  if (!file.exists(path_data))
    stop(sprintf("No existe data procesada: %s", path_data))
  if (!file.exists(path_shap))
    stop(sprintf("No existe SHAP output: %s  ->  corre primero 03_shap_%s.ipynb",
                 path_shap, cfg$modalidad))

  data_full    <- import(path_data)
  shap_df      <- import(path_shap)
  top_features <- shap_df$feature[seq_len(cfg$top_n)]

  faltantes <- setdiff(top_features, colnames(data_full))
  if (length(faltantes) > 0) {
    log(sprintf("Features SHAP no encontradas en data: %s",
                paste(faltantes, collapse = ", ")), level = "WARN")
    top_features <- intersect(top_features, colnames(data_full))
  }

  subdata <- data_full[, c(top_features, "brecha_existente"), drop = FALSE]
  subdata <- na.omit(subdata)

  orig_names     <- colnames(subdata)
  clean_names    <- make.names(orig_names, unique = TRUE)
  colnames(subdata) <- clean_names
  name_map       <- setNames(clean_names, orig_names)   # orig -> clean
  inv_map        <- setNames(orig_names, clean_names)   # clean -> orig

  list(
    data         = subdata,
    top_features = top_features,
    name_map     = name_map,
    inv_map      = inv_map,
    shap_df      = shap_df,
    n            = nrow(subdata),
    n1           = sum(subdata$brecha_existente == 1),
    n0           = sum(subdata$brecha_existente == 0)
  )
}

# ── Ajustar modelos ───────────────────────────────────────────────────────────
fit_models <- function(ld) {
  pred_vars <- setdiff(colnames(ld$data), "brecha_existente")
  fml       <- as.formula(paste("brecha_existente ~",
                                paste(pred_vars, collapse = " + ")))
  logit <- glm(fml, family = binomial(link = "logit"), data = ld$data)
  lpm   <- lm(fml, data = ld$data)
  list(logit = logit, lpm = lpm, formula = fml, pred_vars = pred_vars)
}

# ── Métricas ──────────────────────────────────────────────────────────────────
fit_metrics <- function(logit, data) {
  null_ll <- logLik(glm(brecha_existente ~ 1,
                         family = binomial, data = data))
  mcf   <- round(1 - as.numeric(logLik(logit)) / as.numeric(null_ll), 4)
  probs <- predict(logit, type = "response")
  roc_o <- roc(data$brecha_existente, probs, quiet = TRUE)
  preds <- ifelse(probs >= 0.5, 1, 0)
  acc   <- mean(preds == data$brecha_existente)
  list(mcfadden = mcf, auc = round(auc(roc_o), 4),
       accuracy = round(acc, 4), roc = roc_o)
}

# ── AME ───────────────────────────────────────────────────────────────────────
compute_ame <- function(logit, inv_map) {
  ame_raw <- margins(logit)
  ame_df  <- as.data.frame(summary(ame_raw))

  ame_df$var_original <- inv_map[ame_df$factor]
  ame_df$label <- shorten_label(
    ifelse(is.na(ame_df$var_original), ame_df$factor, ame_df$var_original)
  )
  ame_df$sig <- case_when(
    ame_df$p < 0.01 ~ "***",
    ame_df$p < 0.05 ~ "**",
    ame_df$p < 0.10 ~ "*",
    TRUE             ~ ""
  )
  ame_df$categoria <- case_when(
    ame_df$p >= 0.05  ~ "n.s.",
    ame_df$AME > 0    ~ "Positivo sig.",
    TRUE               ~ "Negativo sig."
  )
  ame_df[order(ame_df$AME, decreasing = TRUE), ]
}

# ── Hallazgos en texto ────────────────────────────────────────────────────────
generate_findings <- function(ame_df, metrics, ld, cfg) {
  sig     <- ame_df[ame_df$p < 0.05, ]
  pos_sig <- sig[sig$AME > 0, ]
  neg_sig <- sig[sig$AME < 0, ]
  max_row <- ame_df[which.max(abs(ame_df$AME)), ]
  dir_str <- ifelse(max_row$AME > 0, "aumenta", "reduce")

  lines <- c(
    sprintf("MODALIDAD: %s | XGB %s | Top-%d SHAP",
            toupper(cfg$modalidad), toupper(cfg$sampling), cfg$top_n),
    sprintf("Muestra   : %d obs (brecha=1: %d [%.1f%%], brecha=0: %d [%.1f%%])",
            ld$n, ld$n1, 100*ld$n1/ld$n, ld$n0, 100*ld$n0/ld$n),
    sprintf("Ajuste    : AUC=%.3f | McFadden R2=%.3f | Exactitud=%.1f%%",
            metrics$auc, metrics$mcfadden, 100*metrics$accuracy),
    sprintf("Sig. (5%%) : %d de %d variables", nrow(sig), nrow(ame_df)),
    ""
  )

  if (nrow(pos_sig) > 0)
    lines <- c(lines, sprintf("(+) %s  [AME=%.3f, p=%.3f]%s",
                               pos_sig$label, pos_sig$AME,
                               pos_sig$p, pos_sig$sig))
  if (nrow(neg_sig) > 0)
    lines <- c(lines, sprintf("(-) %s  [AME=%.3f, p=%.3f]%s",
                               neg_sig$label, neg_sig$AME,
                               neg_sig$p, neg_sig$sig))
  if (nrow(sig) == 0)
    lines <- c(lines, "ALERTA: ninguna variable significativa al 5%")

  lines <- c(lines, "",
    sprintf("Mayor efecto abs.: '%s' (AME=%.3f) — %s la Pr(brecha) en %.1f pp",
            max_row$label, max_row$AME, dir_str, abs(max_row$AME)*100))
  lines
}

# =============================================================================
# 3. GRÁFICOS
# =============================================================================

plot_ame <- function(ame_df, metrics, ld, cfg, dir_out) {
  p <- ame_df %>%
    mutate(label = factor(label, levels = label[order(AME)])) %>%
    ggplot(aes(x = AME, y = label, color = categoria)) +
    geom_vline(xintercept = 0, linetype = "dashed",
               color = "gray60", linewidth = 0.6) +
    geom_errorbarh(aes(xmin = lower, xmax = upper),
                   height = 0.3, linewidth = 0.7, alpha = 0.8) +
    geom_point(size = 3.5) +
    scale_color_manual(values = COLORES) +
    scale_x_continuous(labels = percent_format(accuracy = 1)) +
    labs(
      title    = "Efectos Marginales Promedio (AME) — Logit",
      subtitle = sprintf("%s | XGB %s | Top %d vars SHAP",
                         toupper(cfg$modalidad), toupper(cfg$sampling), cfg$top_n),
      x        = "Efecto sobre Pr(brecha = 1)",
      y        = NULL,
      caption  = sprintf("AUC = %.3f | McFadden R² = %.3f | n = %d",
                          metrics$auc, metrics$mcfadden, ld$n)
    ) +
    theme_gfp()

  path <- file.path(dir_out, sprintf("plot_ame_top%d.png", cfg$top_n))
  ggsave(path, plot = p, width = 9, height = 5.5, dpi = 300)
  log(sprintf("Guardado: %s", path))
  p
}

plot_coef <- function(fit, cfg, dir_out) {
  # Usar IC de Wald (confint.default) en lugar de profile likelihood
  # Profile likelihood falla en modelos grandes con muchas variables
  ci      <- as.data.frame(confint.default(fit$logit))
  colnames(ci) <- c("conf.low", "conf.high")

  coef_df <- tidy(fit$logit) %>%
    filter(term != "(Intercept)") %>%
    bind_cols(ci[rownames(ci) != "(Intercept)", , drop = FALSE]) %>%
    mutate(
      label     = shorten_label(term),
      categoria = case_when(
        p.value >= 0.05 ~ "n.s.",
        estimate > 0    ~ "Positivo sig.",
        TRUE             ~ "Negativo sig."
      ),
      label = factor(label, levels = label[order(estimate)])
    )

  p <- ggplot(coef_df, aes(x = estimate, y = label, color = categoria)) +
    geom_vline(xintercept = 0, linetype = "dashed",
               color = "gray60", linewidth = 0.6) +
    geom_errorbarh(aes(xmin = conf.low, xmax = conf.high),
                   height = 0.3, linewidth = 0.7, alpha = 0.8) +
    geom_point(size = 3.5) +
    scale_color_manual(values = COLORES) +
    labs(
      title    = "Coeficientes Logit (log-odds)",
      subtitle = sprintf("%s | XGB %s | Top %d vars SHAP",
                         toupper(cfg$modalidad), toupper(cfg$sampling), cfg$top_n),
      x        = "Log-odds",
      y        = NULL
    ) +
    theme_gfp()

  path <- file.path(dir_out, sprintf("plot_coef_logit_top%d.png", cfg$top_n))
  ggsave(path, plot = p, width = 9, height = 5.5, dpi = 300)
  log(sprintf("Guardado: %s", path))
  p
}

plot_roc <- function(metrics, cfg, dir_out) {
  roc_df <- data.frame(
    fpr = 1 - metrics$roc$specificities,
    tpr = metrics$roc$sensitivities
  )
  p <- ggplot(roc_df, aes(x = fpr, y = tpr)) +
    geom_abline(slope = 1, intercept = 0,
                linetype = "dashed", color = "gray70") +
    geom_line(color = "#2c7bb6", linewidth = 1.1) +
    geom_area(fill = "#2c7bb6", alpha = 0.08) +
    annotate("text", x = 0.65, y = 0.15,
             label  = sprintf("AUC = %.3f", metrics$auc),
             size   = 4.5, color = "#2c7bb6", fontface = "bold") +
    scale_x_continuous(labels = percent_format()) +
    scale_y_continuous(labels = percent_format()) +
    labs(
      title    = "Curva ROC — Logit",
      subtitle = sprintf("%s | XGB %s | Top %d vars",
                         toupper(cfg$modalidad), toupper(cfg$sampling), cfg$top_n),
      x        = "Tasa de Falsos Positivos (1 - Especificidad)",
      y        = "Sensibilidad"
    ) +
    theme_gfp()

  path <- file.path(dir_out, sprintf("plot_roc_logit_top%d.png", cfg$top_n))
  ggsave(path, plot = p, width = 6, height = 5.5, dpi = 300)
  log(sprintf("Guardado: %s", path))
  p
}

plot_ame_comparativo <- function(ame_all) {
  vars_comunes <- ame_all %>%
    group_by(var_original) %>%
    summarise(n_mod = n_distinct(modalidad), .groups = "drop") %>%
    filter(n_mod >= 2) %>%
    pull(var_original)

  if (length(vars_comunes) == 0) {
    log("No hay variables comunes entre modalidades — omitiendo comparativo.", level = "WARN")
    return(invisible(NULL))
  }

  p <- ame_all %>%
    filter(var_original %in% vars_comunes) %>%
    mutate(label = shorten_label(var_original)) %>%
    ggplot(aes(x = AME, y = reorder(label, AME),
               color = categoria, shape = modalidad)) +
    geom_vline(xintercept = 0, linetype = "dashed",
               color = "gray60", linewidth = 0.5) +
    geom_point(size = 3.5, position = position_dodge(width = 0.5)) +
    scale_color_manual(values = COLORES) +
    scale_shape_manual(values = c(16, 17, 15)) +
    scale_x_continuous(labels = percent_format(accuracy = 1)) +
    labs(
      title    = "AME comparativo — variables en ≥ 2 modalidades",
      subtitle = "Cada punto = modalidad  |  posición = efecto marginal promedio",
      x        = "AME sobre Pr(brecha = 1)",
      y        = NULL,
      shape    = "Modalidad"
    ) +
    theme_gfp()

  path <- file.path(DIR_LOGS, "comparativo_ame.png")
  ggsave(path, plot = p, width = 10, height = 6, dpi = 300)
  log(sprintf("Guardado: %s", path))
  p
}

plot_heatmap <- function(ame_all) {
  all_labels <- ame_all %>%
    group_by(var_original) %>%
    summarise(max_abs = max(abs(AME)), .groups = "drop") %>%
    arrange(desc(max_abs)) %>%
    pull(var_original)

  p <- ame_all %>%
    mutate(
      label = factor(shorten_label(var_original),
                     levels = rev(shorten_label(all_labels))),
      asterisco = ifelse(p < 0.05, "*", "")
    ) %>%
    ggplot(aes(x = modalidad, y = label, fill = AME)) +
    geom_tile(color = "white", linewidth = 0.5) +
    geom_text(aes(label = sprintf("%.3f%s", AME, asterisco)),
              size = 3.2, color = "white", fontface = "bold") +
    scale_fill_gradient2(low = "#d7191c", mid = "white", high = "#2c7bb6",
                          midpoint = 0, name = "AME") +
    labs(
      title    = "Efectos Marginales Promedio — Heatmap por modalidad",
      subtitle = "* = p < 0.05  |  Azul = aumenta brecha  |  Rojo = reduce brecha",
      x        = NULL,
      y        = NULL
    ) +
    theme_gfp() +
    theme(axis.text.x  = element_text(face = "bold", size = 12),
          panel.grid   = element_blank())

  path <- file.path(DIR_LOGS, "heatmap_ame.png")
  ggsave(path, plot = p,
         width  = 10,
         height = max(5, length(all_labels) * 0.45),
         dpi    = 300)
  log(sprintf("Guardado: %s", path))
  p
}

# =============================================================================
# 4. PIPELINE PRINCIPAL
# =============================================================================

RESULTS <- list()

for (cfg in CONFIGS) {
  mod <- cfg$modalidad
  log_section(sprintf("MODALIDAD: %s", toupper(mod)))

  # ── Carga ──────────────────────────────────────────────────────────────────
  err_msg <- NULL
  ld <- tryCatch(
    load_modality(cfg),
    error = function(e) { err_msg <<- e$message; NULL }
  )
  if (is.null(ld)) {
    log(sprintf("ERROR al cargar: %s", err_msg), level = "ERROR")
    RESULTS[[mod]] <- list(error = TRUE, msg = err_msg)
    next
  }
  log(sprintf("Data cargada: %d obs | %d features | brecha=1: %.1f%%",
              ld$n, length(ld$top_features), 100 * ld$n1 / ld$n))
  log(sprintf("Top-%d features (SHAP):", cfg$top_n))
  for (i in seq_along(ld$top_features))
    log(sprintf("  %2d. %s", i, ld$top_features[i]))

  # ── Modelos ────────────────────────────────────────────────────────────────
  log("Ajustando Logit + LPM...")
  fit <- tryCatch(
    fit_models(ld),
    error = function(e) { err_msg <<- e$message; NULL }
  )
  if (is.null(fit)) {
    log(sprintf("ERROR en modelos: %s", err_msg), level = "ERROR")
    RESULTS[[mod]] <- list(error = TRUE, msg = err_msg); next
  }

  # ── Métricas ───────────────────────────────────────────────────────────────
  log("Calculando métricas...")
  metrics <- fit_metrics(fit$logit, ld$data)
  log(sprintf("AUC=%.4f | McFadden R2=%.4f | Exactitud=%.1f%%",
              metrics$auc, metrics$mcfadden, 100 * metrics$accuracy))

  # ── AME ────────────────────────────────────────────────────────────────────
  log("Calculando AME (puede tardar)...")
  ame_df <- tryCatch(
    compute_ame(fit$logit, ld$inv_map),
    error = function(e) { err_msg <<- e$message; NULL }
  )
  if (is.null(ame_df)) {
    log(sprintf("ERROR en AME: %s", err_msg), level = "ERROR")
    RESULTS[[mod]] <- list(error = TRUE, msg = err_msg); next
  }

  sig_vars <- ame_df[ame_df$p < 0.05, ]
  log(sprintf("Variables significativas (p<0.05): %d de %d",
              nrow(sig_vars), nrow(ame_df)))

  # ── Log tabla AME ──────────────────────────────────────────────────────────
  ame_display <- ame_df %>%
    select(label, AME, SE, p, sig, categoria) %>%
    mutate(AME = round(AME, 4), SE = round(SE, 4), p = round(p, 4))
  log_table(ame_display, caption = "Tabla AME completa:")

  # ── Hallazgos ──────────────────────────────────────────────────────────────
  findings <- generate_findings(ame_df, metrics, ld, cfg)
  log_section(sprintf("HALLAZGOS — %s", toupper(mod)))
  for (line in findings) log(line)

  # ── Outputs ────────────────────────────────────────────────────────────────
  dir_out <- file.path(BASE_DIR, "outputs/regresiones", mod)
  dir.create(dir_out, showWarnings = FALSE, recursive = TRUE)

  # Excel AME
  xlsx_path <- file.path(dir_out,
    sprintf("ame_logit_xgb%s_top%d.xlsx", cfg$sampling, cfg$top_n))
  export(ame_df, xlsx_path)
  log(sprintf("Guardado: %s", xlsx_path))

  # Resumen logit al log
  coef_txt <- capture.output(summary(fit$logit))
  cat("\n--- SUMMARY LOGIT ---\n", file = log_con)
  for (l in coef_txt) cat(l, "\n", file = log_con)

  # Gráficos — cada uno en tryCatch para que un fallo no detenga el pipeline
  log("Generando gráficos...")
  tryCatch(plot_ame(ame_df, metrics, ld, cfg, dir_out),
           error = function(e) log(sprintf("plot_ame falló: %s", e$message), level = "WARN"))
  tryCatch(plot_coef(fit, cfg, dir_out),
           error = function(e) log(sprintf("plot_coef falló: %s", e$message), level = "WARN"))
  tryCatch(plot_roc(metrics, cfg, dir_out),
           error = function(e) log(sprintf("plot_roc falló: %s", e$message), level = "WARN"))

  # Guardar resultado
  RESULTS[[mod]] <- list(
    cfg      = cfg,
    ld       = ld,
    fit      = fit,
    metrics  = metrics,
    ame_df   = ame_df,
    findings = findings,
    dir_out  = dir_out,
    error    = FALSE
  )

  log(sprintf("[%s] COMPLETADO.", toupper(mod)))
}

# =============================================================================
# 5. COMPARATIVO ENTRE MODALIDADES
# =============================================================================

log_section("COMPARATIVO ENTRE MODALIDADES")

res_ok <- Filter(function(r) !isTRUE(r$error), RESULTS)

if (length(res_ok) >= 2) {

  # Tabla resumen métricas
  metricas_df <- lapply(names(res_ok), function(mod) {
    r <- res_ok[[mod]]
    data.frame(
      Modalidad  = toupper(mod),
      Sampling   = toupper(r$cfg$sampling),
      N          = r$ld$n,
      Brecha_pct = round(100 * r$ld$n1 / r$ld$n, 1),
      AUC        = r$metrics$auc,
      McFadden   = r$metrics$mcfadden,
      Exactitud  = round(100 * r$metrics$accuracy, 1),
      Vars_sig   = sum(r$ame_df$p < 0.05),
      check.names = FALSE
    )
  }) %>% bind_rows()

  log_table(metricas_df, caption = "Métricas globales por modalidad:")

  # Pool AME para comparativos
  ame_all <- lapply(names(res_ok), function(mod) {
    res_ok[[mod]]$ame_df %>%
      mutate(modalidad = toupper(mod)) %>%
      select(var_original, label, AME, p, categoria, modalidad)
  }) %>% bind_rows()

  log("Generando gráfico comparativo AME...")
  plot_ame_comparativo(ame_all)

  log("Generando heatmap AME...")
  plot_heatmap(ame_all)

  # Patrones transversales
  consistentes <- ame_all %>%
    group_by(var_original) %>%
    summarise(
      n_sig     = sum(p < 0.05),
      n_pos_sig = sum(p < 0.05 & AME > 0),
      n_neg_sig = sum(p < 0.05 & AME < 0),
      .groups   = "drop"
    ) %>%
    filter(n_sig >= 2) %>%
    arrange(desc(n_sig))

  log_section("CONCLUSIONES — PATRONES TRANSVERSALES")

  if (nrow(consistentes) > 0) {
    for (i in seq_len(nrow(consistentes))) {
      v   <- consistentes[i, ]
      lbl <- shorten_label(v$var_original)
      if (v$n_pos_sig == v$n_sig)
        log(sprintf("(+) '%s' aumenta brecha consistentemente en %d modalidades.",
                    lbl, v$n_sig))
      else if (v$n_neg_sig == v$n_sig)
        log(sprintf("(-) '%s' reduce brecha consistentemente en %d modalidades.",
                    lbl, v$n_sig))
      else
        log(sprintf("(~) '%s' significativa en %d modalidades con dirección mixta.",
                    lbl, v$n_sig))
    }
  } else {
    log("No se encontraron patrones transversales (variables sig. en >=2 modelos).",
        level = "WARN")
  }

} else {
  log("Menos de 2 modalidades completadas — omitiendo comparativo.", level = "WARN")
}

# =============================================================================
# 6. RESUMEN FINAL
# =============================================================================

log_section("RESUMEN FINAL")
log(sprintf("Fin: %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S")))

for (mod in names(RESULTS)) {
  r <- RESULTS[[mod]]
  if (isTRUE(r$error)) {
    log(sprintf("[%s] FALLIDO — %s", toupper(mod), r$msg), level = "ERROR")
  } else {
    log(sprintf("[%s] OK | AUC=%.3f | McFadden=%.3f | %d/%d sig.",
                toupper(mod), r$metrics$auc, r$metrics$mcfadden,
                sum(r$ame_df$p < 0.05), nrow(r$ame_df)))
  }
}

log(sprintf("Log guardado en: %s", LOG_FILE))
log("Pipeline finalizado.")
