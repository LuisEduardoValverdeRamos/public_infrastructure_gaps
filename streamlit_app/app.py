import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ─────────────────────────────────────────────────────────────────────────────
# Configuración de página
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GFP — Predictor de Brechas de Implementación",
    page_icon="🏗️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Estilos
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title   { font-size:2rem; font-weight:700; color:#1a3a5c; margin-bottom:0; }
    .sub-title    { font-size:1rem; color:#5a7a9a; margin-top:0; }
    .prob-box     { border-radius:12px; padding:24px; text-align:center; margin:16px 0; }
    .prob-high    { background:#fde8e8; border:2px solid #e53e3e; }
    .prob-med     { background:#fef3cd; border:2px solid #d69e2e; }
    .prob-low     { background:#e6f4ea; border:2px solid #38a169; }
    .prob-number  { font-size:3.5rem; font-weight:800; line-height:1; }
    .prob-label   { font-size:0.95rem; color:#555; margin-top:8px; }
    .factor-pos   { color:#c53030; }
    .factor-neg   { color:#276749; }
    .section-head { font-size:1.1rem; font-weight:600; color:#2d3748;
                    border-bottom:2px solid #e2e8f0; padding-bottom:6px; margin-top:24px; }
    .badge        { display:inline-block; padding:3px 10px; border-radius:999px;
                    font-size:0.8rem; font-weight:600; margin-right:6px; }
    .badge-blue   { background:#ebf4ff; color:#2b6cb0; }
    .badge-green  { background:#f0fff4; color:#276749; }
    .badge-orange { background:#fffaf0; color:#c05621; }
    .info-card    { background:#f7fafc; border-radius:8px; padding:14px 18px;
                    border-left:4px solid #4299e1; margin:10px 0; font-size:0.9rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Carga de modelos
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

@st.cache_resource
def load_model(modalidad: str):
    path = os.path.join(MODEL_DIR, f"{modalidad}.joblib")
    return joblib.load(path)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🏗️ GFP — Predictor de Brechas</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Estima la probabilidad de que una obra pública peruana '
    'termine después de su fecha programada.</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="info-card">Modelo entrenado con datos de <strong>Infobras (MEF Perú)</strong> '
    '— Gobiernos Locales 2018–2024. Los campos marcados con ★ son los de mayor impacto '
    'según el análisis SHAP.</div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Selección de modalidad
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-head">Modalidad de ejecución</p>', unsafe_allow_html=True)

MODALIDADES = {
    "Contrata":              ("contrata", "badge-blue"),
    "Administración Directa": ("ad",       "badge-green"),
    "ARCC (Reconstrucción)": ("arcc",     "badge-orange"),
}

modalidad_label = st.selectbox(
    "¿Cómo se ejecuta la obra?",
    list(MODALIDADES.keys()),
    help="Contrata = ejecutada por empresa privada | AD = ejecutada por la entidad pública | ARCC = Autoridad para la Reconstrucción con Cambios",
)
modalidad_key, badge_class = MODALIDADES[modalidad_label]

model = load_model(modalidad_key)
feature_names = model.get_booster().feature_names

# ─────────────────────────────────────────────────────────────────────────────
# Inputs comunes
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-head">Características de la obra</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    n_mod = st.number_input(
        "★ Nº de modificaciones al contrato",
        min_value=0, max_value=50, value=2, step=1,
        help="Cada modificación adicional aumenta significativamente la probabilidad de brecha."
    )
    n_adic = st.number_input(
        "★ Nº de adicionales de obra",
        min_value=0, max_value=30, value=0, step=1,
        help="Trabajos adicionales no contemplados en el contrato original."
    )
    n_inf = st.number_input(
        "★ Nº de informes de control",
        min_value=0, max_value=100, value=3, step=1,
        help="Informes de la OCI u organismos de control. Más informes indican mayor fiscalización."
    )

with col2:
    monto = st.number_input(
        "★ Monto aprobado (S/)",
        min_value=10_000, max_value=50_000_000, value=500_000, step=10_000,
        format="%d",
        help="Monto total aprobado en soles."
    )
    plazo = st.number_input(
        "Plazo de ejecución (días)",
        min_value=30, max_value=1825, value=180, step=30,
        help="Plazo contractual en días calendario."
    )

    REGIONES = {
        "Costa Sur (base)":         None,
        "Costa Norte":              "Region_costa norte",
        "Selva":                    "Region_selva",
        "Sierra Sur":               "Region_sierra sur",
        "Sierra Centro / Norte":    "Region_sierra centro/norte",
    }
    region_label = st.selectbox("Región geográfica", list(REGIONES.keys()))

# ─────────────────────────────────────────────────────────────────────────────
# Inputs específicos por modalidad
# ─────────────────────────────────────────────────────────────────────────────
paralizacion       = False
marca_recons       = False
es_transporte      = False
es_construccion    = False
es_educacion       = False

if modalidad_key == "contrata":
    st.markdown('<p class="section-head">Campos específicos — Contrata</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        paralizacion = st.checkbox(
            "★ ¿Hubo paralización de obra?",
            help="Si la obra fue paralizada en algún momento. Fuerte predictor de retraso."
        )
    with c2:
        marca_recons = st.checkbox(
            "¿Es obra de reconstrucción (ARCC)?",
            help="Obra que también tiene marca de reconstrucción con cambios."
        )

elif modalidad_key == "ad":
    st.markdown('<p class="section-head">Campos específicos — Administración Directa</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        es_transporte = st.checkbox(
            "¿Es obra de Transporte Terrestre?",
            help="Caminos vecinales, carreteras, trochas. Asociado a menor probabilidad de brecha en AD."
        )
    with c2:
        es_construccion = st.checkbox(
            "¿Es obra de construcción/creación?",
            value=True,
            help="Naturaleza de la obra: construcción o creación nueva."
        )

elif modalidad_key == "arcc":
    st.markdown('<p class="section-head">Campos específicos — ARCC</p>', unsafe_allow_html=True)
    es_educacion = st.checkbox(
        "¿Es obra de Educación o Cultura?",
        help="Colegios, institutos, centros culturales. Asociado a mayor probabilidad de brecha en ARCC."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Construcción del vector de features
# ─────────────────────────────────────────────────────────────────────────────
def build_feature_vector(feature_names):
    X = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)

    # Comunes
    if "n_modificaciones"    in X.columns: X["n_modificaciones"]    = n_mod
    if "log_monto_aprobado"  in X.columns: X["log_monto_aprobado"]  = np.log(max(monto, 1))
    if "n_adicionales_obra"  in X.columns: X["n_adicionales_obra"]  = n_adic
    if "n_informes_control"  in X.columns: X["n_informes_control"]  = n_inf
    if "plazo_ejecucion_dias" in X.columns: X["plazo_ejecucion_dias"] = plazo

    # Región
    region_col = REGIONES[region_label]
    if region_col and region_col in X.columns:
        X[region_col] = 1

    # Contrata
    if paralizacion    and "existe_paralizacion"   in X.columns: X["existe_paralizacion"]   = 1
    if marca_recons    and "marca_reconstruccion"  in X.columns: X["marca_reconstruccion"]  = 1

    # AD
    if es_transporte   and "tipo_obra_nivel2_Transporte Terrestre"   in X.columns:
        X["tipo_obra_nivel2_Transporte Terrestre"] = 1
    if es_construccion and "naturaleza_obra_Construcción/Creación"   in X.columns:
        X["naturaleza_obra_Construcción/Creación"] = 1

    # ARCC
    if es_educacion    and "tipo_obra_nivel1_Educación/Cultura"      in X.columns:
        X["tipo_obra_nivel1_Educación/Cultura"] = 1

    return X

# ─────────────────────────────────────────────────────────────────────────────
# Predicción
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
predict_btn = st.button("🔍 Calcular probabilidad de brecha", use_container_width=True, type="primary")

if predict_btn:
    X = build_feature_vector(feature_names)
    prob = model.predict_proba(X)[0][1]
    pct  = prob * 100

    # Color y etiqueta
    if pct >= 65:
        box_class = "prob-high"
        color     = "#e53e3e"
        riesgo    = "🔴 Riesgo ALTO"
        mensaje   = "Alta probabilidad de que la obra termine después de su fecha programada."
    elif pct >= 40:
        box_class = "prob-med"
        color     = "#d69e2e"
        riesgo    = "🟡 Riesgo MEDIO"
        mensaje   = "Probabilidad moderada de retraso. Monitoreo recomendado."
    else:
        box_class = "prob-low"
        color     = "#38a169"
        riesgo    = "🟢 Riesgo BAJO"
        mensaje   = "Baja probabilidad de retraso bajo las condiciones ingresadas."

    st.markdown(f"""
    <div class="prob-box {box_class}">
        <div class="prob-number" style="color:{color}">{pct:.1f}%</div>
        <div class="prob-label"><strong>{riesgo}</strong><br>{mensaje}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Factores que más influyeron ──────────────────────────────────────────
    st.markdown('<p class="section-head">Factores principales en esta predicción</p>',
                unsafe_allow_html=True)

    FACTOR_LABELS = {
        "n_modificaciones":                          "Modificaciones al contrato",
        "log_monto_aprobado":                        "Monto aprobado (log)",
        "n_adicionales_obra":                        "Adicionales de obra",
        "n_informes_control":                        "Informes de control",
        "plazo_ejecucion_dias":                      "Plazo de ejecución",
        "existe_paralizacion":                       "Paralización de obra",
        "marca_reconstruccion":                      "Marca reconstrucción",
        "Region_costa norte":                        "Región: Costa Norte",
        "Region_selva":                              "Región: Selva",
        "Region_sierra sur":                         "Región: Sierra Sur",
        "Region_sierra centro/norte":                "Región: Sierra Centro/Norte",
        "tipo_obra_nivel2_Transporte Terrestre":     "Tipo: Transporte Terrestre",
        "naturaleza_obra_Construcción/Creación":     "Naturaleza: Construcción",
        "tipo_obra_nivel1_Educación/Cultura":        "Tipo: Educación/Cultura",
    }

    # AME aproximados por modalidad (del análisis logit)
    AME_SIGNS = {
        "contrata": {
            "n_modificaciones": +0.448, "n_adicionales_obra": +0.057,
            "n_informes_control": +0.054, "log_monto_aprobado": +0.035,
            "existe_paralizacion": +0.269, "marca_reconstruccion": +0.034,
            "plazo_ejecucion_dias": +0.000, "Region_sierra sur": +0.044,
            "Region_sierra centro/norte": +0.041,
        },
        "ad": {
            "n_modificaciones": +0.434, "n_informes_control": +0.078,
            "Region_sierra sur": +0.072, "n_adicionales_obra": +0.050,
            "log_monto_aprobado": +0.032, "plazo_ejecucion_dias": -0.000,
            "tipo_obra_nivel2_Transporte Terrestre": -0.059,
            "naturaleza_obra_Construcción/Creación": +0.039,
        },
        "arcc": {
            "n_modificaciones": +0.451, "log_monto_aprobado": +0.059,
            "n_adicionales_obra": +0.030, "n_informes_control": +0.023,
            "plazo_ejecucion_dias": +0.000, "Region_costa norte": -0.070,
            "tipo_obra_nivel1_Educación/Cultura": +0.060,
        },
    }

    ame = AME_SIGNS.get(modalidad_key, {})
    user_vals = {
        "n_modificaciones":                       n_mod,
        "log_monto_aprobado":                     np.log(max(monto, 1)),
        "n_adicionales_obra":                     n_adic,
        "n_informes_control":                     n_inf,
        "plazo_ejecucion_dias":                   plazo,
        "existe_paralizacion":                    int(paralizacion),
        "marca_reconstruccion":                   int(marca_recons),
        "Region_costa norte":                     int(region_label == "Costa Norte"),
        "Region_selva":                           int(region_label == "Selva"),
        "Region_sierra sur":                      int(region_label == "Sierra Sur"),
        "Region_sierra centro/norte":             int(region_label == "Sierra Centro / Norte"),
        "tipo_obra_nivel2_Transporte Terrestre":  int(es_transporte),
        "naturaleza_obra_Construcción/Creación":  int(es_construccion),
        "tipo_obra_nivel1_Educación/Cultura":     int(es_educacion),
    }

    factors = []
    for feat, val in user_vals.items():
        if feat in ame and val != 0:
            contribution = ame[feat] * val
            label = FACTOR_LABELS.get(feat, feat)
            factors.append((label, contribution, ame[feat]))

    factors.sort(key=lambda x: abs(x[1]), reverse=True)

    if factors:
        for label, contrib, ame_val in factors[:6]:
            direction = "▲ aumenta" if contrib > 0 else "▼ reduce"
            css_class = "factor-pos" if contrib > 0 else "factor-neg"
            pp = abs(contrib * 100)
            st.markdown(
                f'<span class="{css_class}"><strong>{direction}</strong></span> '
                f'— **{label}**: contribuye aprox. **{pp:.1f} pp** '
                f'{"al riesgo" if contrib > 0 else "al reducir riesgo"}',
                unsafe_allow_html=True,
            )
    else:
        st.info("Con los valores ingresados, los factores de riesgo son mínimos.")

    # ── Nota metodológica ────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        f"**Modelo:** XGBoost ({modalidad_label}) | "
        f"**AUC-ROC:** {'0.872' if modalidad_key=='contrata' else '0.845' if modalidad_key=='ad' else '0.821'} | "
        f"**F1:** {'0.799' if modalidad_key=='contrata' else '0.754' if modalidad_key=='ad' else '0.736'} | "
        f"Entrenado con datos Infobras 2018–2024 · Gobiernos Locales · "
        f"[Ver código](https://github.com/LuisEduardoValverdeRamos/public_infrastructure_gaps)"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "**GFP — Implementation Gaps in Public Infrastructure** · "
    "Datos: [Infobras, MEF Perú](https://www.infobras.gob.pe/) · "
    "[Repositorio](https://github.com/LuisEduardoValverdeRamos/public_infrastructure_gaps)",
    unsafe_allow_html=False,
)
