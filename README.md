# GFP — Predicting Implementation Gaps in Public Infrastructure (Peru)

Machine learning pipeline to predict whether public infrastructure projects (*obras*) in Peru will have a delay between their programmed and real completion date (**implementation gap / brecha de implementación**), and interpret the drivers of that gap through logistic regression and average marginal effects.

**Data source:** [Infobras](https://www.infobras.gob.pe/) — Contraloría General del Perú.  
**Universe:** Gobiernos Locales (Local Governments), 2018–2024.

🔍 **[Live demo — Predictor de Brechas](https://publicinfrastructuregaps-tmw6bbr3errm5rwsqdxa75.streamlit.app/)**  
🌐 **[Project site (GitHub Pages)](https://luiseduardovalverderamos.github.io/public_infrastructure_gaps/)**

---

## Models

Three separate models, one per execution modality:

| Modality | Filter | N | Brecha = 1 | Features |
|---|---|---|---|---|
| **Contrata** | `modalidad_ejecucion == 'Contrata'` | 20,593 | 53.5% | 241 |
| **Administración Directa (AD)** | `modalidad_ejecucion == 'Administración directa'` | 17,111 | 60.6% | 257 |
| **ARCC** | `marca_reconstruccion == 'Si'` (cross-modality) | 2,539 | 57.8% | 106 |

**Algorithms:** Logistic Regression, Lasso, Ridge, Elastic Net, Random Forest, XGBoost  
**Resampling strategies:** Original (O), SMOTE (S), SMOTE-Tomek (ST), Naive Random Sampling (NRS)

---

## Key Results

### ML Model Performance (XGBoost — best per modality, test set)

| Modality | Best sampling | F1 Score | AUC-ROC | Accuracy |
|---|---|---|---|---|
| Contrata | NRS | 0.799 | 0.872 | 79.9% |
| AD | Original | 0.754 | 0.845 | 76.2% |
| ARCC | Original | 0.736 | 0.821 | 74.0% |

### Logistic Regression — Average Marginal Effects (Top 10 SHAP features)

| Modality | AUC | McFadden R² | Accuracy |
|---|---|---|---|
| Contrata | 0.852 | 0.325 | 77.3% |
| AD | 0.826 | 0.260 | 73.9% |
| ARCC | 0.843 | 0.307 | 76.1% |

### Main Findings

**Variables with consistent positive effects across modalities (increase brecha probability):**

- `n_modificaciones` — strongest predictor in all three modalities (AME: +44.8 pp Contrata, +43.4 pp AD, +45.1 pp ARCC). Each additional contract modification substantially raises the probability of delay.
- `n_adicionales_obra` — additional work orders consistently increase delay probability (+5.7 pp, +5.0 pp, +3.0 pp).
- `n_informes_control` — more control reports signal more problems (+5.4 pp, +7.8 pp, +2.3 pp).
- `log_monto_aprobado` — larger projects are more likely to face delays (+3.5 pp, +3.2 pp, +5.9 pp).

**Notable effects by modality:**

| Modality | Variable | AME | Direction |
|---|---|---|---|
| Contrata | `existe_paralizacion` | +26.9 pp | Project stoppages are a strong predictor of delay |
| Contrata | `Region_sierra sur` | +4.4 pp | Geographic effect: sierra sur has higher delay rates |
| Contrata | `Region_sierra centro/norte` | +4.1 pp | Same geographic pattern |
| AD | `tipo_obra_Transporte Terrestre` | −5.9 pp | Road infrastructure less likely to delay in AD |
| AD | `anio_inicio_obra` | −2.6 pp | More recent projects show lower delay probability |
| ARCC | `Region_costa norte` | −6.9 pp | Costa norte projects less likely to delay in ARCC |
| ARCC | `tipo_obra_Educación/Cultura` | +6.0 pp | Education/culture works more likely to delay |

---

## Project Structure

```
C:/15_GFP/
│
├── data/
│   ├── raw/                                  # Raw data from Infobras (not in repo)
│   ├── dictionaries/
│   │   ├── var_names_of_proyectos_ejecucion.xlsx
│   │   └── macro_zona.xlsx
│   └── processed/                            # Cleaned outputs (not in repo)
│       ├── contrata/1_data_contrata.xlsx
│       ├── ad/1_data_ad.xlsx
│       └── arcc/1_data_arcc.xlsx
│
├── notebooks/
│   ├── 01_cleaning/
│   │   └── 01_cleaning_pipeline.ipynb        # Unified cleaning — set MODALIDAD
│   ├── 02_modeling/
│   │   └── 02_modeling_pipeline.ipynb        # Unified modeling — set MODALIDAD
│   ├── 03_shap/
│   │   ├── 03_shap_contrata.ipynb            # SHAP — Contrata (XGB NRS)
│   │   ├── 03_shap_ad.ipynb                  # SHAP — AD (XGB O)
│   │   └── 03_shap_arcc.ipynb                # SHAP — ARCC (XGB O)
│   └── 03_renamu/
│       ├── 01_renamu_cleaning.ipynb
│       └── 02_renamu_merge_contrata.ipynb
│
├── outputs/
│   ├── models/{contrata,ad,arcc}/            # 24 .joblib files per modality
│   ├── results/{contrata,ad,arcc}/           # results_test/train.xlsx + gridsearch RF/XGB
│   ├── figures/{contrata,ad,arcc}/           # ROC curves (RF and XGBoost)
│   ├── feature_importance/{contrata,ad,arcc}/# Feature importance tables
│   ├── shap/{contrata,ad,arcc}/              # SHAP values .xlsx + beeswarm/bar plots
│   └── regresiones/{contrata,ad,arcc}/       # AME tables, coefficient plots, ROC curves
│       ├── comparativo_ame.png               # Cross-modality AME comparison
│       ├── heatmap_ame.png                   # AME heatmap across modalities
│       └── analysis_log_YYYYMMDD_HHMMSS.txt  # Full execution log
│
├── r_scripts/
│   ├── 04_analisis_regresion.R               # Main analysis: Logit + AME (all 3 modalities)
│   ├── RENAMU.Rmd
│   ├── Efectos_Marginales_Boosted.Rmd
│   └── preprocesamiento_sin_match.Rmd
│
├── src/
│   └── gfp_utils.py                          # Shared functions
│
├── archive/                                  # Old working versions (not in repo)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Pipeline

```
01_cleaning_pipeline.ipynb     MODALIDAD = 'contrata' | 'ad' | 'arcc'
          ↓
02_modeling_pipeline.ipynb     MODALIDAD = ...  →  outputs/models/
          ↓
03_shap_{modalidad}.ipynb      pre-configured per modality  →  outputs/shap/
          ↓
04_analisis_regresion.R        runs all 3 automatically  →  outputs/regresiones/
```

### How to Run

**1. Install Python dependencies**
```bash
pip install -r requirements.txt
```

**2. Place raw data**
Copy `data.xlsx` to `data/raw/data.xlsx` and dictionary files to `data/dictionaries/`.

**3. Run cleaning** — open `notebooks/01_cleaning/01_cleaning_pipeline.ipynb` and set:
```python
MODALIDAD = 'contrata'  # repeat for 'ad' and 'arcc'
```

**4. Run modeling** — open `notebooks/02_modeling/02_modeling_pipeline.ipynb` and set:
```python
MODALIDAD = 'contrata'  # repeat for 'ad' and 'arcc'
```

**5. Run SHAP** — open each pre-configured notebook (no changes needed):
```
notebooks/03_shap/03_shap_contrata.ipynb
notebooks/03_shap/03_shap_ad.ipynb
notebooks/03_shap/03_shap_arcc.ipynb
```

**6. Run regression analysis**
```bash
Rscript r_scripts/04_analisis_regresion.R
```
Outputs go to `outputs/regresiones/`. A timestamped log file is written automatically.

---

## Dependent Variable

**`brecha_existente`** (binary):
- `1` = project finished **after** programmed date (delay exists)
- `0` = project finished on time or early

Calculated as: `fecha_finalizacion_real − fecha_fin_programada > 0 days`

---

## Key Files

| File | Description |
|---|---|
| `src/gfp_utils.py` | `evaluate_model`, `build_results_table`, `plot_roc_curves`, `get_feature_importance`, `grid_search_rf`, `grid_search_xgb`, `save_models` |
| `data/dictionaries/var_names_of_proyectos_ejecucion.xlsx` | Variable metadata: names, types (NUM/DICO/POLI), include/exclude flags |
| `data/dictionaries/macro_zona.xlsx` | Province → geographic macro-region crosswalk |
| `outputs/regresiones/heatmap_ame.png` | AME heatmap across all three modalities |
| `outputs/regresiones/analysis_log_*.txt` | Full execution log with metrics and findings |
