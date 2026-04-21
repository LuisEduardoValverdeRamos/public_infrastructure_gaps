"""
gfp_utils.py — Funciones reutilizables para el pipeline de modelado GFP.

Cubre: evaluación de modelos, tabla de resultados, curvas ROC,
feature importance y grid search para RF y XGB.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    classification_report,
    RocCurveDisplay,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBClassifier


# ── Evaluación ────────────────────────────────────────────────────────────────

def evaluate_model(model, X, y):
    """
    Evalúa un modelo clasificador y devuelve un dict con todas las métricas.

    Métricas: Accuracy, ROC-AUC, F1 macro, MCC, Log-Loss,
              Precision/Recall/F1 por clase (no / si).

    Parameters
    ----------
    model : clasificador sklearn/xgb ya entrenado
    X     : features (array-like)
    y     : etiquetas reales (0/1)

    Returns
    -------
    dict con las métricas
    """
    y_pred_class = model.predict(X)
    y_pred_prob  = model.predict_proba(X)[:, 1]

    report = classification_report(
        y, y_pred_class, target_names=['no', 'si'], output_dict=True
    )

    return {
        'accuracy':       accuracy_score(y, y_pred_class),
        'roc_auc':        roc_auc_score(y, y_pred_prob),
        'f1_macro':       f1_score(y, y_pred_class, average='macro'),
        'mcc':            matthews_corrcoef(y, y_pred_class),
        'log_loss':       log_loss(y, y_pred_class),
        'no_precision':   report['no']['precision'],
        'no_recall':      report['no']['recall'],
        'no_f1':          report['no']['f1-score'],
        'si_precision':   report['si']['precision'],
        'si_recall':      report['si']['recall'],
        'si_f1':          report['si']['f1-score'],
    }


# ── Tabla de resultados ───────────────────────────────────────────────────────

def build_results_table(results):
    """
    Construye la tabla de comparación de modelos.

    Parameters
    ----------
    results : list de dicts con keys:
        'model'    : nombre del modelo (str)
        'sampling' : estrategia de resampling (str)
        'metrics'  : dict devuelto por evaluate_model()

    Returns
    -------
    pd.DataFrame con índice = 'model + sampling', columnas = métricas
    """
    col_map = {
        'accuracy':     'Overall_Accuracy',
        'roc_auc':      'Roc_Auc',
        'f1_macro':     'Global_F1_Score',
        'mcc':          'Matthews_Corr_Coef',
        'no_precision': 'No_Precision',
        'no_recall':    'No_Recall',
        'no_f1':        'No_F1_Score',
        'si_precision': 'Si_Precision',
        'si_recall':    'Si_Recall',
        'si_f1':        'Si_F1_Score',
    }

    rows = []
    index = []

    for r in results:
        row = {col_map[k]: v for k, v in r['metrics'].items() if k in col_map}
        rows.append(row)
        index.append(f"{r['sampling'].upper()}. {r['model']}")

    return pd.DataFrame(rows, index=index).round(3)


# ── Curvas ROC ────────────────────────────────────────────────────────────────

def plot_roc_curves(models_dict, X, y, title, filepath):
    """
    Genera y guarda una figura con las curvas ROC de varios modelos.

    Parameters
    ----------
    models_dict : dict  {label: modelo_entrenado}
                  Ejemplo: {'O. RF': rf_o, 'S. RF': rf_s}
    X           : features del set a evaluar
    y           : etiquetas reales
    title       : título del gráfico
    filepath    : ruta donde guardar la imagen (.jpg / .png)
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for label, model in models_dict.items():
        RocCurveDisplay.from_estimator(model, X, y, ax=ax, name=label)

    ax.plot([0, 1], [0, 1], 'k--', label='Random chance')
    ax.set_title(title)
    ax.legend(loc='lower right')

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'ROC guardada: {filepath}')


# ── Feature importance ────────────────────────────────────────────────────────

def get_feature_importance(model, pred_vars, top_n=50):
    """
    Extrae e importancia de features para RF o XGB.

    Parameters
    ----------
    model     : modelo entrenado con .feature_importances_
    pred_vars : lista de nombres de variables predictoras
    top_n     : cuántas variables mostrar (default 50)

    Returns
    -------
    pd.DataFrame con columnas ['vars', 'score'], ordenado desc.
    """
    return (
        pd.Series(model.feature_importances_, index=pred_vars)
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
        .rename(columns={'index': 'vars', 0: 'score'})
    )


# ── Grid Search ───────────────────────────────────────────────────────────────

def grid_search_rf(X, y, param_grid=None, cv=5, random_state=2023):
    """
    Grid search para Random Forest.

    Parameters
    ----------
    X          : features de entrenamiento
    y          : etiquetas de entrenamiento
    param_grid : dict con hiperparámetros a explorar.
                 Si es None, usa el grid por defecto del proyecto.
    cv         : número de folds (default 5)
    random_state : semilla aleatoria

    Returns
    -------
    GridSearchCV ya ajustado
    """
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 250, 500],
            'max_features': [60, 90, 120],
            'max_depth':    [10, 20, 30],
        }

    model  = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    kfold  = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kfold, n_jobs=-1)
    search.fit(X, y)
    return search


def grid_search_xgb(X, y, param_grid=None, cv=5, random_state=2023):
    """
    Grid search para XGBoost.

    Parameters
    ----------
    X          : features de entrenamiento
    y          : etiquetas de entrenamiento
    param_grid : dict con hiperparámetros a explorar.
                 Si es None, usa el grid por defecto del proyecto.
    cv         : número de folds (default 5)
    random_state : semilla aleatoria

    Returns
    -------
    GridSearchCV ya ajustado
    """
    if param_grid is None:
        param_grid = {
            'n_estimators':    [100, 250, 500],
            'colsample_bytree':[0.2, 0.3, 0.4],
            'max_depth':       [2, 4, 6],
            'learning_rate':   [0.05, 0.1],
            'subsample':       [0.8, 1.0],
        }

    model = XGBClassifier(
        objective='binary:logistic',
        verbosity=0,
        random_state=random_state,
        n_jobs=-1,
    )
    kfold  = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kfold, n_jobs=-1)
    search.fit(X, y)
    return search


# ── Guardado de modelos ───────────────────────────────────────────────────────

def save_models(models_dict, output_dir):
    """
    Guarda un dict de modelos como archivos .joblib.

    Parameters
    ----------
    models_dict : dict  {filename_sin_extension: modelo}
                  Ejemplo: {'rf_o': rf_optimal_model_o}
    output_dir  : carpeta de destino
    """
    os.makedirs(output_dir, exist_ok=True)
    for name, model in models_dict.items():
        path = os.path.join(output_dir, f'{name}.joblib')
        joblib.dump(model, path)
    print(f'{len(models_dict)} modelos guardados en: {output_dir}')


def save_grid_search_results(searches_dict, output_dir):
    """
    Exporta los cv_results_ de cada GridSearchCV a Excel.

    Parameters
    ----------
    searches_dict : dict  {filename_sin_extension: GridSearchCV}
    output_dir    : carpeta de destino
    """
    os.makedirs(output_dir, exist_ok=True)
    for name, search in searches_dict.items():
        path = os.path.join(output_dir, f'{name}.xlsx')
        pd.DataFrame(search.cv_results_).to_excel(path, index=False)
    print(f'Grid search results guardados en: {output_dir}')
