"""
Entrenamiento y evaluación del modelo XGBoost para WODPredict-AI — Fase 3.

Flujo:
  1. Cargar DatasetBundle (ingestion)
  2. Construir FeatureMatrix (features)
  3. Split temporal train / val / test (80 / 10 / 10 por fecha)
  4. Entrenar XGBClassifier con scale_pos_weight para desbalanceo
  5. Evaluar: ROC-AUC, PR-AUC, F1, classification report
  6. Persistir modelo + medianas de imputación en model/artifacts/

Uso como módulo:
    from model.trainer import train
    result = train()

Uso directo:
    python -m model.trainer
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from features.builder import FeatureMatrix, MODEL_FEATURES, build_features
from ingestion.loader import load_dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rutas de artefactos
# ---------------------------------------------------------------------------

from config import ARTIFACTS_DIR, MODEL_PATH, MEDIANS_PATH  # noqa: F401 — reexportadas para compatibilidad

# ---------------------------------------------------------------------------
# Hiperparámetros por defecto
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {
    "n_estimators":     400,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "eval_metric":      "aucpr",
    "early_stopping_rounds": 30,
    "random_state":     42,
    "n_jobs":           -1,
    "verbosity":        0,
}

# ---------------------------------------------------------------------------
# Resultado del entrenamiento
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    """Métricas y artefactos del entrenamiento."""

    roc_auc_val:  float
    pr_auc_val:   float
    f1_val:       float
    roc_auc_test: float
    pr_auc_test:  float
    f1_test:      float
    n_train:      int
    n_val:        int
    n_test:       int
    best_iteration: int
    model_path:   Path
    medians_path: Path
    feature_importances: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=== Resultados del modelo ===",
            f"  Train / Val / Test : {self.n_train:,} / {self.n_val:,} / {self.n_test:,} filas",
            f"  Mejor iteración    : {self.best_iteration}",
            "",
            "  Validación:",
            f"    ROC-AUC : {self.roc_auc_val:.4f}",
            f"    PR-AUC  : {self.pr_auc_val:.4f}",
            f"    F1      : {self.f1_val:.4f}",
            "",
            "  Test:",
            f"    ROC-AUC : {self.roc_auc_test:.4f}",
            f"    PR-AUC  : {self.pr_auc_test:.4f}",
            f"    F1      : {self.f1_test:.4f}",
            "",
            f"  Modelo guardado en : {self.model_path}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Split temporal
# ---------------------------------------------------------------------------

def _temporal_split(
    fm: FeatureMatrix,
    sessions_feat: pd.DataFrame,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
) -> tuple[
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
]:
    """
    Divide X e y por orden cronológico de la sesión.
    Usa la columna 'date' de sessions_feat para ordenar antes de cortar.
    Los índices de fm.X se corresponden con sessions_feat por posición
    (ambos pasan por el mismo merge + reset_index en build_features).
    """
    # Recuperar fechas en el mismo orden que fm.X
    # fm.X ya tiene reset_index; alineamos por posición al sessions_feat filtrado
    dates = sessions_feat["date"].values  # misma longitud que fm.X

    sorted_idx = np.argsort(dates, kind="stable")
    n = len(sorted_idx)
    n_test = max(1, int(n * test_frac))
    n_val  = max(1, int(n * val_frac))
    n_train = n - n_val - n_test

    train_idx = sorted_idx[:n_train]
    val_idx   = sorted_idx[n_train: n_train + n_val]
    test_idx  = sorted_idx[n_train + n_val:]

    X, y = fm.X, fm.y
    return (
        X.iloc[train_idx], y.iloc[train_idx],
        X.iloc[val_idx],   y.iloc[val_idx],
        X.iloc[test_idx],  y.iloc[test_idx],
    )


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def _evaluate(model: XGBClassifier, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc":  float(average_precision_score(y, proba)),
        "f1":      float(f1_score(y, pred, zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Entrenamiento principal
# ---------------------------------------------------------------------------

def train(
    token: str | None = None,
    params: dict | None = None,
    save: bool = True,
) -> TrainResult:
    """
    Ejecuta el pipeline completo: ingesta → features → train → eval → save.

    Args:
        token:  HuggingFace token opcional.
        params: Hiperparámetros XGBoost. Si None usa DEFAULT_PARAMS.
        save:   Si True, persiste modelo y medianas en model/artifacts/.

    Returns:
        TrainResult con métricas y rutas de artefactos.
    """
    xgb_params = {**DEFAULT_PARAMS, **(params or {})}

    # 1. Ingesta
    logger.info("Cargando dataset...")
    bundle = load_dataset(token=token, validate=False)

    # 2. Features
    logger.info("Construyendo features...")
    fm = build_features(bundle)

    # Obtener fechas alineadas (post-merge, post-filtro de y=NaN)
    # build_features hace merge(sessions_feat, outcomes) y luego mask_valid
    # Necesitamos replicar esos índices para extraer las fechas
    outcomes_label = bundle.sessions_outcomes[["session_id", "overreaching_label"]]
    df_merged = bundle.sessions_feat.merge(outcomes_label, on="session_id", how="inner")
    mask_valid = df_merged["overreaching_label"].notna()
    dates_aligned = df_merged.loc[mask_valid, "date"].reset_index(drop=True)

    # 3. Split temporal
    logger.info("Dividiendo train / val / test por fecha...")
    X_train, y_train, X_val, y_val, X_test, y_test = _temporal_split(
        fm, pd.DataFrame({"date": dates_aligned})
    )
    logger.info(
        "Split: train=%d | val=%d | test=%d",
        len(X_train), len(X_val), len(X_test),
    )

    # 4. scale_pos_weight para desbalanceo de clases
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = neg / max(pos, 1)
    logger.info("Positivos=%.1f%% → scale_pos_weight=%.2f", 100 * pos / (pos + neg), spw)

    # 5. Entrenamiento con early stopping
    early_stop = xgb_params.pop("early_stopping_rounds", 30)
    model = XGBClassifier(
        scale_pos_weight=spw,
        **xgb_params,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    best_iter = int(model.best_iteration) if hasattr(model, "best_iteration") else xgb_params.get("n_estimators", 400)
    logger.info("Mejor iteración: %d", best_iter)

    # 6. Evaluación
    val_metrics  = _evaluate(model, X_val,  y_val)
    test_metrics = _evaluate(model, X_test, y_test)
    logger.info(
        "Val  → ROC-AUC=%.4f | PR-AUC=%.4f | F1=%.4f",
        val_metrics["roc_auc"], val_metrics["pr_auc"], val_metrics["f1"],
    )
    logger.info(
        "Test → ROC-AUC=%.4f | PR-AUC=%.4f | F1=%.4f",
        test_metrics["roc_auc"], test_metrics["pr_auc"], test_metrics["f1"],
    )
    logger.info("\n%s", classification_report(y_test, (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)))

    # 7. Importancias
    importances = dict(zip(MODEL_FEATURES, model.feature_importances_.tolist()))

    # 8. Persistencia
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if save:
        model.save_model(str(MODEL_PATH))
        MEDIANS_PATH.write_text(json.dumps(fm.imputation_medians, indent=2), encoding="utf-8")
        logger.info("Artefactos guardados en %s", ARTIFACTS_DIR)

    return TrainResult(
        roc_auc_val=val_metrics["roc_auc"],
        pr_auc_val=val_metrics["pr_auc"],
        f1_val=val_metrics["f1"],
        roc_auc_test=test_metrics["roc_auc"],
        pr_auc_test=test_metrics["pr_auc"],
        f1_test=test_metrics["f1"],
        n_train=len(X_train),
        n_val=len(X_val),
        n_test=len(X_test),
        best_iteration=best_iter,
        model_path=MODEL_PATH,
        medians_path=MEDIANS_PATH,
        feature_importances=importances,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = train()
    print(result.summary())
    print("\nTop-10 features por importancia:")
    top10 = sorted(result.feature_importances.items(), key=lambda x: x[1], reverse=True)[:10]
    for name, score in top10:
        print(f"  {score:.4f}  {name}")
