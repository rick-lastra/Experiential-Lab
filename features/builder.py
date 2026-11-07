"""
Feature engineering para WODPredict-AI — Fase 2.

Toma el DatasetBundle (ya cargado y validado) y produce una FeatureMatrix
lista para entrenar el modelo XGBoost de overreaching.

Columnas de entrada esperadas (SESSION_FEAT_COLS de ingestion.loader):
  - Ventanas de carga: load_last_3d, load_last_7d, mean_session_load_28d
  - Ventanas de frecuencia: sessions_last_3d, sessions_last_7d, sessions_last_28d
  - ACWR: acwr3_freq, acwr3_load, acwr_freq, acwr_load, acwr_valid
  - Sesión anterior: prev_rpe, prev_session_load, prev_has_note
  - Señal de fatiga: prob_fatiga_prev_session, prob_fatiga_last_note
  - Temporales: days_since_prev_session, days_since_first_session,
                days_since_last_note, notes_last_28d
  - Perfil: level

Features derivadas que genera este módulo:
  - level_enc              int        Ordinal: novato=0, intermedio=1, elite=2
  - load_ratio_3_28        float      load_last_3d / (mean_session_load_28d + ε)
  - load_ratio_7_28        float      load_last_7d / (mean_session_load_28d×4 + ε)
  - freq_ratio_3_7         float      sessions_last_3d / (sessions_last_7d + ε)
  - fatigue_signal         float      media de las señales subjetivas de fatiga
  - days_since_note_capped float      days_since_last_note truncado a 28
  - acwr_load_risk         int        1 si acwr_load > 1.3
  - acwr_freq_risk         int        1 si acwr_freq > 1.3
  - prev_rpe_norm          float      prev_rpe / 10  (0–1)

Uso:
    from features.builder import build_features
    fm = build_features(bundle)
    # fm.X  → DataFrame con todas las features
    # fm.y  → Series con overreaching_label
"""
from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

LEVEL_ORDER = {"novato": 0, "intermedio": 1, "elite": 2}
ACWR_RISK_THRESHOLD = 1.3
NOTE_CAP_DAYS = 28
EPS = 1e-6  # evitar división por cero

# Columnas base que se pasan directamente al modelo (ya normalizadas)
BASE_FEATURES = [
    "days_since_prev_session",
    "sessions_last_3d",
    "sessions_last_7d",
    "sessions_last_28d",
    "load_last_3d",
    "load_last_7d",
    "mean_session_load_28d",
    "acwr3_freq",
    "acwr3_load",
    "acwr_freq",
    "acwr_load",
    "acwr_valid",
    "days_since_first_session",
    "prev_rpe",
    "prev_session_load",
    "prev_has_note",
    "prob_fatiga_prev_session",
    "prob_fatiga_last_note",
    "days_since_last_note",
    "notes_last_28d",
]

# Columnas derivadas que genera este módulo
DERIVED_FEATURES = [
    "level_enc",
    "load_ratio_3_28",
    "load_ratio_7_28",
    "freq_ratio_3_7",
    "fatigue_signal",
    "days_since_note_capped",
    "acwr_load_risk",
    "acwr_freq_risk",
    "prev_rpe_norm",
]

# Todas las features del modelo (orden fijo → reproducibilidad)
MODEL_FEATURES = BASE_FEATURES + DERIVED_FEATURES


# ---------------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------------

@dataclass
class FeatureMatrix:
    """X e y listos para entrenar/evaluar el modelo."""

    X: pd.DataFrame           # shape (n, len(MODEL_FEATURES))
    y: pd.Series              # overreaching_label (0/1, NaN filtrado)
    feature_names: list[str] = field(default_factory=list)
    imputation_medians: dict[str, float] = field(default_factory=dict)
    n_dropped: int = 0        # filas descartadas por y=NaN

    def __post_init__(self):
        if not self.feature_names:
            self.feature_names = list(self.X.columns)


# ---------------------------------------------------------------------------
# Funciones internas
# ---------------------------------------------------------------------------

def _normalize_level_str(v: str) -> str:
    """Quita acentos y pasa a minúsculas (misma lógica que SessionFeatures.normalize_level)."""
    normalized = unicodedata.normalize("NFD", v.lower().strip())
    return "".join(c for c in normalized if unicodedata.category(c) != "Mn")


def _encode_level(s: pd.Series) -> pd.Series:
    """Convierte level (str) en entero ordinal, normalizando acentos antes."""
    cleaned = s.map(lambda v: _normalize_level_str(v) if isinstance(v, str) else v)
    encoded = cleaned.map(LEVEL_ORDER)
    unknown = encoded.isna() & s.notna()
    if unknown.any():
        logger.warning(
            "level_enc: %d valores desconocidos → imputados con 0 (novato)",
            unknown.sum(),
        )
        encoded = encoded.fillna(0)
    return encoded.astype(int)


def _fatigue_signal(df: pd.DataFrame) -> pd.Series:
    """
    Combina prob_fatiga_prev_session y prob_fatiga_last_note en una sola señal.
    - Si ambas están presentes: media aritmética.
    - Si solo una: esa misma.
    - Si ninguna: NaN (será imputada con la mediana del training set).
    """
    a = df["prob_fatiga_prev_session"]
    b = df["prob_fatiga_last_note"]

    def _combine(x, y):
        vals = [v for v in (x, y) if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(vals)) if vals else np.nan

    return a.combine(b, _combine)


def _compute_imputation_medians(df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    """Calcula las medianas de las columnas con nulls (sobre el df dado)."""
    medians: dict[str, float] = {}
    for col in cols:
        if col in df.columns and df[col].isna().any():
            medians[col] = float(df[col].median())
    return medians


def _apply_imputation(df: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    """Imputa NaN con los valores del diccionario de medianas."""
    df = df.copy()
    for col, val in medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    return df


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

def build_features(
    bundle,                                          # DatasetBundle
    label_col: str = "overreaching_label",
    imputation_medians: dict[str, float] | None = None,
) -> FeatureMatrix:
    """
    Construye la matriz de features a partir de un DatasetBundle.

    Args:
        bundle: DatasetBundle ya cargado desde ingestion.loader.load_dataset().
        label_col: columna objetivo en sessions_outcomes.
        imputation_medians: medianas precomputadas (para inferencia/test set).
                            Si None, se calculan sobre este mismo conjunto.

    Returns:
        FeatureMatrix con X (features) e y (label).
    """
    feat = bundle.sessions_feat.copy()
    outcomes = bundle.sessions_outcomes[["session_id", label_col]].copy()

    # ---- Merge feat + label por session_id --------------------------------
    df = feat.merge(outcomes, on="session_id", how="inner")
    logger.info("Filas tras merge feat+outcomes: %d", len(df))

    # ---- Columnas derivadas -----------------------------------------------
    df["level_enc"] = _encode_level(df["level"])

    df["load_ratio_3_28"]  = df["load_last_3d"] / (df["mean_session_load_28d"] + EPS)
    df["load_ratio_7_28"]  = df["load_last_7d"] / (df["mean_session_load_28d"] * 4 + EPS)
    df["freq_ratio_3_7"]   = df["sessions_last_3d"] / (df["sessions_last_7d"] + EPS)

    df["fatigue_signal"]         = _fatigue_signal(df)
    df["days_since_note_capped"] = df["days_since_last_note"].clip(upper=NOTE_CAP_DAYS)

    df["acwr_load_risk"] = (df["acwr_load"].fillna(0) > ACWR_RISK_THRESHOLD).astype(int)
    df["acwr_freq_risk"] = (df["acwr_freq"].fillna(0) > ACWR_RISK_THRESHOLD).astype(int)
    df["prev_rpe_norm"]  = df["prev_rpe"] / 10.0

    # ---- Extraer X con columnas en orden fijo -----------------------------
    X = df[MODEL_FEATURES].copy()

    # ---- Imputación de NaN ------------------------------------------------
    if imputation_medians is None:
        imputation_medians = _compute_imputation_medians(X, MODEL_FEATURES)
        logger.info(
            "Imputación calculada para %d columnas con NaN", len(imputation_medians)
        )

    X = _apply_imputation(X, imputation_medians)

    # ---- Label: descartar filas sin etiqueta ------------------------------
    y_raw = df[label_col]
    mask_valid = y_raw.notna()
    n_dropped = int((~mask_valid).sum())
    if n_dropped:
        logger.warning("Descartadas %d filas con %s=NaN", n_dropped, label_col)

    X = X.loc[mask_valid].reset_index(drop=True)
    y = y_raw.loc[mask_valid].astype(int).reset_index(drop=True)

    logger.info(
        "FeatureMatrix: %d filas × %d features | label='%s' | positivos=%.1f%%",
        len(X), len(MODEL_FEATURES), label_col,
        100 * y.mean(),
    )

    return FeatureMatrix(
        X=X,
        y=y,
        feature_names=MODEL_FEATURES,
        imputation_medians=imputation_medians,
        n_dropped=n_dropped,
    )
