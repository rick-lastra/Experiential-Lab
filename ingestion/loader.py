"""
Ingesta de datos desde HuggingFace para WODPredict-AI.

Descarga los archivos del repo rick-lastra/wodpredict-synth, los carga
como DataFrames y valida su esquema con los contratos Pydantic de schema.py.

Uso como módulo:
    from ingestion.loader import load_dataset
    data = load_dataset()

Uso directo:
    python -m ingestion.loader
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from huggingface_hub import hf_hub_download
from pydantic import ValidationError

from ingestion.schema import (
    AthleteRow,
    SessionFeatures,
    SessionOutcomes,
    ValidationReport,
    WorkoutRow,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

REPO_ID = "rick-lastra/wodpredict-synth"
REPO_TYPE = "dataset"

# Archivos principales del dataset
FILES = {
    "athletes":  "datos_ini/athletes.csv",
    "sessions":  "datos_ini/sessions.csv",
    "workouts":  "datos_ini/workouts_catalog.csv",
    "components":"sintetico_generador/components.csv",
}

# Columnas feat permitidas para el modelo (regla anti-fuga)
SESSION_FEAT_COLS = [
    "session_id", "athlete_id", "date", "level",
    "days_since_prev_session",
    "sessions_last_3d", "sessions_last_7d", "sessions_last_28d",
    "load_last_3d", "load_last_7d", "mean_session_load_28d",
    "acwr3_freq", "acwr3_load", "acwr_freq", "acwr_load", "acwr_valid",
    "days_since_first_session",
    "prev_rpe", "prev_session_load", "prev_has_note",
    "prob_fatiga_prev_session", "prob_fatiga_last_note",
    "days_since_last_note", "notes_last_28d",
]

SESSION_OUTCOME_COLS = [
    "session_id",
    "n_components", "has_metcon", "session_minutes",
    "rpe", "session_load", "pr_count",
    "has_note", "prob_fatiga",
    "overreaching_h1", "overreaching_h3", "overreaching_h7",
    "overreaching_label",
    "note_id", "source", "captured_at",
]


# ---------------------------------------------------------------------------
# Resultado de carga
# ---------------------------------------------------------------------------

@dataclass
class DatasetBundle:
    """Tablas limpias listas para feature engineering y entrenamiento."""

    athletes: pd.DataFrame
    sessions_feat: pd.DataFrame    # solo columnas feat (entrada del modelo)
    sessions_outcomes: pd.DataFrame  # targets + outcomes (NO mezclar con feat)
    workouts: pd.DataFrame
    reports: list[ValidationReport] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return all(r.is_clean for r in self.reports)


# ---------------------------------------------------------------------------
# Helpers de descarga
# ---------------------------------------------------------------------------

def _download(filename: str, token: Optional[str] = None) -> Path:
    """Descarga un archivo del repo y retorna su ruta local en caché."""
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type=REPO_TYPE,
        token=token,
    )
    return Path(path)


def _normalize_athletes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza inconsistencias de capitalización en athletes.csv.
    El generador sintético produce variantes como 'Novato'/'novato' y 'm'/'M'.
    Aplica lowercase en level, uppercase en sex, e imputa age=NaN con la mediana.
    """
    df = df.copy()
    df["level"] = df["level"].str.lower().str.strip()
    df["sex"] = df["sex"].str.upper().str.strip()
    n_nan_age = df["age"].isna().sum()
    if n_nan_age:
        median_age = df["age"].median()
        df["age"] = df["age"].fillna(median_age)
        logger.warning(
            "athletes.csv: %d filas con age=NaN imputadas con la mediana (%.1f)",
            n_nan_age, median_age,
        )
    return df


# ---------------------------------------------------------------------------
# Validación por tabla
# ---------------------------------------------------------------------------

def _validate_athletes(df: pd.DataFrame) -> ValidationReport:
    errors: list[dict] = []
    for i, row in df.iterrows():
        try:
            AthleteRow(**row.to_dict())
        except ValidationError as exc:
            errors.append({"row": int(i), "athlete_id": row.get("athlete_id"), "errors": exc.errors()})

    return ValidationReport(
        table="athletes",
        total_rows=len(df),
        valid_rows=len(df) - len(errors),
        invalid_rows=len(errors),
        errors=errors[:20],  # máximo 20 errores en el reporte
    )


def _validate_sessions_feat(df: pd.DataFrame) -> ValidationReport:
    errors: list[dict] = []
    for i, row in df.iterrows():
        try:
            SessionFeatures(**row.to_dict())
        except ValidationError as exc:
            errors.append({"row": int(i), "session_id": row.get("session_id"), "errors": exc.errors()})

    return ValidationReport(
        table="sessions_feat",
        total_rows=len(df),
        valid_rows=len(df) - len(errors),
        invalid_rows=len(errors),
        errors=errors[:20],
    )


def _validate_sessions_outcomes(df: pd.DataFrame) -> ValidationReport:
    errors: list[dict] = []
    for i, row in df.iterrows():
        try:
            SessionOutcomes(**row.to_dict())
        except ValidationError as exc:
            errors.append({"row": int(i), "session_id": row.get("session_id"), "errors": exc.errors()})

    return ValidationReport(
        table="sessions_outcomes",
        total_rows=len(df),
        valid_rows=len(df) - len(errors),
        invalid_rows=len(errors),
        errors=errors[:20],
    )


# ---------------------------------------------------------------------------
# Carga principal
# ---------------------------------------------------------------------------

def load_dataset(
    token: Optional[str] = None,
    validate: bool = True,
    sample_validate: int = 500,
) -> DatasetBundle:
    """
    Descarga el dataset desde HuggingFace y retorna un DatasetBundle.

    Args:
        token: HF token opcional (para repos privados o mayor rate limit).
        validate: Si True, valida esquema Pydantic sobre una muestra.
        sample_validate: Número de filas a validar (0 = todas, puede ser lento).

    Returns:
        DatasetBundle con las tablas limpias y un reporte de validación.
    """
    logger.info("Descargando dataset desde %s ...", REPO_ID)

    # --- Athletes ---
    athletes_path = _download(FILES["athletes"], token)
    athletes = _normalize_athletes(pd.read_csv(athletes_path))
    logger.info("athletes.csv: %d filas", len(athletes))

    # --- Sessions ---
    sessions_path = _download(FILES["sessions"], token)
    sessions_raw = pd.read_csv(sessions_path)
    logger.info("sessions.csv: %d filas, %d columnas", *sessions_raw.shape)

    # Separar features de outcomes (regla anti-fuga)
    sessions_feat = sessions_raw[SESSION_FEAT_COLS].copy()
    sessions_outcomes = sessions_raw[SESSION_OUTCOME_COLS].copy()

    # --- Workouts ---
    workouts_path = _download(FILES["workouts"], token)
    workouts = pd.read_csv(workouts_path)
    logger.info("workouts_catalog.csv: %d filas", len(workouts))

    reports: list[ValidationReport] = []

    if validate:
        logger.info("Validando esquema (muestra de %d filas)...", sample_validate or len(sessions_feat))

        # Atletas: dataset pequeño, validar todo
        rep_ath = _validate_athletes(athletes)
        reports.append(rep_ath)
        _log_report(rep_ath)

        # Sessions: validar muestra para no bloquear
        feat_sample = sessions_feat if sample_validate == 0 else sessions_feat.sample(
            min(sample_validate, len(sessions_feat)), random_state=42
        )
        rep_feat = _validate_sessions_feat(feat_sample)
        reports.append(rep_feat)
        _log_report(rep_feat)

        out_sample = sessions_outcomes if sample_validate == 0 else sessions_outcomes.sample(
            min(sample_validate, len(sessions_outcomes)), random_state=42
        )
        rep_out = _validate_sessions_outcomes(out_sample)
        reports.append(rep_out)
        _log_report(rep_out)

    return DatasetBundle(
        athletes=athletes,
        sessions_feat=sessions_feat,
        sessions_outcomes=sessions_outcomes,
        workouts=workouts,
        reports=reports,
    )


def _log_report(report: ValidationReport) -> None:
    if report.is_clean:
        logger.info("[OK] %s: %d/%d filas válidas", report.table, report.valid_rows, report.total_rows)
    else:
        logger.warning(
            "[WARN] %s: %d filas inválidas de %d",
            report.table, report.invalid_rows, report.total_rows,
        )
        for err in report.errors[:3]:
            logger.warning("  %s", err)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    bundle = load_dataset()
    print("\n=== Resumen de validación ===")
    for r in bundle.reports:
        status = "✓" if r.is_clean else "✗"
        print(f"  {status} {r.table}: {r.valid_rows}/{r.total_rows} filas válidas")
    print(f"\nDataset {'limpio' if bundle.is_clean else 'con errores'}.")
    print(f"  athletes:         {len(bundle.athletes):>6,} filas")
    print(f"  sessions (feat):  {len(bundle.sessions_feat):>6,} filas × {len(bundle.sessions_feat.columns)} cols")
    print(f"  sessions (out):   {len(bundle.sessions_outcomes):>6,} filas × {len(bundle.sessions_outcomes.columns)} cols")
    print(f"  workouts:         {len(bundle.workouts):>6,} filas")
