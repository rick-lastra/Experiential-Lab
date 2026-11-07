"""
Contratos de datos para WODPredict-AI.

Cada modelo Pydantic refleja exactamente una tabla del dataset sintético
rick-lastra/wodpredict-synth.  Los campos están anotados con su rol:
  feat    → variable permitida como entrada del modelo (info ANTES de la sesión)
  outcome → resultado de la sesión (prohibido como feature de esa misma fila)
  label   → objetivo de predicción
  audit   → solo trazabilidad
  id      → identificador

La separación en SessionFeatures / SessionOutcomes hace que sea imposible
pasar un outcome como feature sin una conversión explícita (regla anti-fuga).
"""
from __future__ import annotations

import math
import unicodedata
from datetime import date, datetime
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, BeforeValidator, Field, field_validator


def _nan_to_none(v: object) -> object:
    """Convierte float NaN a None. Pydantic v2 no lo hace automáticamente."""
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


# Tipo reutilizable: float opcional que acepta NaN de pandas como None
OptFloat = Annotated[Optional[float], BeforeValidator(_nan_to_none)]


# ---------------------------------------------------------------------------
# Athlete
# ---------------------------------------------------------------------------

class AthleteRow(BaseModel):
    """Una fila de athletes.csv."""

    # id
    athlete_id: str

    # feat
    level: Literal["novato", "intermedio", "elite"]
    sex: Literal["M", "F"]
    age: float = Field(ge=14, le=80)
    language: Literal["es", "en"]
    join_date: date

    # audit
    attendance_pattern: Literal["regular", "erratic"]
    base_freq_per_week: float = Field(ge=0)
    source: str

    @field_validator("join_date", mode="before")
    @classmethod
    def parse_date(cls, v: object) -> date:
        if isinstance(v, date):
            return v
        return date.fromisoformat(str(v))


# ---------------------------------------------------------------------------
# Session — features SOLO (información disponible ANTES de la sesión)
# ---------------------------------------------------------------------------

class SessionFeatures(BaseModel):
    """
    Variables permitidas como entrada del modelo para la sesión t.
    Toda la información aquí es anterior a la sesión (hasta t-1).
    """

    # id
    session_id: str
    athlete_id: str
    date: date
    level: Literal["novato", "intermedio", "elite"]

    # feat — frecuencia previa
    days_since_prev_session: OptFloat = Field(default=None, ge=0)
    sessions_last_3d: float = Field(ge=0)
    sessions_last_7d: float = Field(ge=0)
    sessions_last_28d: float = Field(ge=0)

    # feat — carga previa
    load_last_3d: float = Field(ge=0)
    load_last_7d: float = Field(ge=0)
    mean_session_load_28d: OptFloat = Field(default=None, ge=0)

    # feat — ACWR
    acwr3_freq: OptFloat = Field(default=None, ge=0)
    acwr3_load: OptFloat = Field(default=None, ge=0)
    acwr_freq: OptFloat = Field(default=None, ge=0)
    acwr_load: OptFloat = Field(default=None, ge=0)
    acwr_valid: int = Field(ge=0, le=1)

    # feat — antigüedad
    days_since_first_session: int = Field(ge=0)

    # feat — sesión anterior
    prev_rpe: OptFloat = Field(default=None, ge=1, le=10)
    prev_session_load: OptFloat = Field(default=None, ge=0)
    prev_has_note: OptFloat = Field(default=None, ge=0, le=1)

    # feat — señal de fatiga subjetiva previa
    prob_fatiga_prev_session: OptFloat = Field(default=None, ge=0, le=1)
    prob_fatiga_last_note: OptFloat = Field(default=None, ge=0, le=1)
    days_since_last_note: OptFloat = Field(default=None, ge=0)
    notes_last_28d: int = Field(ge=0)

    @field_validator("level", mode="before")
    @classmethod
    def normalize_level(cls, v: object) -> object:
        if isinstance(v, str):
            normalized = unicodedata.normalize("NFD", v.lower().strip())
            return "".join(c for c in normalized if unicodedata.category(c) != "Mn")
        return v

    @field_validator("date", mode="before")
    @classmethod
    def parse_date(cls, v: object) -> date:
        if isinstance(v, date):
            return v
        return date.fromisoformat(str(v)[:10])


# ---------------------------------------------------------------------------
# Session — outcomes (PROHIBIDOS como features de esa misma sesión)
# ---------------------------------------------------------------------------

class SessionOutcomes(BaseModel):
    """
    Resultados observados DURANTE / DESPUÉS de la sesión.
    NO deben usarse para predecir esa misma sesión (regla anti-fuga).
    """

    session_id: str

    # outcome
    n_components: int = Field(ge=0)
    has_metcon: int = Field(ge=0, le=1)
    session_minutes: float = Field(ge=0)
    rpe: int = Field(ge=1, le=10)
    session_load: float = Field(ge=0)
    pr_count: int = Field(ge=0)
    has_note: int = Field(ge=0, le=1)
    prob_fatiga: OptFloat = Field(default=None, ge=0, le=1)

    # label
    overreaching_h1: Optional[float] = None
    overreaching_h3: Optional[float] = None
    overreaching_h7: Optional[float] = None
    overreaching_label: Optional[float] = None  # objetivo principal

    # audit
    note_id: int
    source: str
    captured_at: Optional[datetime] = None

    @field_validator("captured_at", mode="before")
    @classmethod
    def parse_captured_at(cls, v: object) -> Optional[datetime]:
        if v is None or (isinstance(v, float)):
            return None
        return datetime.fromisoformat(str(v))


# ---------------------------------------------------------------------------
# Workout catalog
# ---------------------------------------------------------------------------

class WorkoutRow(BaseModel):
    """Una fila de workouts_catalog.csv."""

    workout_id: str
    title: str
    kind: str
    ref_value: Optional[float] = None
    time_cap_s: Optional[float] = None
    demand: Optional[float] = None
    lift: Optional[str] = None
    family: Optional[str] = None
    pct: Optional[float] = None
    reps: Optional[float] = None
    ref_1rm: Optional[float] = None
    minutes: Optional[float] = None
    round_size: Optional[float] = None


# ---------------------------------------------------------------------------
# Resultado de validación
# ---------------------------------------------------------------------------

class ValidationReport(BaseModel):
    """Resumen de la validación de un DataFrame cargado."""

    table: str
    total_rows: int
    valid_rows: int
    invalid_rows: int
    errors: list[dict] = Field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return self.invalid_rows == 0
