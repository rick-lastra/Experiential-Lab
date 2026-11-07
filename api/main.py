"""
API FastAPI para WODPredict-AI — Fase 5.

Expone tres endpoints:
  GET  /health   → estado del servicio y versión del modelo
  POST /predict  → probabilidad de overreaching para una sesión
  POST /explain  → probabilidad + explicación SHAP (top drivers)

El modelo XGBoost y el SHAPExplainer se inicializan una sola vez al arrancar
(lifespan) y se comparten como estado de la aplicación para evitar latencia
en cada petición.

Arranque:
    uvicorn api.main:app --reload --port 8000

Ejemplo de petición:
    curl -X POST http://localhost:8000/predict \\
      -H "Content-Type: application/json" \\
      -d @payload.json
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import MEDIANS_PATH, MODEL_PATH
from explainability.shap_explainer import SHAPExplainer
from features.builder import MODEL_FEATURES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Estado global de la aplicación (cargado una sola vez en lifespan)
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo y el explainer al arrancar; libera al cerrar."""
    logger.info("Cargando modelo y SHAPExplainer...")
    _state["explainer"] = SHAPExplainer.from_artifacts(
        model_path=MODEL_PATH,
        medians_path=MEDIANS_PATH,
    )
    logger.info("Modelo listo.")
    yield
    _state.clear()
    logger.info("Recursos liberados.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="WODPredict-AI",
    description="Predicción de overreaching y prescripción adaptativa de cargas para CrossFit & Hyrox",
    version="0.1.0",
    lifespan=lifespan,
)


def _get_explainer() -> SHAPExplainer:
    if "explainer" not in _state:
        raise HTTPException(status_code=503, detail="Modelo no disponible. Reinicia el servicio.")
    return _state["explainer"]


# ---------------------------------------------------------------------------
# Schemas de entrada / salida
# ---------------------------------------------------------------------------

class SessionFeaturesRequest(BaseModel):
    """
    Features de una sesión de entrenamiento.
    Todos los campos opcionales se imputarán con las medianas del training set.
    """
    session_id: str = Field(..., description="Identificador único de la sesión (ej. 'A0001_2025-03-01')")

    # Frecuencia previa
    days_since_prev_session: float | None = Field(None, ge=0)
    sessions_last_3d:        float        = Field(..., ge=0)
    sessions_last_7d:        float        = Field(..., ge=0)
    sessions_last_28d:       float        = Field(..., ge=0)

    # Carga previa
    load_last_3d:           float        = Field(..., ge=0)
    load_last_7d:           float        = Field(..., ge=0)
    mean_session_load_28d:  float | None = Field(None, ge=0)

    # ACWR
    acwr3_freq:  float | None = Field(None, ge=0)
    acwr3_load:  float | None = Field(None, ge=0)
    acwr_freq:   float | None = Field(None, ge=0)
    acwr_load:   float | None = Field(None, ge=0)
    acwr_valid:  int          = Field(..., ge=0, le=1)

    # Antigüedad
    days_since_first_session: int = Field(..., ge=0)

    # Sesión anterior
    prev_rpe:          float | None = Field(None, ge=1, le=11)
    prev_session_load: float | None = Field(None, ge=0)
    prev_has_note:     float | None = Field(None, ge=0, le=1)

    # Señal de fatiga subjetiva
    prob_fatiga_prev_session: float | None = Field(None, ge=0, le=1)
    prob_fatiga_last_note:    float | None = Field(None, ge=0, le=1)
    days_since_last_note:     float | None = Field(None, ge=0)
    notes_last_28d:           int          = Field(..., ge=0)

    # Perfil del atleta
    level: str = Field(..., description="Nivel: 'novato', 'intermedio' o 'elite'")

    def to_feature_df(self) -> pd.DataFrame:
        """
        Convierte la request a un DataFrame listo para build_features / SHAPExplainer.
        Solo incluye las columnas base (MODEL_FEATURES base); las derivadas
        las calcula el explainer internamente vía _prepare + builder logic.
        """
        data = self.model_dump(exclude={"session_id"})
        return pd.DataFrame([data])


class PredictResponse(BaseModel):
    session_id:       str
    predicted_proba:  float = Field(description="Probabilidad de overreaching [0, 1]")
    risk_label:       str   = Field(description="'bajo' | 'moderado' | 'alto'")
    model_version:    str   = "0.1.0"


class DriverItem(BaseModel):
    feature:       str
    shap_value:    float
    feature_value: float


class ExplainResponse(BaseModel):
    session_id:      str
    predicted_proba: float
    risk_label:      str
    base_value:      float
    top_drivers:     list[DriverItem]
    model_version:   str = "0.1.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _risk_label(proba: float) -> str:
    if proba < 0.30:
        return "bajo"
    if proba < 0.60:
        return "moderado"
    return "alto"


def _request_to_model_df(req: SessionFeaturesRequest, explainer: SHAPExplainer) -> pd.DataFrame:
    """
    Construye el DataFrame completo con MODEL_FEATURES (base + derivadas)
    aplicando la misma lógica que features.builder, pero para una sola fila.
    Reutiliza _prepare del explainer para imputación.
    """
    import unicodedata

    from features.builder import (
        ACWR_RISK_THRESHOLD, EPS, LEVEL_ORDER, NOTE_CAP_DAYS,
    )

    raw = req.model_dump(exclude={"session_id"})

    # Derivadas
    load_28 = raw.get("mean_session_load_28d") or 0.0
    load_7  = raw.get("load_last_7d", 0.0)
    load_3  = raw.get("load_last_3d", 0.0)
    sess_3  = raw.get("sessions_last_3d", 0.0)
    sess_7  = raw.get("sessions_last_7d", 0.0)

    def _norm_level(v: str) -> str:
        n = unicodedata.normalize("NFD", v.lower().strip())
        return "".join(c for c in n if unicodedata.category(c) != "Mn")

    level_str = _norm_level(raw.get("level", "novato"))
    level_enc = LEVEL_ORDER.get(level_str, 0)

    a = raw.get("prob_fatiga_prev_session")
    b = raw.get("prob_fatiga_last_note")
    vals = [v for v in (a, b) if v is not None and not (isinstance(v, float) and np.isnan(v))]
    fatigue_signal = float(np.mean(vals)) if vals else np.nan

    days_note = raw.get("days_since_last_note")
    days_note_capped = min(days_note, NOTE_CAP_DAYS) if days_note is not None else np.nan

    acwr_load = raw.get("acwr_load") or 0.0
    acwr_freq = raw.get("acwr_freq") or 0.0

    prev_rpe = raw.get("prev_rpe")
    prev_rpe_norm = (prev_rpe / 10.0) if prev_rpe is not None else np.nan

    row = {
        # base features (en el mismo orden que BASE_FEATURES)
        "days_since_prev_session":  raw.get("days_since_prev_session"),
        "sessions_last_3d":         sess_3,
        "sessions_last_7d":         sess_7,
        "sessions_last_28d":        raw.get("sessions_last_28d", 0.0),
        "load_last_3d":             load_3,
        "load_last_7d":             load_7,
        "mean_session_load_28d":    raw.get("mean_session_load_28d"),
        "acwr3_freq":               raw.get("acwr3_freq"),
        "acwr3_load":               raw.get("acwr3_load"),
        "acwr_freq":                raw.get("acwr_freq"),
        "acwr_load":                raw.get("acwr_load"),
        "acwr_valid":               raw.get("acwr_valid", 0),
        "days_since_first_session": raw.get("days_since_first_session", 0),
        "prev_rpe":                 prev_rpe,
        "prev_session_load":        raw.get("prev_session_load"),
        "prev_has_note":            raw.get("prev_has_note"),
        "prob_fatiga_prev_session": a,
        "prob_fatiga_last_note":    b,
        "days_since_last_note":     days_note,
        "notes_last_28d":           raw.get("notes_last_28d", 0),
        # derived features
        "level_enc":                level_enc,
        "load_ratio_3_28":          load_3 / (load_28 + EPS),
        "load_ratio_7_28":          load_7 / (load_28 * 4 + EPS),
        "freq_ratio_3_7":           sess_3 / (sess_7 + EPS),
        "fatigue_signal":           fatigue_signal,
        "days_since_note_capped":   days_note_capped,
        "acwr_load_risk":           int(acwr_load > ACWR_RISK_THRESHOLD),
        "acwr_freq_risk":           int(acwr_freq > ACWR_RISK_THRESHOLD),
        "prev_rpe_norm":            prev_rpe_norm,
    }

    df = pd.DataFrame([row])[MODEL_FEATURES]
    return explainer._prepare(df)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Sistema"])
def health() -> dict:
    """Verifica que el servicio está activo y el modelo cargado."""
    loaded = "explainer" in _state
    return {
        "status":        "ok" if loaded else "degraded",
        "model_loaded":  loaded,
        "model_version": "0.1.0",
    }


@app.post("/predict", response_model=PredictResponse, tags=["Predicción"])
def predict(req: SessionFeaturesRequest) -> PredictResponse:
    """
    Predice la probabilidad de overreaching para una sesión.

    Devuelve:
    - `predicted_proba`: probabilidad [0, 1]
    - `risk_label`: 'bajo' (<0.30) | 'moderado' (0.30–0.60) | 'alto' (>0.60)
    """
    explainer = _get_explainer()
    try:
        df = _request_to_model_df(req, explainer)
        proba = float(explainer.model.predict_proba(df)[0, 1])
    except Exception as exc:
        logger.exception("Error en /predict para session_id=%s", req.session_id)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PredictResponse(
        session_id=req.session_id,
        predicted_proba=round(proba, 4),
        risk_label=_risk_label(proba),
    )


@app.post("/explain", response_model=ExplainResponse, tags=["Explicabilidad"])
def explain(req: SessionFeaturesRequest) -> ExplainResponse:
    """
    Predice la probabilidad de overreaching y explica los factores principales.

    Devuelve:
    - `predicted_proba`: probabilidad [0, 1]
    - `risk_label`: 'bajo' | 'moderado' | 'alto'
    - `base_value`: valor esperado del modelo (línea base SHAP)
    - `top_drivers`: top-8 features que más influyen en la predicción, con
      su valor SHAP (positivo = empuja hacia overreaching) y el valor de la feature
    """
    explainer = _get_explainer()
    try:
        df = _request_to_model_df(req, explainer)
        local_exps = explainer.explain_local(df, session_ids=[req.session_id], top_n=8)
    except Exception as exc:
        logger.exception("Error en /explain para session_id=%s", req.session_id)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    exp = local_exps[0]
    return ExplainResponse(
        session_id=exp.session_id,
        predicted_proba=round(exp.predicted_proba, 4),
        risk_label=_risk_label(exp.predicted_proba),
        base_value=round(exp.base_value, 4),
        top_drivers=[DriverItem(**d) for d in exp.top_drivers],
    )
