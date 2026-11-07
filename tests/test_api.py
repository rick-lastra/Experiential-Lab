"""
Tests de la API FastAPI para WODPredict-AI — Fase 5.

Cubre:
  - GET  /health: estado ok, model_loaded=True
  - POST /predict: respuesta válida, proba en rango, risk_label correcto
  - POST /explain: top_drivers presente, base_value, serialización
  - Validación de entrada: campos faltantes → 422
  - _risk_label: umbrales correctos
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app, _risk_label


# ---------------------------------------------------------------------------
# Fixture: cliente de test (levanta el lifespan una sola vez por módulo)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Payload de referencia (atleta con carga moderada)
# ---------------------------------------------------------------------------

VALID_PAYLOAD = {
    "session_id":                "TEST_2025-03-01",
    "days_since_prev_session":   2.0,
    "sessions_last_3d":          1.0,
    "sessions_last_7d":          3.0,
    "sessions_last_28d":         10.0,
    "load_last_3d":              380.0,
    "load_last_7d":              950.0,
    "mean_session_load_28d":     350.0,
    "acwr3_freq":                None,
    "acwr3_load":                None,
    "acwr_freq":                 1.1,
    "acwr_load":                 1.05,
    "acwr_valid":                1,
    "days_since_first_session":  60,
    "prev_rpe":                  6.0,
    "prev_session_load":         300.0,
    "prev_has_note":             0.0,
    "prob_fatiga_prev_session":  None,
    "prob_fatiga_last_note":     0.3,
    "days_since_last_note":      7.0,
    "notes_last_28d":            2,
    "level":                     "intermedio",
}

# Payload de alto riesgo: ACWR muy alto, muchas sesiones recientes, RPE alto
HIGH_RISK_PAYLOAD = {
    **VALID_PAYLOAD,
    "session_id":                "HIGH_2025-03-01",
    "sessions_last_3d":          3.0,
    "sessions_last_7d":          7.0,
    "load_last_3d":              1200.0,
    "load_last_7d":              2800.0,
    "mean_session_load_28d":     600.0,
    "acwr_freq":                 1.8,
    "acwr_load":                 1.9,
    "acwr_valid":                1,
    "prev_rpe":                  9.0,
    "prob_fatiga_prev_session":  0.85,
    "prob_fatiga_last_note":     0.9,
    "days_since_last_note":      1.0,
    "notes_last_28d":            5,
}


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_status_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_model_loaded(self, client):
        data = client.get("/health").json()
        assert data["model_loaded"] is True

    def test_status_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_model_version_present(self, client):
        data = client.get("/health").json()
        assert "model_version" in data


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------

class TestPredict:
    def test_status_200(self, client):
        r = client.post("/predict", json=VALID_PAYLOAD)
        assert r.status_code == 200

    def test_session_id_echoed(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert data["session_id"] == VALID_PAYLOAD["session_id"]

    def test_proba_in_range(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert 0.0 <= data["predicted_proba"] <= 1.0

    def test_risk_label_valid(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert data["risk_label"] in {"bajo", "moderado", "alto"}

    def test_model_version_present(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert "model_version" in data

    def test_high_risk_payload_proba_higher(self, client):
        """El payload de alto riesgo debe tener mayor proba que el moderado."""
        normal = client.post("/predict", json=VALID_PAYLOAD).json()["predicted_proba"]
        high   = client.post("/predict", json=HIGH_RISK_PAYLOAD).json()["predicted_proba"]
        assert high > normal

    def test_missing_required_field_returns_422(self, client):
        bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "sessions_last_3d"}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_nullable_fields_accepted(self, client):
        """Payload con todos los opcionales en None debe responder 200."""
        payload = {**VALID_PAYLOAD, "acwr_freq": None, "acwr_load": None,
                   "prev_rpe": None, "days_since_prev_session": None}
        r = client.post("/predict", json=payload)
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# POST /explain
# ---------------------------------------------------------------------------

class TestExplain:
    def test_status_200(self, client):
        r = client.post("/explain", json=VALID_PAYLOAD)
        assert r.status_code == 200

    def test_session_id_echoed(self, client):
        data = client.post("/explain", json=VALID_PAYLOAD).json()
        assert data["session_id"] == VALID_PAYLOAD["session_id"]

    def test_proba_in_range(self, client):
        data = client.post("/explain", json=VALID_PAYLOAD).json()
        assert 0.0 <= data["predicted_proba"] <= 1.0

    def test_risk_label_valid(self, client):
        data = client.post("/explain", json=VALID_PAYLOAD).json()
        assert data["risk_label"] in {"bajo", "moderado", "alto"}

    def test_base_value_present(self, client):
        data = client.post("/explain", json=VALID_PAYLOAD).json()
        assert "base_value" in data
        assert isinstance(data["base_value"], float)

    def test_top_drivers_length(self, client):
        data = client.post("/explain", json=VALID_PAYLOAD).json()
        assert len(data["top_drivers"]) == 8

    def test_top_drivers_keys(self, client):
        data = client.post("/explain", json=VALID_PAYLOAD).json()
        for d in data["top_drivers"]:
            assert "feature" in d
            assert "shap_value" in d
            assert "feature_value" in d

    def test_predict_and_explain_proba_match(self, client):
        """predict y explain deben retornar la misma probabilidad."""
        p = client.post("/predict", json=VALID_PAYLOAD).json()["predicted_proba"]
        e = client.post("/explain", json=VALID_PAYLOAD).json()["predicted_proba"]
        assert abs(p - e) < 1e-4


# ---------------------------------------------------------------------------
# _risk_label (unit)
# ---------------------------------------------------------------------------

class TestRiskLabel:
    def test_bajo(self):
        assert _risk_label(0.10) == "bajo"
        assert _risk_label(0.29) == "bajo"

    def test_moderado(self):
        assert _risk_label(0.30) == "moderado"
        assert _risk_label(0.59) == "moderado"

    def test_alto(self):
        assert _risk_label(0.60) == "alto"
        assert _risk_label(0.99) == "alto"
