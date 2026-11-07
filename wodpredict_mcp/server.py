"""
Servidor MCP para WODPredict-AI — Fase 6.

Expone las capacidades de la API FastAPI como tools MCP que puede invocar
cualquier agente de IA (Bob, Claude, watsonx Orchestrate).

Tools disponibles:
  - predict_overreaching : probabilidad de overreaching para una sesión
  - explain_overreaching : probabilidad + top drivers SHAP en lenguaje natural
  - health_check         : verifica que la API FastAPI está disponible

Uso (stdio, para conectar con Bob):
    python -m mcp.server

Configuración en Bob (.bob/mcp.json):
    {
      "mcpServers": {
        "wodpredict": {
          "command": "D:\\Miniconda3\\Scripts\\conda.exe",
          "args": ["run", "-n", "wodpredict", "python", "-m", "mcp.server"],
          "cwd": "<ruta_al_proyecto>",
          "env": { "WODPREDICT_API_URL": "http://localhost:8002" }
        }
      }
    }
"""
from __future__ import annotations

import json
import logging
import os
import sys

import httpx
#from mcp.server.mcpserver import MCPServer as FastMCP
from mcp.server.fastmcp import FastMCP
# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

API_URL = os.environ.get("WODPREDICT_API_URL", "http://localhost:8002")

# Logging a stderr (stdout es el canal MCP — nunca escribir ahí)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("wodpredict.mcp")

# ---------------------------------------------------------------------------
# Servidor MCP
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="wodpredict-ai",
    instructions=(
        "Herramienta de predicción de overreaching para atletas de CrossFit y Hyrox. "
        "Usa predict_overreaching para estimar el riesgo de sobreentrenamiento, "
        "y explain_overreaching para obtener además los factores principales que "
        "explican la predicción. Todos los campos opcionales pueden omitirse."
    ),
)


# ---------------------------------------------------------------------------
# Helper HTTP
# ---------------------------------------------------------------------------

def _post(endpoint: str, payload: dict) -> dict:
    """Llama a un endpoint POST de la API FastAPI y devuelve el JSON."""
    url = f"{API_URL}{endpoint}"
    with httpx.Client(timeout=15.0) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()


def _get(endpoint: str) -> dict:
    with httpx.Client(timeout=10.0) as client:
        resp = httpx.get(f"{API_URL}{endpoint}", timeout=10.0)
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def health_check() -> str:
    """
    Verifica que la API de WODPredict-AI está activa y el modelo está cargado.
    Úsala antes de hacer predicciones si no estás seguro de que el servicio está corriendo.
    """
    try:
        data = _get("/health")
        if data.get("model_loaded"):
            return f"✅ API activa. Modelo cargado (v{data.get('model_version', '?')})."
        return f"⚠️ API activa pero modelo NO cargado. Estado: {data.get('status')}."
    except Exception as exc:
        return f"❌ API no disponible en {API_URL}: {exc}"


@mcp.tool()
def predict_overreaching(
    sessions_last_3d: float,
    sessions_last_7d: float,
    sessions_last_28d: float,
    load_last_3d: float,
    load_last_7d: float,
    acwr_valid: int,
    days_since_first_session: int,
    notes_last_28d: int,
    level: str,
    days_since_prev_session: float | None = None,
    mean_session_load_28d: float | None = None,
    acwr3_freq: float | None = None,
    acwr3_load: float | None = None,
    acwr_freq: float | None = None,
    acwr_load: float | None = None,
    prev_rpe: float | None = None,
    prev_session_load: float | None = None,
    prev_has_note: float | None = None,
    prob_fatiga_prev_session: float | None = None,
    prob_fatiga_last_note: float | None = None,
    days_since_last_note: float | None = None,
) -> str:
    """
    Predice la probabilidad de overreaching (sobreentrenamiento) para una sesión de entrenamiento.

    Devuelve la probabilidad [0-1] y el nivel de riesgo: 'bajo' (<30%), 'moderado' (30-60%) o 'alto' (>60%).

    Parámetros requeridos:
    - sessions_last_3d: número de sesiones en los últimos 3 días
    - sessions_last_7d: número de sesiones en los últimos 7 días
    - sessions_last_28d: número de sesiones en los últimos 28 días
    - load_last_3d: carga de entrenamiento (RPE × minutos) en los últimos 3 días
    - load_last_7d: carga en los últimos 7 días
    - acwr_valid: 1 si el ACWR es válido (suficiente historial), 0 si no
    - days_since_first_session: días desde la primera sesión registrada
    - notes_last_28d: número de notas de fatiga subjetiva en los últimos 28 días
    - level: nivel del atleta ('novato', 'intermedio' o 'elite')
    """
    import datetime
    auto_id = f"{level.upper()}_{datetime.date.today().isoformat()}"
    payload = {
        "session_id": auto_id,
        "sessions_last_3d": sessions_last_3d,
        "sessions_last_7d": sessions_last_7d,
        "sessions_last_28d": sessions_last_28d,
        "load_last_3d": load_last_3d,
        "load_last_7d": load_last_7d,
        "acwr_valid": acwr_valid,
        "days_since_first_session": days_since_first_session,
        "notes_last_28d": notes_last_28d,
        "level": level,
        "days_since_prev_session": days_since_prev_session,
        "mean_session_load_28d": mean_session_load_28d,
        "acwr3_freq": acwr3_freq,
        "acwr3_load": acwr3_load,
        "acwr_freq": acwr_freq,
        "acwr_load": acwr_load,
        "prev_rpe": prev_rpe,
        "prev_session_load": prev_session_load,
        "prev_has_note": prev_has_note,
        "prob_fatiga_prev_session": prob_fatiga_prev_session,
        "prob_fatiga_last_note": prob_fatiga_last_note,
        "days_since_last_note": days_since_last_note,
    }
    try:
        data = _post("/predict", payload)
        proba_pct = round(data["predicted_proba"] * 100, 1)
        label = data["risk_label"].upper()
        emoji = {"BAJO": "🟢", "MODERADO": "🟡", "ALTO": "🔴"}.get(label, "⚪")
        return (
            f"{emoji} Riesgo de overreaching: {label}\n"
            f"   Probabilidad: {proba_pct}%\n"
            f"   Sesión: {data['session_id']}"
        )
    except httpx.HTTPStatusError as exc:
        return f"❌ Error de la API ({exc.response.status_code}): {exc.response.text}"
    except Exception as exc:
        return f"❌ Error inesperado: {exc}"


@mcp.tool()
def explain_overreaching(
    sessions_last_3d: float,
    sessions_last_7d: float,
    sessions_last_28d: float,
    load_last_3d: float,
    load_last_7d: float,
    acwr_valid: int,
    days_since_first_session: int,
    notes_last_28d: int,
    level: str,
    days_since_prev_session: float | None = None,
    mean_session_load_28d: float | None = None,
    acwr3_freq: float | None = None,
    acwr3_load: float | None = None,
    acwr_freq: float | None = None,
    acwr_load: float | None = None,
    prev_rpe: float | None = None,
    prev_session_load: float | None = None,
    prev_has_note: float | None = None,
    prob_fatiga_prev_session: float | None = None,
    prob_fatiga_last_note: float | None = None,
    days_since_last_note: float | None = None,
    top_n: int = 5,
) -> str:
    """
    Predice el riesgo de overreaching Y explica los factores principales que lo causan.

    Devuelve la probabilidad, el nivel de riesgo, y los top_n factores más influyentes
    con su dirección (↑ empuja hacia overreaching, ↓ lo reduce).

    Parámetros: los mismos que predict_overreaching, más:
    - top_n: número de factores a mostrar (por defecto 5)
    """
    import datetime
    auto_id = f"{level.upper()}_{datetime.date.today().isoformat()}"
    payload = {
        "session_id": auto_id,
        "sessions_last_3d": sessions_last_3d,
        "sessions_last_7d": sessions_last_7d,
        "sessions_last_28d": sessions_last_28d,
        "load_last_3d": load_last_3d,
        "load_last_7d": load_last_7d,
        "acwr_valid": acwr_valid,
        "days_since_first_session": days_since_first_session,
        "notes_last_28d": notes_last_28d,
        "level": level,
        "days_since_prev_session": days_since_prev_session,
        "mean_session_load_28d": mean_session_load_28d,
        "acwr3_freq": acwr3_freq,
        "acwr3_load": acwr3_load,
        "acwr_freq": acwr_freq,
        "acwr_load": acwr_load,
        "prev_rpe": prev_rpe,
        "prev_session_load": prev_session_load,
        "prev_has_note": prev_has_note,
        "prob_fatiga_prev_session": prob_fatiga_prev_session,
        "prob_fatiga_last_note": prob_fatiga_last_note,
        "days_since_last_note": days_since_last_note,
    }
    try:
        data = _post("/explain", payload)
        proba_pct = round(data["predicted_proba"] * 100, 1)
        label = data["risk_label"].upper()
        emoji = {"BAJO": "🟢", "MODERADO": "🟡", "ALTO": "🔴"}.get(label, "⚪")

        lines = [
            f"{emoji} Riesgo de overreaching: {label}  ({proba_pct}%)",
            f"   Sesión: {data['session_id']}",
            "",
            f"Factores principales (top {top_n}):",
        ]

        # Nombres amigables para las features
        LABELS = {
            "load_last_3d":              "Carga últimos 3 días",
            "load_last_7d":              "Carga últimos 7 días",
            "mean_session_load_28d":     "Carga media mensual",
            "sessions_last_28d":         "Sesiones último mes",
            "sessions_last_7d":          "Sesiones última semana",
            "sessions_last_3d":          "Sesiones últimos 3 días",
            "acwr_load":                 "ACWR de carga",
            "acwr_freq":                 "ACWR de frecuencia",
            "days_since_prev_session":   "Días desde última sesión",
            "prev_rpe":                  "RPE sesión anterior",
            "prob_fatiga_prev_session":  "Fatiga subjetiva previa",
            "prob_fatiga_last_note":     "Fatiga en última nota",
            "days_since_first_session":  "Días de historial",
            "level_enc":                 "Nivel del atleta",
            "prev_session_load":         "Carga sesión anterior",
            "fatigue_signal":            "Señal de fatiga combinada",
            "acwr_load_risk":            "Alerta ACWR carga",
            "acwr_freq_risk":            "Alerta ACWR frecuencia",
        }

        drivers = data.get("top_drivers", [])[:top_n]
        for d in drivers:
            name = LABELS.get(d["feature"], d["feature"])
            sign = "↑ sube riesgo" if d["shap_value"] > 0 else "↓ baja riesgo"
            lines.append(f"  • {name}: {sign} (valor={d['feature_value']})")

        return "\n".join(lines)

    except httpx.HTTPStatusError as exc:
        return f"❌ Error de la API ({exc.response.status_code}): {exc.response.text}"
    except Exception as exc:
        return f"❌ Error inesperado: {exc}"


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Iniciando servidor MCP wodpredict-ai (stdio) → API en %s", API_URL)
    mcp.run(transport="stdio")
