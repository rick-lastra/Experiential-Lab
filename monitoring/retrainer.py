"""
Reentrenamiento automático para WODPredict-AI — Fase 7.

Orquesta el ciclo completo de monitoreo → decisión → reentrenamiento:

  1. Carga el dataset actual desde HuggingFace
  2. Construye features
  3. Ejecuta detección de drift contra el modelo en producción
  4. Si el drift supera el umbral, reentrena y reemplaza los artefactos
  5. Registra el evento en monitoring/retrain_log.jsonl

Uso como módulo:
    from monitoring.retrainer import maybe_retrain
    result = maybe_retrain()

Uso directo (p.ej. cron diario):
    python -m monitoring.retrainer
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from features.builder import build_features
from ingestion.loader import load_dataset
from model.trainer import MODEL_PATH, MEDIANS_PATH, train
from monitoring.drift import DriftDetector, DriftReport

logger = logging.getLogger(__name__)

RETRAIN_LOG_PATH = Path(__file__).parent / "retrain_log.jsonl"

# Umbral: reentrenar si hay al menos N features con drift severo
SEVERE_DRIFT_THRESHOLD = 3


# ---------------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------------

@dataclass
class RetrainResult:
    """Resultado del ciclo de monitoreo + reentrenamiento."""

    drift_report:    DriftReport
    retrained:       bool
    reason:          str
    timestamp:       str
    roc_auc_new:     float | None = None
    roc_auc_prev:    float | None = None

    def summary(self) -> str:
        action = "✅ REENTRENADO" if self.retrained else "⏭️  Sin reentrenamiento"
        lines = [
            f"=== Ciclo de Monitoreo ===",
            f"  Timestamp   : {self.timestamp}",
            f"  Acción      : {action}",
            f"  Motivo      : {self.reason}",
        ]
        if self.retrained and self.roc_auc_new is not None:
            lines += [
                f"  ROC-AUC anterior : {self.roc_auc_prev:.4f}" if self.roc_auc_prev else "",
                f"  ROC-AUC nuevo    : {self.roc_auc_new:.4f}",
            ]
        lines.append("")
        lines.append(self.drift_report.summary())
        return "\n".join(l for l in lines if l is not None)


# ---------------------------------------------------------------------------
# Lógica principal
# ---------------------------------------------------------------------------

def maybe_retrain(
    token: str | None = None,
    severe_threshold: int = SEVERE_DRIFT_THRESHOLD,
    force: bool = False,
) -> RetrainResult:
    """
    Ejecuta el ciclo de monitoreo y, si se detecta drift suficiente, reentrena.

    Args:
        token:            HuggingFace token opcional.
        severe_threshold: Número de features con drift severo para disparar reentrenamiento.
        force:            Si True, reentrena independientemente del drift.

    Returns:
        RetrainResult con el reporte de drift y el estado del reentrenamiento.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    logger.info("Iniciando ciclo de monitoreo [%s]", timestamp)

    # 1. Cargar datos actuales
    logger.info("Cargando dataset para monitoreo...")
    bundle = load_dataset(token=token, validate=False)
    fm = build_features(bundle)

    # 2. Detectar drift
    # Usamos la segunda mitad del dataset como "producción" simulada
    # (en un sistema real, X_new sería datos recientes de la BD de producción)
    n = len(fm.X)
    split = n // 2
    X_ref = fm.X.iloc[:split]
    X_new = fm.X.iloc[split:]

    logger.info(
        "Comparando ref=%d filas vs new=%d filas", len(X_ref), len(X_new)
    )
    detector = DriftDetector.from_dataframe(X_ref)
    drift_report = detector.detect(X_new, save_report=True)

    # 3. Decidir si reentrenar
    n_severe = len(drift_report.severe_features)
    if force:
        reason = "Reentrenamiento forzado (force=True)"
        should_retrain = True
    elif n_severe >= severe_threshold:
        reason = (
            f"Drift severo en {n_severe} features "
            f"(umbral={severe_threshold}): {drift_report.severe_features}"
        )
        should_retrain = True
    elif drift_report.has_drift:
        reason = (
            f"Drift detectado pero bajo el umbral de severidad "
            f"({n_severe}/{severe_threshold} features severas)"
        )
        should_retrain = False
    else:
        reason = "Sin drift significativo"
        should_retrain = False

    logger.info("Decisión: %s → %s", reason, "REENTRENAR" if should_retrain else "MANTENER")

    # 4. Reentrenar si procede
    roc_auc_new = None
    roc_auc_prev = None

    if should_retrain:
        # Guardar ROC-AUC previo si existe el modelo
        if MODEL_PATH.exists():
            from model.trainer import _evaluate
            from xgboost import XGBClassifier
            prev_model = XGBClassifier()
            prev_model.load_model(str(MODEL_PATH))
            metrics = _evaluate(prev_model, X_new, fm.y.iloc[split:])
            roc_auc_prev = metrics["roc_auc"]
            logger.info("ROC-AUC modelo anterior: %.4f", roc_auc_prev)

        logger.info("Iniciando reentrenamiento...")
        train_result = train(token=token, save=True)
        roc_auc_new = train_result.roc_auc_test
        logger.info("Reentrenamiento completado. ROC-AUC nuevo: %.4f", roc_auc_new)

    # 5. Registrar en log
    result = RetrainResult(
        drift_report=drift_report,
        retrained=should_retrain,
        reason=reason,
        timestamp=timestamp,
        roc_auc_new=roc_auc_new,
        roc_auc_prev=roc_auc_prev,
    )
    _append_log(result)

    return result


def _append_log(result: RetrainResult) -> None:
    """Añade una entrada al log de reentrenamiento (JSONL)."""
    entry = {
        "timestamp":    result.timestamp,
        "retrained":    result.retrained,
        "reason":       result.reason,
        "n_severe":     len(result.drift_report.severe_features),
        "severe_features": result.drift_report.severe_features,
        "drifted_features": result.drift_report.drifted_features,
        "roc_auc_prev": result.roc_auc_prev,
        "roc_auc_new":  result.roc_auc_new,
    }
    with open(RETRAIN_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    logger.info("Log actualizado: %s", RETRAIN_LOG_PATH)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = maybe_retrain()
    print(result.summary())
