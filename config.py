"""
Configuración de rutas de artefactos para WODPredict-AI.

Módulo ligero para que api/ y explainability/ puedan importar
las rutas de los artefactos sin depender de model.trainer
(que arrastra dependencias de entrenamiento innecesarias en producción).
"""
from __future__ import annotations

from pathlib import Path

# Directorio base — resuelto relativo a este archivo
_BASE = Path(__file__).parent

ARTIFACTS_DIR = _BASE / "model" / "artifacts"
MODEL_PATH    = ARTIFACTS_DIR / "xgb_overreaching.json"
MEDIANS_PATH  = ARTIFACTS_DIR / "imputation_medians.json"
