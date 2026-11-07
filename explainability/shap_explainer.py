"""
Explicabilidad SHAP para WODPredict-AI — Fase 4.

Proporciona dos niveles de explicación:
  - Global  : importancia media |SHAP| sobre un conjunto de referencia
  - Local   : vector SHAP para una predicción individual

El explainer usa TreeExplainer (exacto y rápido para XGBoost).
Las salidas están diseñadas para ser consumidas directamente por la API FastAPI.

Uso como módulo:
    from explainability.shap_explainer import SHAPExplainer
    explainer = SHAPExplainer.from_artifacts()
    local     = explainer.explain_local(row_df)
    global_   = explainer.explain_global(X_background)

Uso directo:
    python -m explainability.shap_explainer
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

from config import MEDIANS_PATH, MODEL_PATH
from features.builder import MODEL_FEATURES, build_features
from ingestion.loader import load_dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Estructuras de salida
# ---------------------------------------------------------------------------

@dataclass
class LocalExplanation:
    """Explicación SHAP para una sola predicción."""

    session_id:   str
    predicted_proba: float                    # P(overreaching=1)
    base_value:   float                       # valor esperado del modelo
    shap_values:  dict[str, float]            # feature → contribución SHAP
    top_drivers:  list[dict]                  # top N features ordenadas por |SHAP|

    def to_dict(self) -> dict:
        return {
            "session_id":       self.session_id,
            "predicted_proba":  round(self.predicted_proba, 4),
            "base_value":       round(self.base_value, 4),
            "shap_values":      {k: round(v, 4) for k, v in self.shap_values.items()},
            "top_drivers":      self.top_drivers,
        }


@dataclass
class GlobalExplanation:
    """Importancia global media |SHAP| sobre un conjunto de referencia."""

    mean_abs_shap: dict[str, float]           # feature → media |SHAP|
    n_samples:     int

    def to_dict(self) -> dict:
        return {
            "n_samples":     self.n_samples,
            "mean_abs_shap": {k: round(v, 6) for k, v in self.mean_abs_shap.items()},
        }

    def top_features(self, n: int = 10) -> list[dict]:
        """Retorna las top-N features ordenadas por importancia descendente."""
        ranked = sorted(self.mean_abs_shap.items(), key=lambda x: x[1], reverse=True)
        return [{"feature": f, "mean_abs_shap": round(v, 6)} for f, v in ranked[:n]]


# ---------------------------------------------------------------------------
# Explainer principal
# ---------------------------------------------------------------------------

class SHAPExplainer:
    """
    Wrapper de shap.TreeExplainer sobre el XGBClassifier entrenado.

    Attributes:
        model:      XGBClassifier cargado desde artefactos.
        explainer:  shap.TreeExplainer inicializado con el modelo.
        medians:    dict con medianas de imputación para preprocesar entradas.
    """

    def __init__(self, model: XGBClassifier, medians: dict[str, float]):
        self.model    = model
        self.medians  = medians
        self.explainer = shap.TreeExplainer(
            model,
            feature_perturbation="tree_path_dependent",
        )
        logger.info("SHAPExplainer inicializado (TreeExplainer, tree_path_dependent)")

    # ------------------------------------------------------------------ #
    # Constructores alternativos                                           #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_artifacts(
        cls,
        model_path:   Path | str = MODEL_PATH,
        medians_path: Path | str = MEDIANS_PATH,
    ) -> "SHAPExplainer":
        """Carga el modelo y las medianas desde los artefactos entrenados."""
        model = XGBClassifier()
        model.load_model(str(model_path))
        medians = json.loads(Path(medians_path).read_text(encoding="utf-8"))
        logger.info("Modelo cargado desde %s", model_path)
        return cls(model=model, medians=medians)

    # ------------------------------------------------------------------ #
    # Preprocesado                                                         #
    # ------------------------------------------------------------------ #

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        """Garantiza columnas en orden, fuerza dtype float64 y aplica imputación de medianas."""
        X = X[MODEL_FEATURES].astype(float)   # convierte None/object → NaN float64 en un solo paso
        for col, val in self.medians.items():
            if col in X.columns and X[col].isna().any():
                X[col] = X[col].fillna(val)
        return X

    # ------------------------------------------------------------------ #
    # Explicación local (una o varias filas)                               #
    # ------------------------------------------------------------------ #

    def explain_local(
        self,
        X: pd.DataFrame,
        session_ids: list[str] | None = None,
        top_n: int = 8,
    ) -> list[LocalExplanation]:
        """
        Genera explicaciones SHAP para cada fila de X.

        Args:
            X:           DataFrame con las features del modelo (MODEL_FEATURES).
            session_ids: Identificadores opcionales de cada fila.
            top_n:       Número de drivers a incluir en top_drivers.

        Returns:
            Lista de LocalExplanation, una por fila.
        """
        X_prep = self._prepare(X)
        n = len(X_prep)

        if session_ids is None:
            session_ids = [f"row_{i}" for i in range(n)]

        shap_out   = self.explainer(X_prep)          # shap.Explanation
        # Para clasificación binaria, shap_out.values tiene shape (n, p, 2)
        # o (n, p) según versión de shap; normalizamos a (n, p) para clase 1
        sv = shap_out.values
        if sv.ndim == 3:
            sv = sv[:, :, 1]                         # clase positiva

        base = shap_out.base_values
        if base.ndim == 2:
            base = base[:, 1]
        base = np.atleast_1d(base)

        probas = self.model.predict_proba(X_prep)[:, 1]

        results: list[LocalExplanation] = []
        for i in range(n):
            sv_row = sv[i]
            named  = dict(zip(MODEL_FEATURES, sv_row.tolist()))

            top_idx = np.argsort(np.abs(sv_row))[::-1][:top_n]
            top_drivers = [
                {
                    "feature":    MODEL_FEATURES[j],
                    "shap_value": round(float(sv_row[j]), 4),
                    "feature_value": round(float(X_prep.iloc[i, j]), 4),
                }
                for j in top_idx
            ]

            results.append(LocalExplanation(
                session_id=session_ids[i],
                predicted_proba=float(probas[i]),
                base_value=float(base[i]),
                shap_values=named,
                top_drivers=top_drivers,
            ))

        return results

    # ------------------------------------------------------------------ #
    # Explicación global                                                   #
    # ------------------------------------------------------------------ #

    def explain_global(
        self,
        X: pd.DataFrame,
        max_samples: int = 500,
    ) -> GlobalExplanation:
        """
        Calcula importancia global media |SHAP| sobre una muestra de X.

        Args:
            X:           DataFrame de referencia (training o background set).
            max_samples: Máximo de filas a usar (muestreo aleatorio si excede).

        Returns:
            GlobalExplanation con mean_abs_shap por feature.
        """
        X_prep = self._prepare(X)
        if len(X_prep) > max_samples:
            X_prep = X_prep.sample(max_samples, random_state=42)

        shap_out = self.explainer(X_prep)
        sv = shap_out.values
        if sv.ndim == 3:
            sv = sv[:, :, 1]

        mean_abs = np.abs(sv).mean(axis=0)
        ranked   = dict(zip(MODEL_FEATURES, mean_abs.tolist()))
        # Ordenar descendente
        ranked   = dict(sorted(ranked.items(), key=lambda x: x[1], reverse=True))

        logger.info(
            "GlobalExplanation calculada sobre %d muestras. Top feature: %s",
            len(X_prep),
            next(iter(ranked)),
        )

        return GlobalExplanation(mean_abs_shap=ranked, n_samples=len(X_prep))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("Cargando dataset y construyendo features...")
    bundle   = load_dataset(validate=False)
    fm       = build_features(bundle)

    print("Inicializando SHAPExplainer desde artefactos...")
    explainer = SHAPExplainer.from_artifacts()

    # Explicación global sobre muestra de 500 filas
    global_exp = explainer.explain_global(fm.X, max_samples=500)
    print("\n=== Importancia global (top 10) ===")
    for item in global_exp.top_features(10):
        print(f"  {item['mean_abs_shap']:.4f}  {item['feature']}")

    # Explicación local para las primeras 3 filas
    sample_X = fm.X.head(3).copy()
    local_exps = explainer.explain_local(sample_X, top_n=5)
    print("\n=== Explicación local (3 sesiones) ===")
    for exp in local_exps:
        print(f"\n  {exp.session_id}  P(overreaching)={exp.predicted_proba:.3f}")
        for d in exp.top_drivers:
            sign = "↑" if d["shap_value"] > 0 else "↓"
            print(f"    {sign} {d['feature']:30s}  SHAP={d['shap_value']:+.4f}  val={d['feature_value']:.3f}")
