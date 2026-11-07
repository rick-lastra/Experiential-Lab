"""
Detección de drift para WODPredict-AI — Fase 7.

Compara la distribución de un conjunto de datos nuevo (producción)
contra la distribución de referencia (training) usando dos métodos:

  - KS test (Kolmogorov-Smirnov): detecta diferencias en la distribución
    de cada feature numérica. p-valor < umbral → drift detectado.

  - PSI (Population Stability Index): mide cuánto ha cambiado la distribución
    en bins fijos. PSI > 0.2 → drift severo, 0.1–0.2 → drift moderado.

Uso:
    from monitoring.drift import DriftDetector
    detector = DriftDetector.from_feature_matrix(fm_train)
    report   = detector.detect(X_new)
    if report.has_drift:
        print(report.summary())
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from features.builder import MODEL_FEATURES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

KS_ALPHA      = 0.05    # p-valor mínimo para considerar drift (KS test)
PSI_WARN      = 0.10    # PSI moderado
PSI_SEVERE    = 0.20    # PSI severo
PSI_BINS      = 10      # número de bins para PSI
DRIFT_REPORT_PATH = Path(__file__).parent / "drift_report.json"


# ---------------------------------------------------------------------------
# Estructuras de resultado
# ---------------------------------------------------------------------------

@dataclass
class FeatureDriftResult:
    """Resultado de drift para una feature."""
    feature:      str
    ks_statistic: float
    ks_pvalue:    float
    psi:          float
    drift_ks:     bool    # True si KS detectó drift
    drift_psi:    str     # "none" | "moderate" | "severe"

    @property
    def has_drift(self) -> bool:
        return self.drift_ks or self.drift_psi != "none"


@dataclass
class DriftReport:
    """Resumen de drift sobre todas las features."""
    n_reference:  int
    n_new:        int
    results:      list[FeatureDriftResult] = field(default_factory=list)

    @property
    def has_drift(self) -> bool:
        return any(r.has_drift for r in self.results)

    @property
    def drifted_features(self) -> list[str]:
        return [r.feature for r in self.results if r.has_drift]

    @property
    def severe_features(self) -> list[str]:
        return [r.feature for r in self.results if r.drift_psi == "severe"]

    def summary(self) -> str:
        status = "⚠️  DRIFT DETECTADO" if self.has_drift else "✅ Sin drift significativo"
        lines = [
            f"=== Reporte de Drift ===",
            f"  Referencia : {self.n_reference:,} filas",
            f"  Nuevos     : {self.n_new:,} filas",
            f"  Estado     : {status}",
        ]
        if self.has_drift:
            lines.append(f"  Features con drift ({len(self.drifted_features)}):")
            for r in sorted(self.results, key=lambda x: x.psi, reverse=True):
                if r.has_drift:
                    lines.append(
                        f"    • {r.feature:<35s} KS-p={r.ks_pvalue:.3f}  PSI={r.psi:.3f}  [{r.drift_psi}]"
                    )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "n_reference": self.n_reference,
            "n_new":       self.n_new,
            "has_drift":   self.has_drift,
            "drifted_features": self.drifted_features,
            "severe_features":  self.severe_features,
            "feature_results": [
                {
                    "feature":      r.feature,
                    "ks_statistic": round(r.ks_statistic, 4),
                    "ks_pvalue":    round(r.ks_pvalue, 4),
                    "psi":          round(r.psi, 4),
                    "drift_ks":     bool(r.drift_ks),
                    "drift_psi":    r.drift_psi,
                }
                for r in self.results
            ],
        }


# ---------------------------------------------------------------------------
# Cálculo de PSI
# ---------------------------------------------------------------------------

def _psi(ref: np.ndarray, new: np.ndarray, bins: int = PSI_BINS) -> float:
    """
    Population Stability Index entre ref (referencia) y new (producción).
    PSI = Σ (P_new - P_ref) × ln(P_new / P_ref)
    """
    # Usar los percentiles de referencia como bordes de bins
    bin_edges = np.nanpercentile(ref, np.linspace(0, 100, bins + 1))
    bin_edges = np.unique(bin_edges)  # eliminar duplicados si hay poca varianza

    if len(bin_edges) < 2:
        return 0.0  # sin varianza → no hay drift posible

    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    new_counts, _ = np.histogram(new, bins=bin_edges)

    # Evitar división por cero con suavizado mínimo
    eps = 1e-4
    ref_pct = (ref_counts + eps) / (ref_counts.sum() + eps * len(ref_counts))
    new_pct = (new_counts + eps) / (new_counts.sum() + eps * len(new_counts))

    psi = float(np.sum((new_pct - ref_pct) * np.log(new_pct / ref_pct)))
    return max(psi, 0.0)  # PSI siempre ≥ 0


def _psi_label(psi: float) -> str:
    if psi >= PSI_SEVERE:
        return "severe"
    if psi >= PSI_WARN:
        return "moderate"
    return "none"


# ---------------------------------------------------------------------------
# Detector principal
# ---------------------------------------------------------------------------

class DriftDetector:
    """
    Detecta drift estadístico comparando distribuciones de features.

    Attributes:
        reference_stats: estadísticas de referencia por feature
                         (guardadas durante el fit para evitar re-descargar datos)
    """

    def __init__(self, reference: pd.DataFrame):
        """
        Args:
            reference: DataFrame con las features del training set (MODEL_FEATURES).
        """
        if not all(f in reference.columns for f in MODEL_FEATURES):
            missing = [f for f in MODEL_FEATURES if f not in reference.columns]
            raise ValueError(f"Faltan features en la referencia: {missing}")

        self._ref: dict[str, np.ndarray] = {
            feat: reference[feat].dropna().values
            for feat in MODEL_FEATURES
        }
        self.n_reference = len(reference)
        logger.info(
            "DriftDetector inicializado con %d filas de referencia, %d features",
            self.n_reference, len(MODEL_FEATURES),
        )

    # ------------------------------------------------------------------ #
    # Constructores alternativos                                           #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_feature_matrix(cls, fm) -> "DriftDetector":
        """Inicializa desde un FeatureMatrix (salida de build_features)."""
        return cls(reference=fm.X)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "DriftDetector":
        return cls(reference=df)

    # ------------------------------------------------------------------ #
    # Detección                                                            #
    # ------------------------------------------------------------------ #

    def detect(
        self,
        X_new: pd.DataFrame,
        features: list[str] | None = None,
        save_report: bool = False,
    ) -> DriftReport:
        """
        Compara X_new contra la referencia y devuelve un DriftReport.

        Args:
            X_new:       DataFrame nuevo a comparar (MODEL_FEATURES).
            features:    Subset de features a analizar. None = todas.
            save_report: Si True, guarda el reporte en monitoring/drift_report.json.

        Returns:
            DriftReport con resultados por feature.
        """
        features_to_check = features or MODEL_FEATURES
        results: list[FeatureDriftResult] = []

        for feat in features_to_check:
            if feat not in self._ref or feat not in X_new.columns:
                continue

            ref_vals = self._ref[feat]
            new_vals = X_new[feat].dropna().values

            if len(ref_vals) < 10 or len(new_vals) < 10:
                logger.warning("Feature %s: muestra insuficiente para KS/PSI, se omite", feat)
                continue

            # KS test
            ks_stat, ks_pval = stats.ks_2samp(ref_vals, new_vals)

            # PSI
            psi_val = _psi(ref_vals, new_vals)

            results.append(FeatureDriftResult(
                feature=feat,
                ks_statistic=float(ks_stat),
                ks_pvalue=float(ks_pval),
                psi=psi_val,
                drift_ks=(ks_pval < KS_ALPHA),
                drift_psi=_psi_label(psi_val),
            ))

        report = DriftReport(
            n_reference=self.n_reference,
            n_new=len(X_new),
            results=results,
        )

        if report.has_drift:
            logger.warning(
                "Drift detectado en %d features: %s",
                len(report.drifted_features), report.drifted_features,
            )
        else:
            logger.info("Sin drift significativo (KS α=%.2f, PSI warn=%.2f)", KS_ALPHA, PSI_WARN)

        if save_report:
            DRIFT_REPORT_PATH.write_text(
                json.dumps(report.to_dict(), indent=2), encoding="utf-8"
            )
            logger.info("Reporte guardado en %s", DRIFT_REPORT_PATH)

        return report
