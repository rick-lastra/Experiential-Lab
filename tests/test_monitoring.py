"""
Tests de monitoreo y reentrenamiento para WODPredict-AI — Fase 7.

Cubre:
  - _psi: cálculo correcto, sin varianza, distribuciones idénticas
  - DriftDetector: inicialización, detect() sin/con drift
  - DriftReport: has_drift, drifted_features, to_dict, summary
  - maybe_retrain: ciclo completo sin reentrenamiento (drift bajo umbral)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.builder import MODEL_FEATURES, build_features
from ingestion.loader import load_dataset
from monitoring.drift import (
    DriftDetector,
    DriftReport,
    FeatureDriftResult,
    KS_ALPHA,
    PSI_SEVERE,
    PSI_WARN,
    _psi,
    _psi_label,
)
from monitoring.retrainer import RetrainResult, maybe_retrain


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fm():
    bundle = load_dataset(validate=False)
    return build_features(bundle)


@pytest.fixture(scope="module")
def detector(fm):
    return DriftDetector.from_feature_matrix(fm)


@pytest.fixture(scope="module")
def X_identical(fm):
    """Mismo dataset que la referencia → sin drift."""
    return fm.X.sample(2000, random_state=0).reset_index(drop=True)


@pytest.fixture(scope="module")
def X_drifted(fm):
    """Dataset con drift artificial: multiplicamos las columnas de carga × 3."""
    X = fm.X.sample(2000, random_state=1).reset_index(drop=True).copy()
    for col in ["load_last_3d", "load_last_7d", "mean_session_load_28d",
                "load_ratio_3_28", "load_ratio_7_28"]:
        X[col] = X[col] * 3.0
    return X


# ---------------------------------------------------------------------------
# _psi
# ---------------------------------------------------------------------------

class TestPSI:
    def test_identical_distributions_near_zero(self):
        arr = np.random.default_rng(0).normal(0, 1, 1000)
        assert _psi(arr, arr) < 0.01

    def test_different_distributions_high_psi(self):
        ref = np.random.default_rng(0).normal(0, 1, 1000)
        new = np.random.default_rng(1).normal(5, 1, 1000)  # media muy diferente
        assert _psi(ref, new) > PSI_SEVERE

    def test_no_variance_returns_zero(self):
        arr = np.ones(100)  # sin varianza
        assert _psi(arr, arr) == 0.0

    def test_returns_nonnegative(self):
        ref = np.random.default_rng(2).uniform(0, 1, 500)
        new = np.random.default_rng(3).uniform(0.3, 1.3, 500)
        assert _psi(ref, new) >= 0.0


# ---------------------------------------------------------------------------
# _psi_label
# ---------------------------------------------------------------------------

class TestPSILabel:
    def test_none(self):
        assert _psi_label(0.05) == "none"

    def test_moderate(self):
        assert _psi_label(PSI_WARN) == "moderate"
        assert _psi_label(0.15) == "moderate"

    def test_severe(self):
        assert _psi_label(PSI_SEVERE) == "severe"
        assert _psi_label(0.5) == "severe"


# ---------------------------------------------------------------------------
# DriftDetector — inicialización
# ---------------------------------------------------------------------------

class TestDriftDetectorInit:
    def test_from_feature_matrix(self, fm):
        d = DriftDetector.from_feature_matrix(fm)
        assert d.n_reference == len(fm.X)

    def test_from_dataframe(self, fm):
        d = DriftDetector.from_dataframe(fm.X)
        assert d.n_reference == len(fm.X)

    def test_missing_features_raises(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="Faltan features"):
            DriftDetector(reference=df)


# ---------------------------------------------------------------------------
# DriftDetector — detect() sin drift
# ---------------------------------------------------------------------------

class TestDetectNoDrift:
    def test_returns_drift_report(self, detector, X_identical):
        report = detector.detect(X_identical)
        assert isinstance(report, DriftReport)

    def test_results_count_matches_features(self, detector, X_identical):
        report = detector.detect(X_identical)
        assert len(report.results) == len(MODEL_FEATURES)

    def test_no_drift_on_identical_sample(self, detector, X_identical):
        report = detector.detect(X_identical)
        # Con la misma distribución, el drift severo debe ser mínimo
        assert len(report.severe_features) == 0

    def test_n_reference_stored(self, detector, X_identical):
        report = detector.detect(X_identical)
        assert report.n_reference == detector.n_reference

    def test_n_new_stored(self, detector, X_identical):
        report = detector.detect(X_identical)
        assert report.n_new == len(X_identical)


# ---------------------------------------------------------------------------
# DriftDetector — detect() con drift artificial
# ---------------------------------------------------------------------------

class TestDetectWithDrift:
    def test_detects_drift_in_load_features(self, detector, X_drifted):
        report = detector.detect(X_drifted)
        assert report.has_drift

    def test_load_features_in_drifted(self, detector, X_drifted):
        report = detector.detect(X_drifted)
        assert "load_last_3d" in report.drifted_features or \
               "load_last_7d" in report.drifted_features

    def test_feature_subset(self, detector, X_drifted):
        """detect() con subset de features solo analiza esas."""
        report = detector.detect(X_drifted, features=["load_last_3d"])
        assert len(report.results) == 1
        assert report.results[0].feature == "load_last_3d"


# ---------------------------------------------------------------------------
# DriftReport
# ---------------------------------------------------------------------------

class TestDriftReport:
    def _make_report(self, has_drift: bool) -> DriftReport:
        r = FeatureDriftResult(
            feature="load_last_3d",
            ks_statistic=0.3 if has_drift else 0.01,
            ks_pvalue=0.001 if has_drift else 0.9,
            psi=0.25 if has_drift else 0.01,
            drift_ks=has_drift,
            drift_psi="severe" if has_drift else "none",
        )
        return DriftReport(n_reference=1000, n_new=500, results=[r])

    def test_has_drift_true(self):
        assert self._make_report(True).has_drift is True

    def test_has_drift_false(self):
        assert self._make_report(False).has_drift is False

    def test_drifted_features_populated(self):
        report = self._make_report(True)
        assert "load_last_3d" in report.drifted_features

    def test_severe_features_populated(self):
        report = self._make_report(True)
        assert "load_last_3d" in report.severe_features

    def test_to_dict_serializable(self):
        import json
        report = self._make_report(True)
        json.dumps(report.to_dict())

    def test_summary_contains_status(self):
        assert "DRIFT" in self._make_report(True).summary()
        assert "Sin drift" in self._make_report(False).summary()


# ---------------------------------------------------------------------------
# maybe_retrain — integración (sin reentrenamiento real)
# ---------------------------------------------------------------------------

class TestMaybeRetrain:
    def test_returns_retrain_result(self):
        result = maybe_retrain(severe_threshold=999)  # umbral inalcanzable → no reentrena
        assert isinstance(result, RetrainResult)

    def test_no_retrain_below_threshold(self):
        result = maybe_retrain(severe_threshold=999)
        assert result.retrained is False

    def test_drift_report_present(self):
        result = maybe_retrain(severe_threshold=999)
        assert isinstance(result.drift_report, DriftReport)

    def test_log_file_created(self):
        from monitoring.retrainer import RETRAIN_LOG_PATH
        maybe_retrain(severe_threshold=999)
        assert RETRAIN_LOG_PATH.exists()

    def test_summary_contains_timestamp(self):
        result = maybe_retrain(severe_threshold=999)
        assert "Timestamp" in result.summary()
