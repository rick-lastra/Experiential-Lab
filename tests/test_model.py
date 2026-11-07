"""
Tests del modelo XGBoost para WODPredict-AI — Fase 3.

Cubre:
  - _temporal_split: proporciones, sin solapamiento, orden temporal
  - _evaluate: métricas en rango válido
  - train (integración): artefactos guardados, métricas razonables,
    modelo cargable, predicciones con la forma correcta
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from features.builder import FeatureMatrix, MODEL_FEATURES, build_features
from ingestion.loader import load_dataset
from model.trainer import (
    ARTIFACTS_DIR,
    MEDIANS_PATH,
    MODEL_PATH,
    TrainResult,
    _evaluate,
    _temporal_split,
    train,
)


# ---------------------------------------------------------------------------
# Fixtures compartidos (descarga una sola vez por módulo)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bundle():
    return load_dataset(validate=False)


@pytest.fixture(scope="module")
def fm(bundle):
    return build_features(bundle)


@pytest.fixture(scope="module")
def train_result():
    """Ejecuta el entrenamiento completo una sola vez para todos los tests."""
    return train(save=True)


# ---------------------------------------------------------------------------
# _temporal_split
# ---------------------------------------------------------------------------

class TestTemporalSplit:
    def _make_fm(self, n: int) -> tuple[FeatureMatrix, pd.DataFrame]:
        X = pd.DataFrame(np.random.rand(n, len(MODEL_FEATURES)), columns=MODEL_FEATURES)
        y = pd.Series(np.random.randint(0, 2, n))
        fm = FeatureMatrix(X=X, y=y)
        # Fechas en orden creciente
        dates = pd.date_range("2024-01-01", periods=n, freq="D").astype(str)
        return fm, pd.DataFrame({"date": dates})

    def test_sizes_sum_to_total(self):
        n = 1000
        fm, sf = self._make_fm(n)
        X_tr, y_tr, X_v, y_v, X_te, y_te = _temporal_split(fm, sf)
        assert len(X_tr) + len(X_v) + len(X_te) == n

    def test_no_overlap_in_indices(self):
        n = 1000
        fm, sf = self._make_fm(n)
        X_tr, _, X_v, _, X_te, _ = _temporal_split(fm, sf)
        train_i = set(X_tr.index)
        val_i   = set(X_v.index)
        test_i  = set(X_te.index)
        assert train_i.isdisjoint(val_i)
        assert train_i.isdisjoint(test_i)
        assert val_i.isdisjoint(test_i)

    def test_val_test_default_fracs(self):
        n = 1000
        fm, sf = self._make_fm(n)
        _, _, X_v, _, X_te, _ = _temporal_split(fm, sf)
        assert abs(len(X_v)  / n - 0.10) < 0.02
        assert abs(len(X_te) / n - 0.10) < 0.02

    def test_custom_fracs(self):
        n = 500
        fm, sf = self._make_fm(n)
        _, _, X_v, _, X_te, _ = _temporal_split(fm, sf, val_frac=0.2, test_frac=0.2)
        assert abs(len(X_v)  / n - 0.20) < 0.02
        assert abs(len(X_te) / n - 0.20) < 0.02


# ---------------------------------------------------------------------------
# _evaluate
# ---------------------------------------------------------------------------

class TestEvaluate:
    def _fit_dummy(self):
        X = pd.DataFrame(np.random.rand(200, 3), columns=["a", "b", "c"])
        y = pd.Series([0] * 100 + [1] * 100)
        m = XGBClassifier(n_estimators=5, verbosity=0)
        m.fit(X, y)
        return m, X, y

    def test_metrics_in_valid_range(self):
        model, X, y = self._fit_dummy()
        metrics = _evaluate(model, X, y)
        assert 0.0 <= metrics["roc_auc"] <= 1.0
        assert 0.0 <= metrics["pr_auc"]  <= 1.0
        assert 0.0 <= metrics["f1"]      <= 1.0

    def test_returns_three_keys(self):
        model, X, y = self._fit_dummy()
        metrics = _evaluate(model, X, y)
        assert set(metrics.keys()) == {"roc_auc", "pr_auc", "f1"}


# ---------------------------------------------------------------------------
# train — integración completa
# ---------------------------------------------------------------------------

class TestTrain:
    def test_returns_train_result(self, train_result):
        assert isinstance(train_result, TrainResult)

    def test_split_sizes_positive(self, train_result):
        assert train_result.n_train > 0
        assert train_result.n_val   > 0
        assert train_result.n_test  > 0

    def test_roc_auc_above_chance(self, train_result):
        """ROC-AUC debe superar 0.60 en val y test."""
        assert train_result.roc_auc_val  > 0.60, f"Val ROC-AUC={train_result.roc_auc_val:.4f}"
        assert train_result.roc_auc_test > 0.60, f"Test ROC-AUC={train_result.roc_auc_test:.4f}"

    def test_pr_auc_positive(self, train_result):
        assert train_result.pr_auc_val  > 0.0
        assert train_result.pr_auc_test > 0.0

    def test_best_iteration_positive(self, train_result):
        assert train_result.best_iteration > 0

    def test_model_artifact_exists(self, train_result):
        assert train_result.model_path.exists(), f"No existe: {train_result.model_path}"

    def test_medians_artifact_exists(self, train_result):
        assert train_result.medians_path.exists(), f"No existe: {train_result.medians_path}"

    def test_medians_json_parseable(self, train_result):
        data = json.loads(train_result.medians_path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert len(data) > 0

    def test_feature_importances_all_features(self, train_result):
        assert set(train_result.feature_importances.keys()) == set(MODEL_FEATURES)

    def test_model_loadable_and_predicts(self, train_result, fm):
        """El modelo guardado debe poder cargarse y producir probabilidades."""
        loaded = XGBClassifier()
        loaded.load_model(str(train_result.model_path))
        proba = loaded.predict_proba(fm.X.head(10))
        assert proba.shape == (10, 2)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_summary_contains_metrics(self, train_result):
        summary = train_result.summary()
        assert "ROC-AUC" in summary
        assert "PR-AUC"  in summary
        assert "F1"      in summary
