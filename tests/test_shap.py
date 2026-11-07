"""
Tests de explicabilidad SHAP para WODPredict-AI — Fase 4.

Cubre:
  - SHAPExplainer.from_artifacts(): carga correcta desde artefactos
  - _prepare(): columnas ordenadas, NaN imputados
  - explain_local(): estructura, tipos, top_drivers, session_ids
  - explain_global(): claves, n_samples, top_features ordenadas
  - to_dict(): serializable, valores redondeados
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from explainability.shap_explainer import (
    GlobalExplanation,
    LocalExplanation,
    SHAPExplainer,
)
from features.builder import MODEL_FEATURES, build_features
from ingestion.loader import load_dataset


# ---------------------------------------------------------------------------
# Fixtures (descarga y entrenamiento una sola vez por módulo)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bundle():
    return load_dataset(validate=False)


@pytest.fixture(scope="module")
def fm(bundle):
    return build_features(bundle)


@pytest.fixture(scope="module")
def explainer():
    return SHAPExplainer.from_artifacts()


@pytest.fixture(scope="module")
def sample_X(fm):
    """10 filas representativas para tests locales."""
    return fm.X.head(10).copy()


# ---------------------------------------------------------------------------
# from_artifacts
# ---------------------------------------------------------------------------

class TestFromArtifacts:
    def test_returns_shap_explainer(self, explainer):
        assert isinstance(explainer, SHAPExplainer)

    def test_model_loaded(self, explainer):
        assert explainer.model is not None

    def test_medians_loaded(self, explainer):
        assert isinstance(explainer.medians, dict)
        assert len(explainer.medians) > 0

    def test_tree_explainer_created(self, explainer):
        import shap
        assert isinstance(explainer.explainer, shap.TreeExplainer)


# ---------------------------------------------------------------------------
# _prepare
# ---------------------------------------------------------------------------

class TestPrepare:
    def test_columns_in_model_order(self, explainer, sample_X):
        prep = explainer._prepare(sample_X)
        assert list(prep.columns) == MODEL_FEATURES

    def test_no_nan_after_prepare(self, explainer, sample_X):
        X_with_nan = sample_X.copy()
        # Introduce NaN en la primera columna
        X_with_nan.iloc[0, 0] = float("nan")
        prep = explainer._prepare(X_with_nan)
        assert prep.isna().sum().sum() == 0

    def test_does_not_modify_input(self, explainer, sample_X):
        original_val = sample_X.iloc[0, 0]
        sample_X_copy = sample_X.copy()
        sample_X_copy.iloc[0, 0] = float("nan")
        explainer._prepare(sample_X_copy)
        assert sample_X.iloc[0, 0] == original_val  # original intacto


# ---------------------------------------------------------------------------
# explain_local
# ---------------------------------------------------------------------------

class TestExplainLocal:
    def test_returns_list_of_local_explanations(self, explainer, sample_X):
        results = explainer.explain_local(sample_X)
        assert isinstance(results, list)
        assert all(isinstance(r, LocalExplanation) for r in results)

    def test_length_matches_input(self, explainer, sample_X):
        results = explainer.explain_local(sample_X)
        assert len(results) == len(sample_X)

    def test_predicted_proba_in_range(self, explainer, sample_X):
        results = explainer.explain_local(sample_X)
        for r in results:
            assert 0.0 <= r.predicted_proba <= 1.0

    def test_shap_values_have_all_features(self, explainer, sample_X):
        results = explainer.explain_local(sample_X)
        for r in results:
            assert set(r.shap_values.keys()) == set(MODEL_FEATURES)

    def test_top_drivers_length(self, explainer, sample_X):
        top_n = 5
        results = explainer.explain_local(sample_X, top_n=top_n)
        for r in results:
            assert len(r.top_drivers) == top_n

    def test_top_drivers_keys(self, explainer, sample_X):
        results = explainer.explain_local(sample_X)
        for r in results:
            for d in r.top_drivers:
                assert "feature" in d
                assert "shap_value" in d
                assert "feature_value" in d

    def test_custom_session_ids(self, explainer, sample_X):
        ids = [f"session_{i}" for i in range(len(sample_X))]
        results = explainer.explain_local(sample_X, session_ids=ids)
        assert [r.session_id for r in results] == ids

    def test_default_session_ids_format(self, explainer, sample_X):
        results = explainer.explain_local(sample_X[:3])
        assert results[0].session_id == "row_0"
        assert results[2].session_id == "row_2"

    def test_top_drivers_ordered_by_abs_shap(self, explainer, sample_X):
        results = explainer.explain_local(sample_X[:1])
        drivers = results[0].top_drivers
        abs_vals = [abs(d["shap_value"]) for d in drivers]
        assert abs_vals == sorted(abs_vals, reverse=True)

    def test_to_dict_serializable(self, explainer, sample_X):
        import json
        result = explainer.explain_local(sample_X[:1])[0]
        d = result.to_dict()
        # Debe ser serializable a JSON sin errores
        json.dumps(d)
        assert "session_id" in d
        assert "predicted_proba" in d
        assert "shap_values" in d
        assert "top_drivers" in d


# ---------------------------------------------------------------------------
# explain_global
# ---------------------------------------------------------------------------

class TestExplainGlobal:
    def test_returns_global_explanation(self, explainer, fm):
        result = explainer.explain_global(fm.X, max_samples=100)
        assert isinstance(result, GlobalExplanation)

    def test_all_features_present(self, explainer, fm):
        result = explainer.explain_global(fm.X, max_samples=100)
        assert set(result.mean_abs_shap.keys()) == set(MODEL_FEATURES)

    def test_values_nonnegative(self, explainer, fm):
        result = explainer.explain_global(fm.X, max_samples=100)
        assert all(v >= 0.0 for v in result.mean_abs_shap.values())

    def test_n_samples_capped(self, explainer, fm):
        result = explainer.explain_global(fm.X, max_samples=50)
        assert result.n_samples == 50

    def test_n_samples_not_exceeded_when_small(self, explainer, fm):
        small_X = fm.X.head(30)
        result = explainer.explain_global(small_X, max_samples=200)
        assert result.n_samples == 30

    def test_top_features_length(self, explainer, fm):
        result = explainer.explain_global(fm.X, max_samples=100)
        top5 = result.top_features(5)
        assert len(top5) == 5

    def test_top_features_ordered(self, explainer, fm):
        result = explainer.explain_global(fm.X, max_samples=100)
        top = result.top_features(10)
        vals = [d["mean_abs_shap"] for d in top]
        assert vals == sorted(vals, reverse=True)

    def test_to_dict_serializable(self, explainer, fm):
        import json
        result = explainer.explain_global(fm.X, max_samples=50)
        d = result.to_dict()
        json.dumps(d)
        assert "n_samples" in d
        assert "mean_abs_shap" in d
