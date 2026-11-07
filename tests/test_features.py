"""
Tests de feature engineering para WODPredict-AI — Fase 2.

Cubre:
  - Funciones internas: _encode_level, _fatigue_signal, imputación
  - build_features: columnas presentes, tipos, sin NaN, dimensiones
  - Integración: build_features sobre el bundle real descargado
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from features.builder import (
    BASE_FEATURES,
    DERIVED_FEATURES,
    MODEL_FEATURES,
    FeatureMatrix,
    _encode_level,
    _fatigue_signal,
    _apply_imputation,
    _compute_imputation_medians,
    build_features,
)
from ingestion.loader import load_dataset


# ---------------------------------------------------------------------------
# Fixture: bundle real (descargado una vez por módulo)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bundle():
    return load_dataset(validate=False)


# ---------------------------------------------------------------------------
# _encode_level
# ---------------------------------------------------------------------------

class TestEncodeLevel:
    def test_known_values(self):
        s = pd.Series(["novato", "intermedio", "elite"])
        result = _encode_level(s)
        assert list(result) == [0, 1, 2]

    def test_unknown_mapped_to_zero(self):
        s = pd.Series(["novato", "principiante"])
        result = _encode_level(s)
        assert result.iloc[1] == 0

    def test_returns_int_dtype(self):
        s = pd.Series(["elite", "novato"])
        assert _encode_level(s).dtype == int


# ---------------------------------------------------------------------------
# _fatigue_signal
# ---------------------------------------------------------------------------

class TestFatigueSignal:
    def _df(self, a, b):
        return pd.DataFrame({
            "prob_fatiga_prev_session": [a],
            "prob_fatiga_last_note": [b],
        })

    def test_both_present(self):
        result = _fatigue_signal(self._df(0.2, 0.4))
        assert abs(result.iloc[0] - 0.3) < 1e-9

    def test_only_first(self):
        result = _fatigue_signal(self._df(0.5, np.nan))
        assert abs(result.iloc[0] - 0.5) < 1e-9

    def test_only_second(self):
        result = _fatigue_signal(self._df(np.nan, 0.7))
        assert abs(result.iloc[0] - 0.7) < 1e-9

    def test_both_nan(self):
        result = _fatigue_signal(self._df(np.nan, np.nan))
        assert math.isnan(result.iloc[0])


# ---------------------------------------------------------------------------
# Imputación
# ---------------------------------------------------------------------------

class TestImputation:
    def test_medians_computed_only_for_null_cols(self):
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [1.0, 2.0, 3.0]})
        medians = _compute_imputation_medians(df, ["a", "b"])
        assert "a" in medians
        assert "b" not in medians  # sin nulls, no debe aparecer

    def test_apply_fills_nan(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        filled = _apply_imputation(df, {"a": 99.0})
        assert filled["a"].isna().sum() == 0
        assert filled["a"].iloc[1] == 99.0

    def test_apply_does_not_modify_original(self):
        df = pd.DataFrame({"a": [np.nan]})
        _apply_imputation(df, {"a": 5.0})
        assert math.isnan(df["a"].iloc[0])  # original intacto


# ---------------------------------------------------------------------------
# build_features — integración sobre bundle real
# ---------------------------------------------------------------------------

class TestBuildFeatures:
    def test_returns_feature_matrix(self, bundle):
        fm = build_features(bundle)
        assert isinstance(fm, FeatureMatrix)

    def test_all_model_features_present(self, bundle):
        fm = build_features(bundle)
        for col in MODEL_FEATURES:
            assert col in fm.X.columns, f"Columna faltante: {col}"

    def test_no_nan_in_X(self, bundle):
        fm = build_features(bundle)
        null_counts = fm.X.isna().sum()
        cols_with_nan = null_counts[null_counts > 0]
        assert cols_with_nan.empty, f"NaN en X tras imputación: {cols_with_nan.to_dict()}"

    def test_y_is_binary(self, bundle):
        fm = build_features(bundle)
        assert set(fm.y.unique()).issubset({0, 1})

    def test_X_y_same_length(self, bundle):
        fm = build_features(bundle)
        assert len(fm.X) == len(fm.y)

    def test_row_count(self, bundle):
        """Debe haber más de 80 000 filas etiquetadas."""
        fm = build_features(bundle)
        assert len(fm.X) > 80_000

    def test_feature_count(self, bundle):
        fm = build_features(bundle)
        assert len(fm.X.columns) == len(MODEL_FEATURES)

    def test_level_enc_values(self, bundle):
        fm = build_features(bundle)
        assert set(fm.X["level_enc"].unique()).issubset({0, 1, 2})

    def test_acwr_risk_flags_are_binary(self, bundle):
        fm = build_features(bundle)
        assert set(fm.X["acwr_load_risk"].unique()).issubset({0, 1})
        assert set(fm.X["acwr_freq_risk"].unique()).issubset({0, 1})

    def test_prev_rpe_norm_nonnegative(self, bundle):
        """prev_rpe_norm = prev_rpe / 10; debe ser ≥ 0 (puede superar 1.0 si hay datos atípicos)."""
        fm = build_features(bundle)
        assert fm.X["prev_rpe_norm"].min() >= 0.0

    def test_imputation_medians_stored(self, bundle):
        fm = build_features(bundle)
        assert isinstance(fm.imputation_medians, dict)
        assert len(fm.imputation_medians) > 0

    def test_precomputed_medians_applied(self, bundle):
        """Pasar medianas externas no debe generar NaN."""
        fm_ref = build_features(bundle)
        fm2 = build_features(bundle, imputation_medians=fm_ref.imputation_medians)
        assert fm2.X.isna().sum().sum() == 0

    def test_feature_names_match_columns(self, bundle):
        fm = build_features(bundle)
        assert fm.feature_names == list(fm.X.columns)
