"""
Tests de ingesta y validación de esquema para WODPredict-AI.

Cubre:
  - Contratos Pydantic (schema.py): campos válidos, inválidos, anti-fuga
  - Loader (loader.py): separación feat/outcomes, columnas presentes,
    completitud del bundle, validación sobre muestra real del dataset
"""
from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from ingestion.schema import (
    AthleteRow,
    SessionFeatures,
    SessionOutcomes,
    ValidationReport,
    WorkoutRow,
)
from ingestion.loader import (
    SESSION_FEAT_COLS,
    SESSION_OUTCOME_COLS,
    load_dataset,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bundle():
    """Descarga el dataset una sola vez para todos los tests del módulo."""
    return load_dataset(validate=True, sample_validate=200)


# ---------------------------------------------------------------------------
# AthleteRow
# ---------------------------------------------------------------------------

class TestAthleteRow:
    def test_valid(self):
        row = AthleteRow(
            athlete_id="A0001",
            level="novato",
            sex="M",
            age=27.0,
            language="en",
            join_date="2024-10-01",
            attendance_pattern="regular",
            base_freq_per_week=3.9,
            source="synthetic",
        )
        assert row.athlete_id == "A0001"
        assert row.level == "novato"

    def test_invalid_level(self):
        with pytest.raises(ValidationError):
            AthleteRow(
                athlete_id="A0001",
                level="principiante",  # valor no permitido
                sex="M",
                age=27.0,
                language="en",
                join_date="2024-10-01",
                attendance_pattern="regular",
                base_freq_per_week=3.9,
                source="synthetic",
            )

    def test_invalid_age_out_of_range(self):
        with pytest.raises(ValidationError):
            AthleteRow(
                athlete_id="A0001",
                level="novato",
                sex="M",
                age=200.0,  # imposible
                language="en",
                join_date="2024-10-01",
                attendance_pattern="regular",
                base_freq_per_week=3.9,
                source="synthetic",
            )

    def test_date_as_string(self):
        row = AthleteRow(
            athlete_id="A0001",
            level="elite",
            sex="F",
            age=30.0,
            language="es",
            join_date="2024-10-01",
            attendance_pattern="erratic",
            base_freq_per_week=2.5,
            source="synthetic",
        )
        from datetime import date
        assert row.join_date == date(2024, 10, 1)


# ---------------------------------------------------------------------------
# SessionFeatures — regla anti-fuga
# ---------------------------------------------------------------------------

class TestSessionFeatures:
    BASE = dict(
        session_id="A0001_2024-10-15",
        athlete_id="A0001",
        date="2024-10-15",
        level="novato",
        days_since_prev_session=2.0,
        sessions_last_3d=1.0,
        sessions_last_7d=3.0,
        sessions_last_28d=10.0,
        load_last_3d=380.0,
        load_last_7d=950.0,
        mean_session_load_28d=350.0,
        acwr3_freq=None,
        acwr3_load=None,
        acwr_freq=1.1,
        acwr_load=1.05,
        acwr_valid=1,
        days_since_first_session=14,
        prev_rpe=6.0,
        prev_session_load=300.0,
        prev_has_note=0.0,
        prob_fatiga_prev_session=None,
        prob_fatiga_last_note=0.3,
        days_since_last_note=7.0,
        notes_last_28d=2,
    )

    def test_valid_feat_row(self):
        feat = SessionFeatures(**self.BASE)
        assert feat.athlete_id == "A0001"
        assert feat.acwr_valid == 1

    def test_invalid_rpe_prev_out_of_range(self):
        bad = {**self.BASE, "prev_rpe": 11.0}  # RPE máximo es 10
        with pytest.raises(ValidationError):
            SessionFeatures(**bad)

    def test_feat_does_not_contain_outcome_columns(self):
        """SessionFeatures NO debe tener campos de outcome (anti-fuga)."""
        feat_fields = set(SessionFeatures.model_fields.keys())
        outcome_only = {"rpe", "session_load", "session_minutes", "pr_count",
                        "overreaching_label", "overreaching_h1", "has_note"}
        leaked = feat_fields & outcome_only
        assert not leaked, f"Fuga detectada: {leaked}"

    def test_optional_nulls_accepted(self):
        sparse = {**self.BASE}
        sparse.update({
            "days_since_prev_session": None,
            "mean_session_load_28d": None,
            "acwr_freq": None,
            "acwr_load": None,
            "prev_rpe": None,
            "prev_session_load": None,
            "prev_has_note": None,
            "prob_fatiga_last_note": None,
            "days_since_last_note": None,
        })
        feat = SessionFeatures(**sparse)
        assert feat.acwr_freq is None


# ---------------------------------------------------------------------------
# SessionOutcomes
# ---------------------------------------------------------------------------

class TestSessionOutcomes:
    BASE = dict(
        session_id="A0001_2024-10-15",
        n_components=3,
        has_metcon=1,
        session_minutes=45.0,
        rpe=7,
        session_load=315.0,
        pr_count=0,
        has_note=1,
        prob_fatiga=0.25,
        overreaching_h1=0.0,
        overreaching_h3=None,
        overreaching_h7=None,
        overreaching_label=0.0,
        note_id=12,
        source="synthetic",
        captured_at=None,
    )

    def test_valid_outcome_row(self):
        out = SessionOutcomes(**self.BASE)
        assert out.rpe == 7
        assert out.overreaching_label == 0.0

    def test_rpe_out_of_range(self):
        bad = {**self.BASE, "rpe": 0}  # RPE mínimo es 1
        with pytest.raises(ValidationError):
            SessionOutcomes(**bad)

    def test_overreaching_nullable(self):
        row = {**self.BASE, "overreaching_label": None}
        out = SessionOutcomes(**row)
        assert out.overreaching_label is None


# ---------------------------------------------------------------------------
# ValidationReport
# ---------------------------------------------------------------------------

class TestValidationReport:
    def test_is_clean_true(self):
        r = ValidationReport(table="t", total_rows=100, valid_rows=100, invalid_rows=0)
        assert r.is_clean is True

    def test_is_clean_false(self):
        r = ValidationReport(table="t", total_rows=100, valid_rows=95, invalid_rows=5)
        assert r.is_clean is False


# ---------------------------------------------------------------------------
# Loader — tests de integración (usan el bundle descargado)
# ---------------------------------------------------------------------------

class TestLoader:
    def test_bundle_has_all_tables(self, bundle):
        assert bundle.athletes is not None
        assert bundle.sessions_feat is not None
        assert bundle.sessions_outcomes is not None
        assert bundle.workouts is not None

    def test_sessions_feat_columns(self, bundle):
        """Todas las columnas feat definidas deben estar presentes."""
        for col in SESSION_FEAT_COLS:
            assert col in bundle.sessions_feat.columns, f"Columna faltante en sessions_feat: {col}"

    def test_sessions_outcomes_columns(self, bundle):
        for col in SESSION_OUTCOME_COLS:
            assert col in bundle.sessions_outcomes.columns, f"Columna faltante en sessions_outcomes: {col}"

    def test_no_outcome_columns_in_feat(self, bundle):
        """La tabla feat NO debe contener columnas de outcome."""
        outcome_only = {"rpe", "session_load", "session_minutes", "pr_count",
                        "overreaching_label", "overreaching_h1", "has_note", "prob_fatiga"}
        leaked = set(bundle.sessions_feat.columns) & outcome_only
        assert not leaked, f"Fuga de outcomes en sessions_feat: {leaked}"

    def test_sessions_row_count(self, bundle):
        """El dataset tiene más de 80 000 sesiones."""
        assert len(bundle.sessions_feat) > 80_000

    def test_athletes_level_values(self, bundle):
        """Después de normalización, level debe estar en minúsculas."""
        valid_levels = {"novato", "intermedio", "elite"}
        assert set(bundle.athletes["level"].unique()).issubset(valid_levels)

    def test_athletes_sex_uppercase(self, bundle):
        """Después de normalización, sex debe ser M o F en mayúsculas."""
        assert set(bundle.athletes["sex"].unique()).issubset({"M", "F"})

    def test_validation_reports_present(self, bundle):
        assert len(bundle.reports) == 3  # athletes, sessions_feat, sessions_outcomes

    def test_validation_reports_clean(self, bundle):
        """Todas las tablas deben pasar validación sin errores tras normalización."""
        for report in bundle.reports:
            assert report.is_clean, (
                f"Errores en {report.table}: {report.errors[:3]}"
            )

    def test_athletes_no_nan_age(self, bundle):
        """age no debe tener NaN tras imputación con la mediana."""
        assert bundle.athletes["age"].isna().sum() == 0

    def test_feat_and_outcomes_share_session_id(self, bundle):
        """Las dos tablas de sesiones deben tener los mismos session_ids."""
        feat_ids = set(bundle.sessions_feat["session_id"])
        out_ids = set(bundle.sessions_outcomes["session_id"])
        assert feat_ids == out_ids
