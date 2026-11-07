---
name: wodpredict-ai
description: Predicts overreaching risk and explains SHAP-based training load factors for CrossFit and Hyrox athletes.
openapi: openapi.json
---

# WODPredict-AI Skills

## predict

Predicts the probability of overreaching (overtraining) for an athlete's training session.

Returns:
- `predicted_proba`: probability value between 0 and 1
- `risk_label`: `bajo` (low, <30%), `moderado` (moderate, 30–60%), or `alto` (high, >60%)

**Required inputs:**
- `session_id`: unique session identifier (e.g. `"A0001_2025-03-01"`)
- `sessions_last_3d`: number of sessions in the last 3 days
- `sessions_last_7d`: number of sessions in the last 7 days
- `sessions_last_28d`: number of sessions in the last 28 days
- `load_last_3d`: training load (RPE × minutes) in the last 3 days
- `load_last_7d`: training load in the last 7 days
- `acwr_valid`: 1 if sufficient history to compute ACWR, 0 otherwise
- `days_since_first_session`: days since the athlete's first recorded session
- `notes_last_28d`: number of subjective fatigue notes in the last 28 days
- `level`: athlete level — `novato`, `intermedio`, or `elite`

**Optional inputs** (imputed with training-set medians if omitted):
- `days_since_prev_session`, `mean_session_load_28d`
- `acwr3_freq`, `acwr3_load`, `acwr_freq`, `acwr_load`
- `prev_rpe`, `prev_session_load`, `prev_has_note`
- `prob_fatiga_prev_session`, `prob_fatiga_last_note`, `days_since_last_note`

---

## explain

Predicts overreaching risk AND explains the top-8 SHAP drivers behind the prediction.

Returns all fields from `predict`, plus:
- `base_value`: model expected value (SHAP baseline)
- `top_drivers`: list of the 8 most influential features, each with:
  - `feature`: feature name
  - `shap_value`: positive = pushes toward overreaching, negative = reduces risk
  - `feature_value`: actual value of the feature for this session

Same inputs as `predict`.
