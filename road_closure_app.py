"""
Road Closure Prediction API
-----------------------------
Predicts whether a traffic incident will require a road closure.

User supplies 7 core fields. All 35+ model features are derived internally.
"""

from __future__ import annotations

import pickle
from typing import Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from huggingface_hub import hf_hub_download



# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (must mirror training code exactly)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = hf_hub_download(
    repo_id="SupratimKukri/road-closure-model",
    filename="road_closure_model.pkl"
)
PEAK_HOURS  = {7, 8, 9, 17, 18, 19, 20, 21}
NIGHT_HOURS = set(range(22, 24)) | set(range(0, 5))

CAT_COLS = ['event_cause', 'priority', 'veh_type', 'corridor', 'zone', 'event_type']

FEATURES = [
    'event_cause', 'priority', 'veh_type', 'corridor', 'zone', 'event_type',
    'hour', 'day_of_week',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
    'is_peak_hour', 'is_weekend', 'is_night',
    'geo_cluster',
    'corridor_volume',
    'cause_closure_rate', 'corridor_closure_rate',
    'cause_x_peak', 'priority_x_unplanned', 'cause_x_corridor_vol', 'priority_num',
    'is_heavy_vehicle', 'has_cargo', 'duration_hours', 'is_old_truck', 'at_junction',
    'has_police', 'is_accident', 'has_direction', 'is_both_directions',
    'is_serious_breakdown', 'is_high_risk_zone', 'time_to_resolve_hours',
    'resolved_at_diff_location', 'has_comment', 'has_metadata',
]


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST SCHEMA  — only what the user actually knows
# ─────────────────────────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    # ── Required (7 fields) ──────────────────────────────────────────────────
    start_datetime: str = Field(
        ...,
        example="2026-06-19 08:30:00+0530",
        description="When the incident started.  Format: YYYY-MM-DD HH:MM:SS±HHMM"
    )
    latitude:    float = Field(..., ge=-90,  le=90,  example=12.97)
    longitude:   float = Field(..., ge=-180, le=180, example=77.59)
    event_cause: str   = Field(..., example="vehicle_breakdown",
                               description="e.g. vehicle_breakdown, tree_fall, accident, construction, vip_movement, water_logging, others")
    priority:    str   = Field(..., example="High", description="High or Low")
    zone:        str   = Field(..., example="East Zone 1")
    corridor:    str   = Field(..., example="ORR",
                               description="Road corridor name, or 'Non-corridor'")

    # ── Optional — improves accuracy when provided ────────────────────────────
    event_type:       Optional[str]   = Field(None,  example="unplanned",
                                              description="planned or unplanned")
    veh_type:         Optional[str]   = Field(None,  example="Truck")
    direction:        Optional[str]   = Field(None,  example="North",
                                              description="Incident direction. Use 'both' for both directions.")
    junction:         Optional[str]   = Field(None,  example="Silk Board")
    reason_breakdown: Optional[str]   = Field(None,  example="Engine Failure")
    cargo_material:   Optional[str]   = Field(None,  example="Container")
    age_of_truck:     Optional[float] = Field(None,  example=12,
                                              description="Age of truck in years (if applicable)")
    comment:          Optional[str]   = Field(None,  example="Tow vehicle requested")


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  (mirrors training notebook exactly)
# ─────────────────────────────────────────────────────────────────────────────
def _parse_dt(dt_str: str) -> pd.Timestamp:
    """Parse datetime string tolerating multiple formats."""
    for fmt in (
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return pd.to_datetime(dt_str, format=fmt, utc=True)
        except (ValueError, TypeError):
            continue
    ts = pd.to_datetime(dt_str, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Cannot parse datetime '{dt_str}'")
    return ts


def build_row(req: PredictionRequest) -> pd.DataFrame:
    """
    Convert user-supplied fields into the raw DataFrame row
    that the training-time preprocess() function expects.
    """
    start = _parse_dt(req.start_datetime)

    # Derive direction flags
    direction_str = (req.direction or "").lower()

    row = {
        # ── Timestamps ──────────────────────────────────────────────────────
        "start_datetime":         start,
        "end_datetime":           pd.NaT,
        "resolved_datetime":      pd.NaT,
        "resolved_at_address":    None,
        "address":                None,

        # ── Categorical fields ───────────────────────────────────────────────
        "event_cause":            req.event_cause,
        "priority":               req.priority,
        "veh_type":               req.veh_type        or "unknown",
        "corridor":               req.corridor,
        "zone":                   req.zone,
        "event_type":             req.event_type      or "unplanned",

        # ── Coordinates ─────────────────────────────────────────────────────
        "latitude":               req.latitude,
        "longitude":              req.longitude,

        # ── Optional domain fields ───────────────────────────────────────────
        "cargo_material":         req.cargo_material,
        "age_of_truck":           req.age_of_truck    or 0,
        "junction":               req.junction,
        "assigned_to_police_id":  None,
        "citizen_accident_id":    None,
        "direction":              req.direction,
        "reason_breakdown":       req.reason_breakdown,
        "comment":                req.comment,
        "meta_data":              None,

        # Dummy target column (required by preprocess but not used at inference)
        "road_closure":           0,
    }

    return pd.DataFrame([row])


def run_preprocess(raw_df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """
    Re-implement the training notebook's preprocess() in inference mode
    using the fitted artifacts stored in the model bundle.
    """
    df = raw_df.copy()

    # ── Ensure all expected columns exist (guard against missing optionals) ──
    for col in ['end_datetime', 'resolved_datetime', 'resolved_at_address',
                'address', 'cargo_material', 'age_of_truck', 'junction',
                'assigned_to_police_id', 'citizen_accident_id', 'direction',
                'reason_breakdown', 'comment', 'meta_data']:
        if col not in df.columns:
            df[col] = None

    # ── clean_categoricals ───────────────────────────────────────────────────
    df['veh_type']    = df['veh_type'].replace('', 'unknown').fillna('unknown')
    df['zone']        = df['zone'].replace('NULL', 'unknown').fillna('unknown')
    df['corridor']    = df['corridor'].fillna('Non-corridor')
    df['event_cause'] = df['event_cause'].fillna('others')
    df['priority']    = df['priority'].replace('NULL', 'Low').fillna('Low')
    df['event_type']  = df['event_type'].fillna('unplanned')

    # ── add_time_features ────────────────────────────────────────────────────
    df['start_datetime'] = pd.to_datetime(df['start_datetime'], utc=True, errors='coerce')
    df['hour']        = df['start_datetime'].dt.hour.fillna(0).astype(int)
    df['day_of_week'] = df['start_datetime'].dt.dayofweek.fillna(0).astype(int)
    df['month']       = df['start_datetime'].dt.month.fillna(1).astype(int)

    df['hour_sin']  = np.sin(2 * np.pi * df['hour']        / 24)
    df['hour_cos']  = np.cos(2 * np.pi * df['hour']        / 24)
    df['dow_sin']   = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos']   = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month']       / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']       / 12)

    df['is_peak_hour'] = df['hour'].isin(PEAK_HOURS).astype(int)
    df['is_weekend']   = (df['day_of_week'] >= 5).astype(int)
    df['is_night']     = df['hour'].isin(NIGHT_HOURS).astype(int)

    # ── add_geo_features  (predict mode) ────────────────────────────────────
    coords = df[['latitude', 'longitude']].fillna(0)
    df['geo_cluster'] = bundle['kmeans'].predict(coords)

    # ── add_statistical_features  (predict mode) ────────────────────────────
    stats = bundle['stats']
    df['cause_closure_rate']    = df['event_cause'].map(stats['cause_rate']).fillna(0)
    df['corridor_closure_rate'] = df['corridor'].map(stats['corridor_rate']).fillna(0)
    df['corridor_volume']       = df['corridor'].map(stats['corridor_vol']).fillna(0)

    # ── add_interaction_features ─────────────────────────────────────────────
    df['priority_num']         = df['priority'].map({'High': 1, 'Low': 0}).fillna(0)
    df['cause_x_peak']         = df['cause_closure_rate'] * df['is_peak_hour']
    df['priority_x_unplanned'] = df['priority_num'] * (df['event_type'] == 'unplanned').astype(int)
    df['cause_x_corridor_vol'] = df['cause_closure_rate'] * np.log1p(df['corridor_volume'])

    # ── add_domain_features ──────────────────────────────────────────────────
    df['is_heavy_vehicle'] = (
        df['veh_type'].astype(str).str.lower()
        .str.contains('truck|lorry|bus|tanker|trailer|heavy', na=False)
        .astype(int)
    )
    df['has_cargo'] = df['cargo_material'].notna().astype(int)

    # Duration: end_datetime may be NaT — fillna(0) handles it safely
    start_ts = pd.to_datetime(df['start_datetime'], utc=True, errors='coerce')
    end_ts   = pd.to_datetime(df['end_datetime'],   utc=True, errors='coerce')
    df['duration_hours'] = (
        (end_ts - start_ts).dt.total_seconds() / 3600
    ).clip(0, 48).fillna(0)

    # age_of_truck is already a column — use it directly, not df.get()
    df['age_of_truck'] = pd.to_numeric(df['age_of_truck'], errors='coerce').fillna(0)
    df['is_old_truck'] = (df['age_of_truck'] > 10).astype(int)

    df['at_junction'] = df['junction'].notna().astype(int)
    df['has_police']  = df['assigned_to_police_id'].notna().astype(int)
    df['is_accident'] = df['citizen_accident_id'].notna().astype(int)

    # direction: fill None/NaN before string ops to avoid 'nan' string issues
    direction = df['direction'].fillna('').astype(str).str.lower()
    df['has_direction']      = (direction != '').astype(int)
    df['is_both_directions'] = direction.str.contains('both|all|contra', na=False).astype(int)

    # reason_breakdown: use column directly (always exists after guard above)
    df['is_serious_breakdown'] = (
        df['reason_breakdown'].fillna('').astype(str).str.lower()
        .str.contains('fire|accident|collision|engine|oil|flood', na=False)
        .astype(int)
    )

    df['is_high_risk_zone'] = (
        df['zone'].astype(str).str.lower()
        .str.contains('highway|tunnel|bridge|school|hospital', na=False)
        .astype(int)
    )

    # resolved_datetime: use column directly
    resolved_ts = pd.to_datetime(df['resolved_datetime'], utc=True, errors='coerce')
    df['time_to_resolve_hours'] = (
        (resolved_ts - start_ts).dt.total_seconds() / 3600
    ).clip(0, 72).fillna(0)

    df['resolved_at_diff_location'] = 0   # not available at inference time
    df['has_comment']  = df['comment'].notna().astype(int)
    df['has_metadata'] = df['meta_data'].notna().astype(int)

    # ── encode + impute ──────────────────────────────────────────────────────
    X = df[FEATURES].copy()
    encoders = bundle['encoders']
    for col in CAT_COLS:
        le = encoders[col]
        # use a fixed default label (-1) for unseen categories
        X[col] = X[col].astype(str).apply(
            lambda v, _le=le: int(_le.transform([v])[0]) if v in _le.classes_ else -1
        )

    X = pd.DataFrame(
        bundle['imputer'].transform(X),
        columns=FEATURES
    )
    return X


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Road Closure Prediction API",
    description=(
        "Predicts whether a traffic incident will require a road closure.\n\n"
        "**Minimum required fields:** `start_datetime`, `latitude`, `longitude`, "
        "`event_cause`, `priority`, `zone`, `corridor`.\n\n"
        "All other fields are optional but improve accuracy."
    ),
    version="1.0.0",
)

# Load model bundle once at startup
try:
    with open(MODEL_PATH, "rb") as f:
        BUNDLE = pickle.load(f)
    THRESHOLD = BUNDLE.get("threshold", 0.5)
    print(f"Model loaded | threshold = {THRESHOLD:.3f}")
except FileNotFoundError:
    BUNDLE = None
    THRESHOLD = 0.5
    print(f"Warning: {MODEL_PATH} not found — /predict will return 503")


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "model_loaded": BUNDLE is not None}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "model_loaded": BUNDLE is not None}


@app.post("/predict", tags=["Prediction"])
def predict(req: PredictionRequest):
    """
    Predict whether a traffic incident requires road closure.

    Returns:
    - **road_closure_required** – True / False
    - **confidence** – model probability (0–1)
    - **risk_level** – Low / Medium / High
    """
    if BUNDLE is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model bundle '{MODEL_PATH}' not found. "
                "Train the model first and place the .pkl in the working directory."
            ),
        )

    try:
        raw_df = build_row(req)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Input error: {exc}")

    try:
        X    = run_preprocess(raw_df, BUNDLE)
        prob = float(BUNDLE["model"].predict_proba(X)[0][1])
    except Exception as exc:
        import traceback
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {exc}\n{traceback.format_exc()}"
        )

    closure  = bool(prob >= THRESHOLD)        # cast numpy.bool_ → Python bool
    risk_lvl = "Low" if prob < 0.3 else ("Medium" if prob < 0.6 else "High")

    return JSONResponse({
        "prediction": {
            "road_closure_required": closure,
            "confidence":            round(float(prob), 3),
            "risk_level":            risk_lvl,
        },
        "input_summary": {
            "start_datetime": req.start_datetime,
            "event_cause":    req.event_cause,
            "priority":       req.priority,
            "zone":           req.zone,
            "corridor":       req.corridor,
            "location":       {"lat": float(req.latitude), "lon": float(req.longitude)},
        },
        "model_info": {
            "threshold_used": round(float(THRESHOLD), 3),
        },
    })


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(requests: list[PredictionRequest]):
    """
    Predict road closure for up to 100 incidents in a single call.
    Returns results in the same order as the input list.
    """
    if BUNDLE is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if len(requests) > 100:
        raise HTTPException(status_code=400, detail="Max 100 records per batch request.")

    results = []
    for i, req in enumerate(requests):
        try:
            raw_df = build_row(req)
            X      = run_preprocess(raw_df, BUNDLE)
            prob   = float(BUNDLE["model"].predict_proba(X)[0][1])
            results.append({
                "index":                  i,
                "road_closure_required":  bool(prob >= THRESHOLD),  # cast numpy.bool_ → Python bool
                "confidence":             round(prob, 3),
                "risk_level":             "Low" if prob < 0.3 else ("Medium" if prob < 0.6 else "High"),
            })
        except Exception as exc:
            results.append({"index": i, "error": str(exc)})

    return JSONResponse({"predictions": results, "count": len(results)})


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("road_closure_app:app", host="0.0.0.0", port=8002, reload=True)