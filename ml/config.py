"""Central configuration for the ml/ package.

All tunable constants live here.  Features.py hashes this dict to produce
pipeline_version, so changing any value here automatically invalidates
existing sealed Parquet shards (correct behaviour).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

# ─── Paths ─────────────────────────────────────────────────────────────────
ML_ROOT           = Path(__file__).parent
ARTIFACTS_DIR     = ML_ROOT / "artifacts"
FEATURE_STORE_DIR = ML_ROOT / "feature_store"
PLOTS_DIR         = ARTIFACTS_DIR / "plots"

for _d in [ARTIFACTS_DIR, FEATURE_STORE_DIR, PLOTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─── Prediction horizon ─────────────────────────────────────────────────────
H_HOURS   = 24.0            # sim-hours ahead to predict
H_SECONDS = H_HOURS * 3600.0

# ─── Feature engineering ────────────────────────────────────────────────────
# Sensor columns read from machine_readings
SENSOR_COLS: list[str] = [
    "vibration_rms",  # bearing fingerprint ↑
    "bearing_temp",   # bearing fingerprint ↑
    "winding_temp",   # heater fingerprint ↑
    "sf_flow",        # steam_valve fingerprint ↓
    "wat_flow",       # water_pump fingerprint ↓
    "motor_current",  # general load stress
    "em_power",       # electrical stress
    "air_pressure",   # air supply
    "speed",          # production rate
]

# Rolling window sizes (in number of readings)
# At dt=30s: W=20 → 10 min, W=60 → 30 min, W=180 → 90 min
WINDOW_SIZES: list[int] = [20, 60, 180]

MAX_WINDOW              = max(WINDOW_SIZES)   # readings needed before first feature row
FEATURE_STRIDE_READINGS = 2                   # 1 feature row every N readings (≈1/min at dt=30s)

NOMINAL_SPEED = 120.0   # m/min reference for OEE Performance pillar

# ─── Model / classes ────────────────────────────────────────────────────────
CLASS_NAMES: list[str] = ["none", "bearing", "steam_valve", "heater", "water_pump"]
N_CLASSES               = len(CLASS_NAMES)
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASS_NAMES)}

# ─── Label generation ───────────────────────────────────────────────────────
# Component health below this → "degraded / will fail soon" label window.
# Set from empirical min health in the backfilled dataset (bearing=62.5, wp=68).
HEALTH_FAIL_THRESHOLD = 75.0

# ─── Train / val / test split ───────────────────────────────────────────────
TRAIN_FRAC = 0.75     # first 75 % of sim time → train
VAL_FRAC   = 0.875    # next 12.5 % → val   (ends at 87.5 %)
GAP_HOURS  = 24.0     # sim-hour gap between splits (prevents horizon bleed)

# ─── XGBoost defaults ───────────────────────────────────────────────────────
XGB_DEFAULTS: dict = dict(
    objective          = "multi:softprob",
    num_class          = N_CLASSES,
    eval_metric        = "mlogloss",
    n_estimators       = 500,
    max_depth          = 6,
    learning_rate      = 0.05,
    subsample          = 0.8,
    colsample_bytree   = 0.8,
    min_child_weight   = 5,
    gamma              = 0.1,
    reg_lambda         = 1.0,
    tree_method        = "hist",
    n_jobs             = -1,
    verbosity          = 0,
)
EARLY_STOPPING_ROUNDS = 50

# ─── Optuna tuning ──────────────────────────────────────────────────────────
OPTUNA_TRIALS  = 30
OPTUNA_TIMEOUT = 600  # seconds

# ─── Pipeline version (auto-computed, do not edit) ──────────────────────────
_VERSION_SEED = {
    "SENSOR_COLS":            SENSOR_COLS,
    "WINDOW_SIZES":           WINDOW_SIZES,
    "H_HOURS":                H_HOURS,
    "CLASS_NAMES":            CLASS_NAMES,
    "NOMINAL_SPEED":          NOMINAL_SPEED,
    "HEALTH_FAIL_THRESHOLD":  HEALTH_FAIL_THRESHOLD,
}
PIPELINE_VERSION: str = (
    "v1."
    + hashlib.sha256(json.dumps(_VERSION_SEED, sort_keys=True).encode()).hexdigest()[:10]
)
