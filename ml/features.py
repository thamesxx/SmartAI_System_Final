"""Feature engineering — used by BOTH training and live serving.

The single source of truth for feature computation.  Any change here changes
PIPELINE_VERSION, which invalidates old Parquet shards and models.

Public API
----------
build_features_for_machine(machine_df, cfg) -> pd.DataFrame
    Given ALL readings for one machine (sorted by ts), returns a feature
    DataFrame at FEATURE_STRIDE_READINGS cadence.

build_feature_row(window_df, cfg) -> dict
    Given the trailing window DataFrame for ONE machine at a single point in
    time, returns one feature dict (used live at serving time).

FEATURE_COLUMNS : list[str]
    Frozen column order that must match what the model was trained on.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ml import config as _cfg_type


# ── Feature column list (auto-built from config, frozen at import time) ─────

def _build_column_list(cfg) -> list[str]:
    cols: list[str] = []
    for sensor in cfg.SENSOR_COLS:
        for w in cfg.WINDOW_SIZES:
            cols += [
                f"{sensor}_w{w}_mean",
                f"{sensor}_w{w}_std",
                f"{sensor}_w{w}_slope",   # (last - first) / W
            ]
    cols += [
        "current_per_speed",
        "power_per_speed",
        "vib_per_current",
        "sf_per_speed",
        "wat_per_speed",
        "sf_tot_rate",
        "wat_tot_rate",
        "em_energy_rate",
        "reject_pct",
        "is_running",
        "machine_time_h",
    ]
    return cols


# ── Core computation ─────────────────────────────────────────────────────────

def build_features_for_machine(machine_df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Vectorised feature computation for a full machine history.

    Parameters
    ----------
    machine_df : sorted by ts ascending, all readings for ONE machine.
    cfg        : ml.config module.

    Returns
    -------
    DataFrame with FEATURE_COLUMNS + metadata columns
    [session_id, machine_name, ts, seq].  Rows start from MAX_WINDOW index
    and are spaced FEATURE_STRIDE_READINGS apart.
    """
    df = machine_df.sort_values("ts").reset_index(drop=True)
    n  = len(df)

    feats: dict[str, np.ndarray] = {}

    for sensor in cfg.SENSOR_COLS:
        col = df[sensor].astype(float).ffill().fillna(0.0)

        for w in cfg.WINDOW_SIZES:
            rolling = col.rolling(window=w, min_periods=max(2, w // 4))
            feats[f"{sensor}_w{w}_mean"] = rolling.mean().values
            feats[f"{sensor}_w{w}_std"]  = rolling.std().fillna(0.0).values
            # Slope proxy: (current − value_W_steps_ago) / W   O(N)
            shifted = col.shift(w - 1)
            slope_raw = (col - shifted) / max(w - 1, 1)
            feats[f"{sensor}_w{w}_slope"] = slope_raw.fillna(0.0).values

    # Cross-sensor ratios (instantaneous)
    speed = df["speed"].astype(float).fillna(0.0).clip(lower=0.5)
    mc    = df["motor_current"].astype(float).fillna(0.0).clip(lower=0.01)
    vib   = df["vibration_rms"].astype(float).fillna(0.0)

    feats["current_per_speed"] = (mc / speed).fillna(0.0).values
    feats["power_per_speed"]   = (df["em_power"].astype(float).fillna(0.0) / speed).fillna(0.0).values
    feats["vib_per_current"]   = (vib / mc).fillna(0.0).values
    feats["sf_per_speed"]      = (df["sf_flow"].astype(float).fillna(0.0) / speed).fillna(0.0).values
    feats["wat_per_speed"]     = (df["wat_flow"].astype(float).fillna(0.0) / speed).fillna(0.0).values

    # Totalizer rates over short window (Δ per reading)
    w0 = cfg.WINDOW_SIZES[0]
    for tot_col, rate_key in [
        ("sf_tot",    "sf_tot_rate"),
        ("wat_tot",   "wat_tot_rate"),
        ("em_energy", "em_energy_rate"),
    ]:
        tot  = df[tot_col].astype(float).ffill().fillna(0.0)
        rate = (tot - tot.shift(w0 - 1)) / max(w0 - 1, 1)
        feats[rate_key] = rate.fillna(0.0).values

    # Quality
    good   = df["good_count"].astype(float).fillna(0.0)
    reject = df["reject_count"].astype(float).fillna(0.0)
    total  = (good + reject).clip(lower=1.0)
    feats["reject_pct"] = (reject / total).values

    # State
    feats["is_running"] = (df["state"] == "running").astype(float).values

    # Wear proxy
    feats["machine_time_h"] = (df["machine_time_s"].astype(float).fillna(0.0) / 3600.0).values

    feat_df = pd.DataFrame(feats, index=df.index)

    # Metadata columns (not features — used for joining labels, grouping, serving)
    feat_df["session_id"]   = df["session_id"].values
    feat_df["machine_name"] = df["machine_name"].values
    feat_df["ts"]           = df["ts"].values
    feat_df["seq"]          = df["seq"].values

    # Apply minimum-window guard + stride
    valid_start = cfg.MAX_WINDOW - 1
    stride_idx  = list(range(valid_start, n, cfg.FEATURE_STRIDE_READINGS))
    return feat_df.iloc[stride_idx].reset_index(drop=True)


def build_feature_row(window_df: pd.DataFrame, cfg) -> dict:
    """Compute ONE feature dict from the trailing window at serving time.

    window_df must be sorted ascending by ts.  Length >= MAX_WINDOW preferred;
    shorter windows are accepted (features will degrade gracefully).
    """
    result = build_features_for_machine(window_df, cfg)
    if result.empty:
        return {}
    last = result.iloc[-1]
    feature_cols = _build_column_list(cfg)
    return {c: float(last[c]) for c in feature_cols if c in last.index}


# ── Materialise FEATURE_COLUMNS ──────────────────────────────────────────────
# Imported from here in train.py, dataset.py, and prediction_service.py.
# This is lazy — avoids circular import at module level.

def get_feature_columns(cfg) -> list[str]:
    return _build_column_list(cfg)
