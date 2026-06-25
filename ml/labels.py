"""Label assignment — two strategies depending on data availability.

Strategy A (preferred): use machine_runs failure_ts events (two-pointer O(N+M)).
Strategy B (fallback):  parse truth_json health column from machine_readings.
    A reading gets label=component when that component's health drops below
    HEALTH_FAIL_THRESHOLD within the next H_SECONDS of sim time.  This
    is equivalent to "the component will fail soon" and is the ground truth
    already embedded in truth_json by the generator.

build_labels() auto-selects: if runs_df is empty it falls through to strategy B.
build_labels_from_truth_json() can be called directly for testing.
"""
from __future__ import annotations

import json as _json

import numpy as np
import pandas as pd

# Import from config so the threshold is part of pipeline_version hash.
# Fallback to 75.0 to avoid circular import if labels is imported before config.
try:
    from ml.config import HEALTH_FAIL_THRESHOLD
except ImportError:
    HEALTH_FAIL_THRESHOLD = 75.0

# Components tracked in truth_json (must match ml.config.CLASS_NAMES minus "none")
_COMPONENTS = ["bearing", "steam_valve", "heater", "water_pump"]


# ── Strategy A: machine_runs two-pointer ──────────────────────────────────────

def _labels_from_runs(
    readings_df: pd.DataFrame,
    runs_df: pd.DataFrame,
    H_seconds: float,
) -> pd.Series:
    H_ns = int(H_seconds * 1e9)
    all_parts: list[pd.Series] = []

    for machine in readings_df["machine_name"].unique():
        m_reads = readings_df[readings_df["machine_name"] == machine].sort_values("ts")
        m_fails = (
            runs_df[
                (runs_df["machine_name"] == machine)
                & runs_df["failure_ts"].notna()
            ]
            .sort_values("failure_ts")
        )

        idx_arr = m_reads.index.to_numpy()
        read_ts = m_reads["ts"].to_numpy(dtype="datetime64[ns]")

        if m_fails.empty:
            all_parts.append(pd.Series("none", index=idx_arr, dtype=object))
            continue

        fail_ts   = m_fails["failure_ts"].to_numpy(dtype="datetime64[ns]")
        fail_comp = m_fails["component"].to_numpy(dtype=object)
        labels = np.full(len(read_ts), "none", dtype=object)
        j = 0

        for i, rts in enumerate(read_ts):
            while j < len(fail_ts) and fail_ts[j] < rts:
                j += 1
            if j < len(fail_ts):
                diff_ns = int((fail_ts[j] - rts).astype("int64"))
                if diff_ns <= H_ns:
                    labels[i] = fail_comp[j]

        all_parts.append(pd.Series(labels, index=idx_arr, dtype=object))

    if not all_parts:
        return pd.Series(dtype=object, index=readings_df.index)
    return pd.concat(all_parts).reindex(readings_df.index)


# ── Strategy B: truth_json health-based labels ────────────────────────────────

def build_labels_from_truth_json(
    readings_df: pd.DataFrame,
    H_seconds: float,
    health_threshold: float = HEALTH_FAIL_THRESHOLD,
) -> pd.Series:
    """Derive labels from the truth_json column embedded in each reading row.

    For each machine's sorted readings, find the first point where a component
    health drops below `health_threshold`.  All readings within H_seconds before
    that point get labeled as that component.  After a "failure" point the
    algorithm looks for the next degradation event (repair resets health).

    Parameters
    ----------
    readings_df : must contain columns [machine_name, ts, truth_json].
    H_seconds   : prediction horizon in sim seconds.
    health_threshold : component health (0–100) below which = "will fail soon".
    """
    if "truth_json" not in readings_df.columns:
        raise ValueError("readings_df must contain 'truth_json' column for strategy B")

    H_td = pd.Timedelta(seconds=H_seconds)
    all_parts: list[pd.Series] = []

    for machine, mdf in readings_df.groupby("machine_name"):
        mdf = mdf.sort_values("ts").reset_index(drop=True)
        orig_index = readings_df[readings_df["machine_name"] == machine].sort_values("ts").index

        ts_arr     = mdf["ts"].values
        json_col   = mdf["truth_json"].values
        n          = len(mdf)
        labels     = np.full(n, "none", dtype=object)

        # Parse health for all rows (vectorized where possible)
        health: dict[str, np.ndarray] = {c: np.full(n, 100.0) for c in _COMPONENTS}

        for i, raw in enumerate(json_col):
            if raw is None:
                continue
            try:
                obj = _json.loads(raw) if isinstance(raw, str) else raw
                h   = obj.get("health", {})
                for comp in _COMPONENTS:
                    if comp in h:
                        health[comp][i] = float(h[comp])
            except Exception:
                pass

        # For each component find degradation events and back-label H_seconds before them
        for comp in _COMPONENTS:
            h_arr = health[comp]
            i = 0
            while i < n:
                # Find first point where health drops below threshold
                fail_idx = None
                for k in range(i, n):
                    if h_arr[k] < health_threshold:
                        fail_idx = k
                        break

                if fail_idx is None:
                    break  # no more failures for this component

                fail_ts = pd.Timestamp(ts_arr[fail_idx])

                # Back-label all readings from (fail_ts - H) to fail_ts
                horizon_start = fail_ts - H_td
                for k in range(fail_idx, -1, -1):
                    rts = pd.Timestamp(ts_arr[k])
                    if rts < horizon_start:
                        break
                    # Only label if not already labeled by a closer failure
                    if labels[k] == "none":
                        labels[k] = comp

                # Advance past this failure event (find repair = health goes back up)
                i = fail_idx + 1
                while i < n and h_arr[i] < health_threshold:
                    i += 1

        all_parts.append(pd.Series(labels, index=orig_index, dtype=object))

    if not all_parts:
        return pd.Series(dtype=object, index=readings_df.index)
    return pd.concat(all_parts).reindex(readings_df.index)


# ── Public API ────────────────────────────────────────────────────────────────

def build_labels(
    readings_df: pd.DataFrame,
    runs_df: pd.DataFrame,
    H_seconds: float,
) -> pd.Series:
    """Assign labels. Uses machine_runs if populated, else falls back to truth_json.

    Parameters
    ----------
    readings_df : columns [machine_name, ts] required; truth_json needed for fallback.
    runs_df     : machine_runs failure events (may be empty).
    H_seconds   : prediction horizon in simulation seconds.
    """
    if runs_df is not None and len(runs_df) > 0:
        return _labels_from_runs(readings_df, runs_df, H_seconds)

    if "truth_json" not in readings_df.columns:
        # No fallback available — return all "none"
        return pd.Series("none", index=readings_df.index, dtype=object)

    return build_labels_from_truth_json(readings_df, H_seconds)
