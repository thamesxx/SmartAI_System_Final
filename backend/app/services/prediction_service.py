"""Live prediction service — loads the trained model artifact and scores each machine.

The feature engineering is IDENTICAL to training (imported from ml/features.py)
to prevent train/serve skew.  The model is cached in-process and reloaded
whenever metadata.json changes on disk.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select

from app.database import get_session
from app.models import Reading

# ── Path to the ml/ package ─────────────────────────────────────────────────
# config.py already added the project root to sys.path, so `import ml` works.
import ml.config as ml_cfg
from ml.features import build_features_for_machine, get_feature_columns

# ── Artifact paths ───────────────────────────────────────────────────────────
_MODEL_PATH    = ml_cfg.ARTIFACTS_DIR / "model.json"
_META_PATH     = ml_cfg.ARTIFACTS_DIR / "metadata.json"

# ── Model cache ──────────────────────────────────────────────────────────────
_model         = None
_meta: dict    = {}
_feat_cols: list[str] = []
_class_names: list[str] = ml_cfg.CLASS_NAMES
_meta_mtime: float = 0.0


def _load_model() -> bool:
    """Load or reload the model artifact if it exists and has changed."""
    global _model, _meta, _feat_cols, _class_names, _meta_mtime

    if not _MODEL_PATH.exists() or not _META_PATH.exists():
        return False

    mtime = _META_PATH.stat().st_mtime
    if mtime == _meta_mtime and _model is not None:
        return True  # already up-to-date

    try:
        from xgboost import XGBClassifier
    except ImportError:
        return False

    try:
        m = XGBClassifier()
        m.load_model(str(_MODEL_PATH))
        with open(_META_PATH) as f:
            meta = json.load(f)

        _model        = m
        _meta         = meta
        _feat_cols    = meta.get("feature_columns", get_feature_columns(ml_cfg))
        _class_names  = meta.get("class_names", ml_cfg.CLASS_NAMES)
        _meta_mtime   = mtime
        return True
    except Exception as e:
        print(f"[prediction_service] Failed to load model: {e}")
        return False


# ── Reading → DataFrame conversion ───────────────────────────────────────────

def _readings_to_df(readings: list) -> pd.DataFrame:
    """Convert a list of ORM Reading objects to a flat DataFrame."""
    rows = []
    for r in readings:
        rows.append({
            "session_id":    r.session_id,
            "seq":           r.seq,
            "machine_name":  r.machine_name,
            "state":         r.state,
            "ts":            r.ts,
            "lot_1":         r.lot_1,
            "lot_2":         r.lot_2,
            "speed":         r.speed,
            "length":        r.length,
            "lot_time_s":    r.lot_time_s,
            "machine_time_s": r.machine_time_s,
            "steam_consumed_lot": r.steam_consumed_lot,
            "water_consumed_lot": r.water_consumed_lot,
            "sf_flow":       r.sf_flow,
            "sf_tot":        r.sf_tot,
            "wat_flow":      r.wat_flow,
            "wat_tot":       r.wat_tot,
            "em_power":      r.em_power,
            "em_energy":     r.em_energy,
            "vibration_rms": r.vibration_rms,
            "motor_current": r.motor_current,
            "bearing_temp":  r.bearing_temp,
            "winding_temp":  r.winding_temp,
            "air_pressure":  r.air_pressure,
            "good_count":    r.good_count,
            "reject_count":  r.reject_count,
        })
    df = pd.DataFrame(rows)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize(None)
    return df


# ── Core scoring function ─────────────────────────────────────────────────────

def predict_machine(machine_name: str) -> dict:
    """Score the current state of one machine.

    Returns a dict with predicted_class, probabilities, risk_score, and metadata.
    """
    model_ready = _load_model()

    # Load the latest MAX_WINDOW + buffer readings (sorted ascending)
    n_load = ml_cfg.MAX_WINDOW + 20
    with get_session() as session:
        readings = list(session.scalars(
            select(Reading)
            .where(Reading.machine_name == machine_name)
            .order_by(Reading.ts.desc())
            .limit(n_load)
        ).all())

    if len(readings) < 5:
        return {
            "machine_name":    machine_name,
            "predicted_class": "insufficient_data",
            "probabilities":   {},
            "risk_score":      0.0,
            "model_version":   _meta.get("pipeline_version", "none"),
            "ts":              datetime.now(timezone.utc).isoformat(),
            "note":            f"Only {len(readings)} readings available (need ≥ {ml_cfg.MAX_WINDOW})",
        }

    # Reverse so ascending by ts
    readings = readings[::-1]
    df = _readings_to_df(readings)

    # Build features — same code path as training
    feat_df = build_features_for_machine(df, ml_cfg)

    if feat_df.empty:
        return {
            "machine_name":    machine_name,
            "predicted_class": "insufficient_data",
            "probabilities":   {},
            "risk_score":      0.0,
            "model_version":   _meta.get("pipeline_version", "none"),
            "ts":              datetime.now(timezone.utc).isoformat(),
            "note":            "Feature frame empty after engineering",
        }

    if not model_ready:
        return {
            "machine_name":    machine_name,
            "predicted_class": "model_not_trained",
            "probabilities":   {},
            "risk_score":      0.0,
            "model_version":   "none",
            "ts":              datetime.now(timezone.utc).isoformat(),
            "note":            f"No model artifact found at {_MODEL_PATH}. Run ml/train.py first.",
        }

    # Take the last feature row (most recent state)
    import numpy as np
    X = feat_df[_feat_cols].tail(1).astype(float).values

    probas     = _model.predict_proba(X)[0]
    pred_idx   = int(probas.argmax())
    pred_class = _class_names[pred_idx]

    # risk_score = P(any failure) = 1 - P(none)
    none_idx   = _class_names.index("none") if "none" in _class_names else 0
    risk_score = float(1.0 - probas[none_idx])

    result = {
        "machine_name":    machine_name,
        "predicted_class": pred_class,
        "probabilities":   {_class_names[i]: round(float(p), 4) for i, p in enumerate(probas)},
        "risk_score":      round(risk_score, 4),
        "model_version":   _meta.get("pipeline_version", "unknown"),
        "ts":              datetime.now(timezone.utc).isoformat(),
    }

    _log_prediction(machine_name, result, feat_df)
    return result


def predict_all() -> list[dict]:
    """Score all machines currently in machine_readings, sorted by risk (descending)."""
    with get_session() as session:
        from sqlalchemy import distinct
        machines = list(session.scalars(
            select(Reading.machine_name).distinct()
        ).all())

    results = [predict_machine(m) for m in sorted(machines)]
    results.sort(key=lambda r: r.get("risk_score", 0), reverse=True)
    return results


def current_model_summary() -> dict:
    """Return a summary of the loaded model artifact."""
    _load_model()
    if not _meta:
        return {"status": "no_model", "message": f"Run ml/train.py to train a model."}
    return {
        "status":            "loaded",
        "pipeline_version":  _meta.get("pipeline_version"),
        "class_names":       _meta.get("class_names", []),
        "n_features":        len(_feat_cols),
        "H_hours":           _meta.get("H_hours"),
        "window_sizes":      _meta.get("window_sizes"),
        "best_iteration":    _meta.get("best_iteration"),
    }


# ── Prediction logging ────────────────────────────────────────────────────────

def _log_prediction(machine_name: str, result: dict, feat_df: pd.DataFrame) -> None:
    """Write one row to the predictions table (best-effort, non-blocking)."""
    try:
        from app.database import _engine
        from sqlalchemy import text
        import json as _json

        last_row = feat_df.tail(1).iloc[0]
        seq = int(last_row.get("seq", 0)) if "seq" in last_row.index else 0
        sid = str(last_row.get("session_id", "")) if "session_id" in last_row.index else ""

        with _engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id              BIGINT PRIMARY KEY AUTO_INCREMENT,
                    ts              DATETIME(3) NOT NULL,
                    session_id      VARCHAR(36),
                    seq             INT,
                    machine_name    VARCHAR(64) NOT NULL,
                    predicted_class VARCHAR(32) NOT NULL,
                    probabilities   JSON,
                    model_version   VARCHAR(64),
                    KEY idx_pred_ts      (ts),
                    KEY idx_pred_machine (machine_name)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """))
            conn.execute(text("""
                INSERT INTO predictions
                    (ts, session_id, seq, machine_name, predicted_class, probabilities, model_version)
                VALUES
                    (:ts, :sid, :seq, :mn, :pc, :pb, :mv)
            """), {
                "ts":  datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "sid": sid,
                "seq": seq,
                "mn":  machine_name,
                "pc":  result["predicted_class"],
                "pb":  _json.dumps(result.get("probabilities", {})),
                "mv":  result.get("model_version", ""),
            })
    except Exception:
        pass  # prediction logging is best-effort
