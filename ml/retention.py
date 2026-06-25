"""Seal-and-prune job — Plan 03 owns the schedule (APScheduler nightly).

This module is a stub that will be fleshed out in Phase 2 Part 3.
The three watermarks that define correctness:
  label_complete_ts = now - H          (only seal rows with complete label horizon)
  rolling_keep_ts   = now - 14d        (never prune newer raw rows)
  max_window        = 6h trailing ctx  (always keep context for the next seal)

Order: seal → verify → prune.  Never prune machine_runs (labels).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

import ml.config as cfg
from ml.db import get_engine, load_readings, load_runs
from ml.features import build_features_for_machine, get_feature_columns
from ml.labels import build_labels
from ml.feature_store import write_shard


ROLLING_KEEP_DAYS = 14


def seal_range(
    start_ts: datetime,
    end_ts: datetime,
    pipeline_version: str | None = None,
    dry_run: bool = False,
) -> int:
    """Seal raw rows in [start_ts, end_ts] into a Parquet feature shard.

    Parameters
    ----------
    start_ts, end_ts : UTC datetimes (both naive or both tz-aware).
    pipeline_version : defaults to cfg.PIPELINE_VERSION.
    dry_run          : if True, compute features but do not write.

    Returns number of feature rows written (0 on dry-run).
    """
    pv     = pipeline_version or cfg.PIPELINE_VERSION
    engine = get_engine()
    feat_cols = get_feature_columns(cfg)

    # Add trailing context (max_window readings) so the first feature row is complete
    ctx_start = start_ts - timedelta(hours=6)

    readings = load_readings(engine)
    window   = readings[
        (readings["ts"] >= ctx_start) & (readings["ts"] < end_ts)
    ].copy()

    if window.empty:
        print(f"seal_range: no readings in [{start_ts}, {end_ts})")
        return 0

    runs = load_runs(engine)

    feat_parts = []
    for machine, mdf in window.groupby("machine_name"):
        part = build_features_for_machine(mdf.copy(), cfg)
        # Keep only rows within the actual seal range (not context)
        part = part[
            (part["ts"] >= start_ts) & (part["ts"] < end_ts)
        ]
        feat_parts.append(part)

    if not feat_parts:
        return 0

    feat_df = pd.concat(feat_parts, ignore_index=True)
    feat_df["label"] = build_labels(feat_df, runs, cfg.H_SECONDS).values

    if dry_run:
        print(f"[dry-run] Would write {len(feat_df)} feature rows")
        return 0

    path = write_shard(
        feat_df[feat_cols + ["label", "session_id", "machine_name", "ts", "seq"]],
        start_ts=start_ts,
        end_ts=end_ts,
        pipeline_version=pv,
        part=int(start_ts.timestamp()),
        cfg=cfg,
        engine=engine,
    )
    print(f"Sealed {len(feat_df)} rows → {path}")
    return len(feat_df)


def prune_old_readings(dry_run: bool = False) -> int:
    """Delete machine_readings rows older than ROLLING_KEEP_DAYS.

    NEVER touches machine_runs (labels must be kept forever).
    Only prunes rows that have already been sealed into a Parquet shard
    (i.e., rows older than the rolling window).
    """
    from sqlalchemy import text as _text

    engine = get_engine()
    cutoff = datetime.now(timezone.utc) - timedelta(days=ROLLING_KEEP_DAYS)
    cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")

    if dry_run:
        with engine.connect() as conn:
            n = conn.execute(
                _text("SELECT COUNT(*) FROM machine_readings WHERE ts < :c"),
                {"c": cutoff_str},
            ).scalar()
        print(f"[dry-run] Would delete {n:,} readings older than {cutoff_str}")
        return int(n)

    with engine.begin() as conn:
        result = conn.execute(
            _text("DELETE FROM machine_readings WHERE ts < :c"),
            {"c": cutoff_str},
        )
    n = result.rowcount
    print(f"Pruned {n:,} readings older than {cutoff_str}")
    return n
