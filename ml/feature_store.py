"""Parquet feature-shard read/write + feature_snapshots catalog helpers.

Shards live at:
  FEATURE_STORE_DIR/<pipeline_version>/dt=YYYY-MM-DD/part-<N>.parquet

The catalog (feature_snapshots MySQL table) records each shard's range,
row count, class distribution, and a reference feature distribution
used by drift detection once raw rows are pruned.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sqlalchemy import text

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine


def shard_path(pipeline_version: str, start_ts: datetime, part: int, cfg) -> Path:
    date_str = start_ts.strftime("%Y-%m-%d")
    p = cfg.FEATURE_STORE_DIR / pipeline_version / f"dt={date_str}" / f"part-{part:04d}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_shard(
    df: pd.DataFrame,
    start_ts: datetime,
    end_ts: datetime,
    pipeline_version: str,
    part: int,
    cfg,
    engine: "Engine | None" = None,
) -> Path:
    """Write a feature DataFrame to a Parquet shard and register it in the catalog."""
    path = shard_path(pipeline_version, start_ts, part, cfg)
    df.to_parquet(path, index=False, compression="snappy")

    # Class distribution
    if "label" in df.columns:
        class_counts = df["label"].value_counts().to_dict()
    else:
        class_counts = {}

    # Reference distribution: per-feature mean/std (used for drift detection)
    feat_cols = [c for c in df.columns if c not in ("session_id", "machine_name", "ts", "seq", "label")]
    ref_dist = {
        col: {"mean": float(df[col].mean()), "std": float(df[col].std())}
        for col in feat_cols
        if df[col].notna().any()
    }

    if engine is not None:
        _catalog_upsert(engine, pipeline_version, str(path), start_ts, end_ts,
                        len(df), class_counts, ref_dist)

    return path


def read_shards(pipeline_version: str, cfg) -> pd.DataFrame:
    """Read all Parquet shards for a pipeline version into a single DataFrame."""
    base = cfg.FEATURE_STORE_DIR / pipeline_version
    if not base.exists():
        return pd.DataFrame()

    parts = sorted(base.rglob("*.parquet"))
    if not parts:
        return pd.DataFrame()

    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def _catalog_upsert(
    engine: "Engine",
    pipeline_version: str,
    shard_path: str,
    start_ts: datetime,
    end_ts: datetime,
    row_count: int,
    class_counts: dict,
    ref_dist: dict,
) -> None:
    """Insert a row into feature_snapshots (create table if missing)."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS feature_snapshots (
                id               BIGINT PRIMARY KEY AUTO_INCREMENT,
                pipeline_version VARCHAR(64)  NOT NULL,
                shard_path       VARCHAR(512) NOT NULL,
                range_start_ts   DATETIME(3)  NOT NULL,
                range_end_ts     DATETIME(3)  NOT NULL,
                row_count        INT          NOT NULL,
                class_counts     JSON         NOT NULL,
                ref_dist         JSON,
                created_at       DATETIME(3)  NOT NULL,
                KEY idx_snap_version (pipeline_version),
                KEY idx_snap_range   (range_start_ts, range_end_ts)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """))

        conn.execute(text("""
            INSERT INTO feature_snapshots
                (pipeline_version, shard_path, range_start_ts, range_end_ts,
                 row_count, class_counts, ref_dist, created_at)
            VALUES
                (:pv, :sp, :rs, :re, :rc, :cc, :rd, :ca)
        """), {
            "pv": pipeline_version,
            "sp": shard_path,
            "rs": start_ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "re": end_ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "rc": row_count,
            "cc": json.dumps(class_counts),
            "rd": json.dumps(ref_dist),
            "ca": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        })
