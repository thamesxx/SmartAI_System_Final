"""Dataset assembly and leakage-safe train/val/test splitting.

Key guarantees
--------------
1. No random split -- split is purely by simulation timestamp.
2. A GAP_HOURS sim-hour buffer sits between each split boundary to avoid
   horizon bleed (a reading in train cannot have its label sourced from
   events that would appear in val/test).
3. truth_json is used for labels but never included in feature columns X.
4. Readings are streamed ONE MACHINE AT A TIME to avoid loading 12M+ rows
   into RAM simultaneously.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import ml.config as cfg
from ml.db import get_engine, load_readings, load_runs, iter_machines, ts_range
from ml.features import build_features_for_machine, get_feature_columns
from ml.labels import build_labels, build_labels_from_truth_json


def _encode_labels(y_str: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    le.classes_ = np.array(cfg.CLASS_NAMES)
    y_enc = le.transform(y_str.fillna("none"))
    return y_enc, le


def make_datasets(verbose: bool = True) -> dict:
    """Load data, build features + labels, and split into train/val/test.

    Streams one machine at a time -- safe for 12M+ row tables.

    Returns
    -------
    dict with keys:
        X_train, y_train, groups_train,
        X_val,   y_val,   groups_val,
        X_test,  y_test,  groups_test,
        feature_columns, label_encoder,
        meta_test  (DataFrame with ts, machine_name, session_id)
    """
    engine    = get_engine()
    feat_cols = get_feature_columns(cfg)

    # Load failure events (small -- always safe to load fully)
    if verbose:
        print("Loading machine_runs from MySQL ...")
    runs_df = load_runs(engine)
    use_truth_json = len(runs_df) == 0
    if verbose:
        print(f"  {len(runs_df):,} failure events")
        if use_truth_json:
            print("  machine_runs is empty -- will derive labels from truth_json")

    # Get global timestamp range without loading all readings
    if verbose:
        print("Fetching global timestamp range ...")
    ts_min, ts_max = ts_range(engine)
    span = (ts_max - ts_min).total_seconds()
    gap  = timedelta(hours=cfg.GAP_HOURS)

    cut_train = ts_min + timedelta(seconds=span * cfg.TRAIN_FRAC)
    cut_val   = ts_min + timedelta(seconds=span * cfg.VAL_FRAC)

    if verbose:
        print(f"  Range: {ts_min}  ->  {ts_max}")
        print(f"  Train cut : {cut_train}")
        print(f"  Val   cut : {cut_val}")

    machines = iter_machines(engine)
    if verbose:
        print(f"  {len(machines)} machines: {machines}")

    # Stream per machine, build features, assign labels
    if verbose:
        print("\nBuilding features per machine (streaming) ...")

    feat_parts: list[pd.DataFrame] = []

    for machine in machines:
        if verbose:
            print(f"  {machine} ... ", end="", flush=True)

        # include_truth=True so we have truth_json for label fallback
        mdf = load_readings(engine, machine_name=machine, include_truth=True)
        if verbose:
            print(f"{len(mdf):,} readings ... ", end="", flush=True)

        # Build feature frame (uses sorted mdf, drops truth_json, keeps ts/seq)
        part = build_features_for_machine(mdf, cfg)
        if part.empty:
            if verbose:
                print("no features (too few readings)")
            del mdf
            continue

        # Assign labels
        if use_truth_json:
            # Join truth_json back onto feature rows by ts+machine_name
            # part has 'ts'; mdf has 'ts' and 'truth_json'
            truth_map = (
                mdf[["ts", "truth_json"]]
                .drop_duplicates("ts")
                .set_index("ts")["truth_json"]
            )
            # Build a mini-df with machine_name + ts + truth_json aligned to part rows
            part_for_labels = part[["machine_name", "ts"]].copy()
            part_for_labels["truth_json"] = part_for_labels["ts"].map(truth_map)
            labels = build_labels_from_truth_json(part_for_labels, cfg.H_SECONDS)
        else:
            labels = build_labels(part, runs_df, cfg.H_SECONDS)

        part["label"] = labels.values
        part["label"] = part["label"].fillna("none")

        if verbose:
            n_fail = (part["label"] != "none").sum()
            print(f"{len(part):,} feature rows  ({n_fail} failure-horizon rows)")

        feat_parts.append(part)
        del mdf  # release raw readings immediately

    if not feat_parts:
        raise RuntimeError("No feature rows produced -- check data and config.")

    feat_df = pd.concat(feat_parts, ignore_index=True)
    del feat_parts

    if verbose:
        print(f"\nTotal feature rows: {len(feat_df):,}")
        dist = feat_df["label"].value_counts()
        print("Label distribution:")
        for cls, cnt in dist.items():
            print(f"  {cls:15s}: {cnt:>8,}  ({cnt / len(feat_df) * 100:.1f}%)")

    # Time-based split
    train_mask = feat_df["ts"] < cut_train
    val_mask   = (feat_df["ts"] >= cut_train + gap) & (feat_df["ts"] < cut_val)
    test_mask  = feat_df["ts"] >= cut_val + gap

    if verbose:
        print(f"\nSplit sizes: train={train_mask.sum():,}  "
              f"val={val_mask.sum():,}  test={test_mask.sum():,}")

    def _split(mask: pd.Series):
        sub = feat_df[mask]
        X      = sub[feat_cols].astype(float).values
        y, le  = _encode_labels(sub["label"])
        groups = sub["session_id"].values
        meta   = sub[["ts", "machine_name", "session_id"]].copy()
        return X, y, groups, meta, le

    X_train, y_train, g_train, _,         le = _split(train_mask)
    X_val,   y_val,   g_val,   _,         _  = _split(val_mask)
    X_test,  y_test,  g_test,  meta_test, _  = _split(test_mask)

    return dict(
        X_train=X_train, y_train=y_train, groups_train=g_train,
        X_val=X_val,     y_val=y_val,     groups_val=g_val,
        X_test=X_test,   y_test=y_test,   groups_test=g_test,
        feat_df=feat_df,
        feature_columns=feat_cols,
        label_encoder=le,
        meta_test=meta_test,
    )
