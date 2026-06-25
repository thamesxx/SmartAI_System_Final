"""Model training with optional Optuna hyperparameter tuning.

Usage
-----
  cd d:\\Ali Stuff\\Taha_fyp
  python -m ml.train                      # train with defaults
  python -m ml.train --tune               # run Optuna first
  python -m ml.train --tune --trials 50   # more Optuna trials

Artifact saved to ml/artifacts/model.json + ml/artifacts/metadata.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.utils.class_weight import compute_sample_weight

import ml.config as cfg
from ml.dataset import make_datasets
from ml.evaluate import evaluate_model

try:
    from xgboost import XGBClassifier
    import xgboost as xgb
except ImportError:
    raise SystemExit("xgboost not installed — run: pip install xgboost")


# ── Sample weights (inverse class frequency) ─────────────────────────────────

def _sample_weights(y: np.ndarray) -> np.ndarray:
    return compute_sample_weight("balanced", y)


# ── Optuna tuning ─────────────────────────────────────────────────────────────

def tune(X_train, y_train, X_val, y_val, n_trials: int = cfg.OPTUNA_TRIALS) -> dict:
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("optuna not installed — skipping tuning (pip install optuna)")
        return {}

    from sklearn.metrics import f1_score

    def objective(trial):
        params = {
            **cfg.XGB_DEFAULTS,
            "n_estimators":     trial.suggest_int("n_estimators", 100, 800),
            "max_depth":        trial.suggest_int("max_depth", 3, 9),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma":            trial.suggest_float("gamma", 0.0, 2.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        }
        model = XGBClassifier(**params, early_stopping_rounds=30, verbosity=0)
        model.fit(
            X_train, y_train,
            sample_weight=_sample_weights(y_train),
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        y_pred = model.predict(X_val)
        return f1_score(y_val, y_pred, average="macro", zero_division=0)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=cfg.OPTUNA_TIMEOUT, show_progress_bar=True)

    print(f"\nBest macro-F1 (val): {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study.best_params


# ── Train ─────────────────────────────────────────────────────────────────────

def train_model(tune_params: dict | None = None) -> tuple:
    """Train XGBoost and return (model, feature_columns, datasets_dict)."""
    t0 = time.time()
    print("\n" + "="*60)
    print("Phase 2 — XGBoost Predictive Maintenance Training")
    print("="*60)

    data = make_datasets(verbose=True)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val   = data["X_val"]
    y_val   = data["y_val"]
    feat_cols = data["feature_columns"]

    # Merge tuned params over defaults
    params = {**cfg.XGB_DEFAULTS, **(tune_params or {})}

    print(f"\nTraining XGBoost  "
          f"(train={len(X_train):,}  val={len(X_val):,}  features={len(feat_cols)})")

    model = XGBClassifier(
        **params,
        early_stopping_rounds=cfg.EARLY_STOPPING_ROUNDS,
    )

    model.fit(
        X_train, y_train,
        sample_weight=_sample_weights(y_train),
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    best_iter = model.best_iteration
    print(f"\nBest iteration: {best_iter}")
    print(f"Training time : {time.time() - t0:.1f}s")

    return model, feat_cols, data


# ── Save artifact ─────────────────────────────────────────────────────────────

def save_artifact(model, feat_cols: list[str]) -> None:
    cfg.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = cfg.ARTIFACTS_DIR / "model.json"
    model.save_model(str(model_path))

    meta = {
        "feature_columns":   feat_cols,
        "class_names":       cfg.CLASS_NAMES,
        "pipeline_version":  cfg.PIPELINE_VERSION,
        "H_hours":           cfg.H_HOURS,
        "window_sizes":      cfg.WINDOW_SIZES,
        "nominal_speed":     cfg.NOMINAL_SPEED,
        "best_iteration":    int(model.best_iteration) if hasattr(model, "best_iteration") else None,
    }
    meta_path = cfg.ARTIFACTS_DIR / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\nArtifact saved:")
    print(f"  Model   : {model_path}")
    print(f"  Metadata: {meta_path}")
    print(f"  Pipeline version: {cfg.PIPELINE_VERSION}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train predictive maintenance model")
    parser.add_argument("--tune",   action="store_true", help="Run Optuna tuning first")
    parser.add_argument("--trials", type=int, default=cfg.OPTUNA_TRIALS, help="Optuna trials")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation after training")
    args = parser.parse_args()

    tune_params = {}
    if args.tune:
        # Need data for tuning
        print("Loading data for Optuna ...")
        data_tmp = make_datasets(verbose=False)
        tune_params = tune(
            data_tmp["X_train"], data_tmp["y_train"],
            data_tmp["X_val"],   data_tmp["y_val"],
            n_trials=args.trials,
        )

    model, feat_cols, data = train_model(tune_params or None)
    save_artifact(model, feat_cols)

    if not args.no_eval:
        print("\n" + "="*60)
        print("Evaluation on test set")
        print("="*60)
        evaluate_model(
            model,
            data["X_test"], data["y_test"],
            data["meta_test"],
            data["label_encoder"],
        )


if __name__ == "__main__":
    import sys, os
    # Ensure project root is in path when running as script
    sys.path.insert(0, str(Path(__file__).parent.parent))
    main()
