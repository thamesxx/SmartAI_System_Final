# Phase 2 — Plan 03: ML Lifecycle (MongoDB + MLflow) & Deployment

> **Goal of this document.** A complete, executable plan to turn the one-off model
> from Plan 02 into a **continuously-training, self-monitoring system**: MongoDB
> holds all data + predictions, **MLflow** owns experiment tracking and the
> versioned model registry, an automated pipeline **retrains / evaluates /
> promotes** models, **drift** is detected, a **feedback loop** measures live
> accuracy, and the whole stack is **deployed**.
>
> Depends on `PHASE2_01_SYNTHETIC_DATA_GENERATION.md` (data + labels) and
> `PHASE2_02_MODEL_TRAINING_EVALUATION.md` (the `ml/` package + serving).

---

## 0. Lifecycle at a glance

```
            ┌──────────────────────────── MongoDB (machine_telemetry) ───────────────────────────┐
            │ machine_readings  machine_runs  features  predictions  drift_metrics  model_registry │
            └───────┬───────────────┬──────────────────────┬───────────────────────┬──────────────┘
                    │ (extract)     │ (labels)             │ (live preds, Plan 02)  │ (dashboard pointer)
                    ▼               ▼                       │                        ▲
        ┌─────────────────────────────────┐                │                        │
        │  Retraining pipeline (ml/train)  │                │                        │
        │  extract→features→train→eval     │                │                        │
        │  →champion/challenger→promote    │──── log ──────►│                        │
        └───────────────┬─────────────────┘     params/metrics/artifacts            │
                        ▼                                                            │
                ┌───────────────┐    promote to "Production"     ┌──────────────────┴─┐
                │    MLflow      │◄──────────────────────────────│  registry.py glue   │
                │ tracking+reg.  │────── load Production ────────►│ (Plan 02 serving)   │
                └───────┬────────┘                                └─────────────────────┘
                        │ triggers
        ┌───────────────┴───────────────┬─────────────────────────────┐
        ▼                               ▼                             ▼
 APScheduler (nightly)           ml/drift.py (PSI/KS)        feedback loop (preds vs machine_runs)
```

---

# PART A — MongoDB collections (the lifecycle data model)

| Collection         | Written by                  | Role                                                             |
|--------------------|-----------------------------|-----------------------------------------------------------------|
| `machine_readings` | consumer / backfill (Plan 01)| raw extended telemetry                                          |
| `machine_runs`     | consumer / backfill (Plan 01)| failure/maintenance events = **label source** + outcome truth   |
| `features`         | `ml/dataset.py` (optional)  | materialized feature rows for reproducible training             |
| `predictions`      | backend serving (Plan 02)   | every live prediction `{ts, session_id, seq, class, probs, model_version}` |
| `drift_metrics`    | `ml/drift.py`               | periodic PSI/KS per feature + predicted-class-rate shift         |
| `model_registry`   | `ml/registry.py`            | **dashboard pointer**: active version, stage, headline metrics, trained_at |

Indexes: `predictions(session_id, ts)`, `drift_metrics(ts)`,
`model_registry(version)`. Mongo stays on Atlas (per `backend/app/config.py`).

> **Model binaries** live in **MLflow** (and/or **GridFS** as a Mongo-native
> backup). `model_registry` holds only metadata + a pointer, so the dashboard can
> show "Production model v7, macro-F1 0.82, trained 2026-06-19" cheaply.

---

# PART B — MLflow (tracking + registry)

## B.1 Tracking
Every training run (Plan 02 `train.py`, now wrapped) logs to MLflow:
- **params** — H, window sizes, XGBoost hyperparameters, data window (date range,
  sessions used, row counts);
- **metrics** — macro-F1, per-class recall/precision, PR-AUC, lead-time median,
  baseline deltas;
- **artifacts** — the `model.json` bundle (B.8 of Plan 02), confusion matrix PNG,
  SHAP summary, the frozen `FEATURE_COLUMNS`.

```python
import mlflow
mlflow.set_tracking_uri(MLFLOW_URI)          # local file store, or MLflow server
mlflow.set_experiment("machine-pdm")
with mlflow.start_run():
    mlflow.log_params(params); mlflow.log_metrics(metrics)
    mlflow.log_artifact("confusion.png"); mlflow.log_artifact("shap.png")
    mlflow.xgboost.log_model(model, artifact_path="model",
                             registered_model_name="machine-pdm")
```

## B.2 Registry + stages
Registered model `machine-pdm` moves through stages:
`None → Staging → Production → Archived`.

## B.3 Champion / challenger promotion
1. New model is logged and registered → **Staging**.
2. Evaluate the **Staging** candidate on a **fresh holdout** (latest runs).
3. Load current **Production** ("champion"); evaluate on the *same* holdout.
4. **Promote** challenger → Production **only if** it beats champion by a margin on
   the key metric (macro-F1), else archive it.
5. On promotion, archive the old Production and write a summary doc to Mongo
   `model_registry` for the dashboard.
6. Start **manual-gated** (require a human OK); flip to fully automatic once trusted
   (the master plan's open follow-up).

The backend (Plan 02 serving) always loads the **Production** model and refreshes
its cache when the registry version changes.

---

# PART C — Retraining pipeline

## C.1 What it does (`ml/train.py`, extended from Plan 02)
```text
extract (Mongo: readings + machine_runs for the training window)
  → build features + labels (ml/features.py, ml/labels.py)
  → grouped/time split (ml/dataset.py)
  → train candidate (tuned, early stopping)
  → evaluate on held-out recent runs (ml/evaluate.py)
  → MLflow: log params/metrics/artifacts, register → Staging
  → champion/challenger (Part B.3) → maybe promote to Production
  → write pointer/summary to model_registry
```

## C.2 Triggers (`ml/schedule.py`)
Use **APScheduler** (in-process with the backend, or a standalone worker):
- **Scheduled** — nightly / weekly full retrain.
- **Drift-triggered** — Part D flags significant drift → kick a retrain.
- **Volume-triggered** — ≥ N new failure events in `machine_runs` since last train.

```python
from apscheduler.schedulers.background import BackgroundScheduler
sched = BackgroundScheduler()
sched.add_job(run_retraining, "cron", hour=2)              # nightly
sched.add_job(check_drift_and_maybe_retrain, "interval", hours=6)
sched.start()
```
(For a simpler FYP deployment, a cron/Windows-Task running `python -m ml.train` is
an acceptable substitute.)

---

# PART D — Drift detection (`ml/drift.py`)

Distribution shift degrades models silently. Periodically compare **recent**
feature distributions against the **training reference**:

- **PSI (Population Stability Index)** per feature:
  `PSI = Σ (recent% − ref%) * ln(recent% / ref%)` over bins.
  Rule of thumb: `<0.1` stable, `0.1–0.25` moderate, `>0.25` significant.
- **KS test** per continuous feature for distribution change.
- **Prediction drift** — shift in predicted class-rate vs training base-rate.

Write results to `drift_metrics` `{ts, feature, psi, ks_p, flag}`. If key features
exceed the PSI threshold → set a flag and trigger retraining (Part C). Optionally
use **Evidently AI** to generate nicer drift reports/dashboards.

---

# PART E — Monitoring / feedback loop (the "self-improving" story)

Closes the loop promised in the proposal:
1. Live predictions accumulate in `predictions` (Plan 02 serving).
2. When a real failure lands in `machine_runs`, **join** it back to the
   predictions that preceded it.
3. Compute rolling **live precision / recall / lead-time** per class.
4. Store these and **surface on the dashboard** (model-health panel) and feed the
   promotion decision (a model that degrades in production gets replaced).

```python
def evaluate_live_outcomes(window_days=7) -> dict:
    # match predictions → subsequent machine_runs failures within H
    # → live per-class recall/precision + lead-time → store + return
```

---

# PART F — Registry glue (`ml/registry.py`)

The single seam between MLflow, GridFS, Mongo, and the serving backend:
```python
def load_production_model():            # fetch current Production from MLflow (cache)
def current_model_summary() -> dict:    # version, stage, metrics for /api/model/info
def promote_if_better(candidate, holdout, margin): ...   # champion/challenger (B.3)
def mirror_to_mongo(summary): ...       # upsert model_registry pointer doc
def backup_to_gridfs(model_bundle): ... # optional Mongo-native binary backup
```
Plan 02's `prediction_service._load_model()` calls `load_production_model()` and
refreshes when the version changes.

---

# PART G — Final deployment

## G.1 Containerize
Dockerfiles already exist for producer/consumer. Add/confirm images for:
- **backend** — FastAPI + uvicorn (serves API + loads Production model);
- **consumer** — RabbitMQ → Mongo;
- **generator/publisher** — live stream (and an on-demand backfill job);
- **mlflow** — tracking server + registry UI;
- **frontend** — `vite build` static bundle behind **nginx**;
- **scheduler** — APScheduler worker (or fold into backend) for retrain/drift.

## G.2 Compose the stack (top-level `docker-compose.yml`)
Wire: RabbitMQ + consumer + generator + backend + mlflow + frontend + scheduler.
Secrets/config via `.env` (already git-ignored). **MongoDB stays on Atlas** (no
container). Example service sketch:
```yaml
services:
  rabbitmq:   { image: rabbitmq:3-management, ports: ["5672:5672","15672:15672"] }
  consumer:   { build: ./rabbitmq/consumer,            env_file: .env, depends_on: [rabbitmq] }
  generator:  { build: ./rabbitmq/producer/publisher,  env_file: .env, depends_on: [rabbitmq] }
  mlflow:     { image: ghcr.io/mlflow/mlflow, command: mlflow server ..., ports: ["5000:5000"] }
  backend:    { build: ./backend, env_file: .env, ports: ["8000:8000"], depends_on: [mlflow] }
  frontend:   { build: ./frontend, ports: ["80:80"] }   # nginx serving the build
```

## G.3 Hosting
- **Backend + MLflow** on a small VM or Render; **frontend** as a static build on
  Render / Netlify / Firebase Hosting (per the proposal); point the frontend at the
  deployed API URL via env.
- Persist the MLflow backend store + artifact store (volume or cloud bucket).

## G.4 Ops
- Health checks + restart policies on every service.
- Scheduler container (or backend-embedded APScheduler) drives retraining + drift.
- Document all run commands in the repo `README` (start stack, run backfill, force
  retrain, view MLflow UI).

---

## Step-by-step execution (lifecycle → deployment)
1. Create the Mongo collections + indexes (Part A); confirm `predictions` already
   fills from Plan 02 serving.
2. Add MLflow logging to `train.py`; run once → see the run + registered model in
   the MLflow UI (Part B.1–B.2).
3. Implement `registry.py` champion/challenger + `mirror_to_mongo` (Parts B.3, F);
   point backend serving at `load_production_model()`.
4. Implement `drift.py` (Part D); run on recent vs reference; write `drift_metrics`.
5. Implement `schedule.py` triggers (Part C.2); verify a scheduled retrain runs.
6. Implement the feedback loop (Part E); surface live metrics on the dashboard.
7. Containerize + compose (Part G.1–G.2); bring the stack up locally.
8. Deploy to the chosen host (Part G.3); smoke-test end-to-end.

## Verification & acceptance
- **Tracking/registry** — MLflow UI shows runs with params/metrics/artifacts and a
  versioned `machine-pdm`; stages transition correctly.
- **Promotion** — a deliberately-better candidate gets promoted to Production and
  the old one archived; `model_registry` pointer + `/api/model/info` update; the
  backend serves the new version without restart.
- **Drift** — artificially shift a feature distribution → `drift_metrics` flags it
  and (if wired) a retrain triggers.
- **Feedback loop** — live precision/recall/lead-time computed from
  `predictions` × `machine_runs` and shown on the dashboard.
- **Deployment** — `docker compose up` brings the full stack online; the deployed
  frontend hits the live API and shows OEE + live predictions end-to-end; MLflow UI
  reachable.

## Pitfalls
- **Train/serve consistency** — retraining and serving must share `ml/features.py`
  and the persisted `FEATURE_COLUMNS`/config; version them with the model.
- **Auto-promote too early** — start with a manual gate; a bad auto-promotion can
  silently break live predictions.
- **Drift reference staleness** — refresh the reference distribution when a new
  Production model is promoted, else drift fires forever.
- **Holdout leakage in promotion** — champion and challenger must be judged on the
  *same* fresh, never-trained-on holdout (grouped/time split, as in Plan 02 B.5).
- **Secrets** — keep Atlas/RabbitMQ creds in `.env` only (already git-ignored);
  never bake into images.
