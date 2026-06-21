# Phase 2 — Plan 04: MySQL Migration + Sampling-Rate & Data-Volume Notes

> **Goal of this document.** Three things in one place:
> 1. **Migrate the data store from MongoDB → MySQL** using **SQLAlchemy** (ORM/engine)
>    and **PyMySQL** (driver).
> 2. Answer the **sampling-rate question** for Phase 2 Part 1 / the XGBoost model:
>    **seconds vs minutes vs hours** — which is better for processing & memory.
> 3. Record the **observed data loss** (≈500k of 1.2M readings landed before the
>    Atlas cluster filled) and why the migration directly fixes it.
>
> Companion docs: `PHASE2_01_SYNTHETIC_DATA_GENERATION.md`,
> `PHASE2_02_MODEL_TRAINING_EVALUATION.md`, `PHASE2_03_ML_LIFECYCLE.md`.

---

# PART A — Migrate MongoDB → MySQL (SQLAlchemy + PyMySQL)

## A.0 Why move

Today MongoDB is the sink for both collections:

```
publisher.py / backfill.py → RabbitMQ → consumer.py → MongoDB
        ( machine_telemetry.machine_readings  +  machine_telemetry.machine_runs )
        → FastAPI backend (backend/app/...) → React dashboard
```

Reasons to move to MySQL:

- **Hard storage cap hit** — the free Atlas tier filled at ~500k/1.2M readings
  (see Part C). MySQL (self-hosted or a sized instance) removes that wall and stores
  the same data far more compactly with typed columns instead of repeated BSON keys.
- **The schema is actually relational** — `machine_readings` and `machine_runs` join
  on `session_id` (exactly what `ml/labels.py` does in Plan 02). SQL does that join
  natively and cheaply.
- **Analytics fit** — the OEE buckets, per-state durations, and totalizer-rate
  features in Plan 02 are `GROUP BY` / window-function queries — SQL's home turf.
- **One driver story** — `SQLAlchemy` + `PyMySQL` gives the backend, consumer, and ML
  layer a single, typed access path (`from sqlalchemy import select`) instead of
  ad-hoc PyMongo dict handling.

**Dependencies to add** (`requirements.txt` / `pip install`):

```
SQLAlchemy>=2.0
PyMySQL>=1.1
cryptography        # PyMySQL needs this for MySQL 8 caching_sha2_password auth
alembic             # optional, for schema migrations
```

Connection URL (SQLAlchemy + PyMySQL dialect):

```
mysql+pymysql://<user>:<password>@<host>:3306/machine_telemetry?charset=utf8mb4
```

## A.1 Schema design

The Mongo documents are nested (`plc.*`, `utility.*`, `health.*`, `quality.*`,
`_truth.*` — see Plan 01 §11). In MySQL, **flatten the nested objects into typed
columns** on two tables. Keep `_truth.*` columns **only for dev/debug** and never feed
them to the model (the leakage rule from Plan 02).

> Use real numeric types, not strings. The current PLC/utility values arrive as
> strings like `"24.81"` — store them as `DECIMAL`/`DOUBLE` so the `to_float()`
> parsing in `database.py` is no longer needed on read.

### `machine_readings`

```sql
CREATE TABLE machine_readings (
  id              BIGINT       NOT NULL AUTO_INCREMENT PRIMARY KEY,
  session_id      VARCHAR(64)  NOT NULL,
  seq             INT          NOT NULL,
  machine_name    VARCHAR(64)  NOT NULL,
  state           VARCHAR(16)  NOT NULL,            -- running|idle|changeover|maintenance|error
  ts              DATETIME(3)  NOT NULL,            -- was "timestamp" (reserved-ish; rename)

  -- plc.*
  lot_1           BIGINT       NULL,
  lot_2           BIGINT       NULL,
  article         INT          NULL,
  speed           DOUBLE       NULL,
  length          DOUBLE       NULL,
  runmemory       TINYINT(1)   NULL,
  lot_time_min    INT          NULL,                -- "H:M:S" → minutes (hms_to_minutes)
  machine_time_min BIGINT      NULL,                -- "H:M:S" → minutes
  steam_consumed_lot   DOUBLE  NULL,
  water_consumed_lot   DOUBLE  NULL,
  power_consumed_lot   DOUBLE  NULL,
  air_consumed_lot     DOUBLE  NULL,

  -- utility.*
  sf_flow         DOUBLE       NULL,
  sf_tot          DOUBLE       NULL,
  wat_flow        DOUBLE       NULL,
  wat_tot         DOUBLE       NULL,
  em_power        DOUBLE       NULL,                -- kW (now populated)
  em_energy       DOUBLE       NULL,                -- kWh

  -- health.*  (PM sensors)
  vibration_rms   DOUBLE       NULL,
  motor_current   DOUBLE       NULL,
  bearing_temp    DOUBLE       NULL,
  winding_temp    DOUBLE       NULL,
  air_pressure    DOUBLE       NULL,

  -- quality.*
  good_count      INT          NULL,
  reject_count    INT          NULL,

  -- _truth.*  (DEV ONLY — exclude from features)
  truth_json      JSON         NULL,                -- keep hidden truth as one JSON blob

  UNIQUE KEY uq_session_seq (session_id, seq),
  KEY idx_ts (ts),
  KEY idx_machine (machine_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

> The three indexes mirror exactly what `consumer.py` creates today:
> `(session_id, seq)`, `(timestamp)`, `(machine_name)`. Add the **unique** key on
> `(session_id, seq)` so a publisher restart can `INSERT ... ON DUPLICATE KEY UPDATE`
> (idempotent re-delivery) instead of duplicating rows.

### `machine_runs`

```sql
CREATE TABLE machine_runs (
  id                   BIGINT       NOT NULL AUTO_INCREMENT PRIMARY KEY,
  session_id           VARCHAR(64)  NOT NULL,
  machine_name         VARCHAR(64)  NOT NULL,
  component            VARCHAR(32)  NOT NULL,       -- bearing|steam_valve|heater|water_pump
  severity             VARCHAR(32)  NOT NULL,       -- failure | scheduled_maintenance
  run_start_ts         DATETIME(3)  NULL,
  failure_ts           DATETIME(3)  NULL,
  repair_ts            DATETIME(3)  NULL,
  run_hours_to_failure DOUBLE       NULL,
  seq_at_failure       INT          NULL,
  KEY idx_session (session_id),
  KEY idx_machine (machine_name),
  KEY idx_failure_ts (failure_ts),
  KEY idx_component (component)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

(Optional) a `generator_state` table to replace the Mongo checkpoint collection from
Plan 01 §13.2 — one row per machine, full serialized engine state + RNG as a `JSON`
or `LONGBLOB` column.

## A.2 SQLAlchemy models — `backend/app/models.py` (new)

```python
from __future__ import annotations
from datetime import datetime
from sqlalchemy import (BigInteger, Integer, String, DateTime, Double,
                        Boolean, JSON, UniqueConstraint, Index)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
    pass

class Reading(Base):
    __tablename__ = "machine_readings"
    id:           Mapped[int]      = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id:   Mapped[str]      = mapped_column(String(64))
    seq:          Mapped[int]      = mapped_column(Integer)
    machine_name: Mapped[str]      = mapped_column(String(64))
    state:        Mapped[str]      = mapped_column(String(16))
    ts:           Mapped[datetime] = mapped_column(DateTime)
    speed:        Mapped[float | None] = mapped_column(Double, nullable=True)
    vibration_rms: Mapped[float | None] = mapped_column(Double, nullable=True)
    # ... remaining columns from A.1 ...
    truth_json:   Mapped[dict | None]   = mapped_column(JSON, nullable=True)
    __table_args__ = (
        UniqueConstraint("session_id", "seq", name="uq_session_seq"),
        Index("idx_ts", "ts"),
        Index("idx_machine", "machine_name"),
    )

class MachineRun(Base):
    __tablename__ = "machine_runs"
    id:           Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id:   Mapped[str] = mapped_column(String(64))
    machine_name: Mapped[str] = mapped_column(String(64))
    component:    Mapped[str] = mapped_column(String(32))
    severity:     Mapped[str] = mapped_column(String(32))
    failure_ts:   Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    seq_at_failure: Mapped[int | None]    = mapped_column(Integer, nullable=True)
    # ... remaining columns ...
```

## A.3 Rewrite the access layer — `backend/app/database.py`

This file is the single chokepoint (`readings_collection`, `get_db`, the parsing
helpers). Replace the PyMongo client with a SQLAlchemy engine + session factory and
**keep the same helper function names** so the services barely change.

```python
"""MySQL access layer (SQLAlchemy + PyMySQL) + shared parsing helpers."""
from __future__ import annotations
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.config import DATABASE_URL          # mysql+pymysql://...

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=3600)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

def get_session():
    """FastAPI dependency / context manager for a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Parsing helpers (to_float, hms_to_minutes, parse_timestamp, map_status) STAY —
# but they now run at INGEST time (consumer) instead of on every read, because the
# columns are already typed. session_name_map() becomes a SQL query (A.5).
```

> **Keep** `to_float`, `hms_to_minutes`, `parse_timestamp`, `map_status` — but move
> their use to the **consumer/ingest** side. Once values are stored as `DOUBLE`/`INT`,
> the services read clean numbers and don't parse strings on every request.

## A.4 Rewrite the consumer — `rabbitmq/consumer/consumer.py`

Swap `MongoClient` + `insert_one` for a SQLAlchemy session. This is where the
string→typed conversion happens (using the existing helpers).

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.mysql import insert as mysql_insert
from models import Reading, MachineRun           # share the models package
from helpers import to_float, hms_to_minutes, parse_timestamp, map_status

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
Session = sessionmaker(bind=engine)

def write_reading(data: dict):
    plc, util, health, qual = (data.get(k, {}) for k in ("plc","utility","health","quality"))
    row = dict(
        session_id=data["session_id"], seq=data["seq"],
        machine_name=data.get("machine_name"), state=map_status(data.get("state")),
        ts=parse_timestamp(data.get("timestamp")),
        speed=to_float(plc.get("speed")), length=to_float(plc.get("length")),
        machine_time_min=hms_to_minutes(plc.get("machine_time")),
        sf_flow=to_float(util.get("SF_Flow")), em_power=to_float(util.get("EM_Power")),
        vibration_rms=to_float(health.get("vibration_rms")),
        bearing_temp=to_float(health.get("bearing_temp")),
        good_count=qual.get("good_count"), reject_count=qual.get("reject_count"),
        truth_json=data.get("_truth"),
        # ... rest of the columns ...
    )
    stmt = mysql_insert(Reading).values(**row)
    # idempotent on redelivery thanks to UNIQUE(session_id, seq):
    stmt = stmt.on_duplicate_key_update(state=stmt.inserted.state)
    with Session() as s:
        s.execute(stmt); s.commit()
```

For throughput during `backfill.py`, **batch**: collect N rows and use
`session.execute(insert(Reading), list_of_dicts)` (executemany) instead of one INSERT
per reading — the MySQL analogue of the current `insert_many`.

## A.5 Update the backend services

The five service files all reach into Mongo via `readings_collection()` /
aggregation. Convert each query to SQLAlchemy `select(...)`. Examples:

| Mongo today | MySQL (SQLAlchemy) |
|---|---|
| `col.distinct("session_id")` | `select(Reading.session_id).distinct()` |
| `col.find_one({"session_id": sid}, sort=[("seq",1)])` | `select(Reading).where(Reading.session_id==sid).order_by(Reading.seq).limit(1)` |
| latest window for a machine | `... .order_by(Reading.seq.desc()).limit(W)` |
| OEE state-duration counts | `select(Reading.state, func.count()).group_by(Reading.state)` |

`session_name_map()` (currently a per-session `find_one` loop) becomes one grouped query:

```python
from sqlalchemy import select, func
def session_name_map(db) -> dict[str, str]:
    rows = db.execute(
        select(Reading.session_id, func.min(Reading.ts).label("first_ts"))
        .group_by(Reading.session_id).order_by("first_ts")
    ).all()
    return {sid: f"Machine {i+1}" for i, (sid, _) in enumerate(rows)}
```

Files to touch (from the grep of Mongo usage):
`backend/app/services/records_service.py`, `utilities_service.py`,
`alerts_service.py`, `analytics_service.py`, `machine_service.py`.

## A.6 Config changes — `backend/app/config.py`

```python
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://smartai:password@localhost:3306/machine_telemetry?charset=utf8mb4",
)
# MONGO_URI / DB_NAME can stay temporarily for the one-off export, then be removed.
```

## A.7 The Plan 02 ML layer (`ml/dataset.py`)

Plan 02 §B.5 currently says "pull readings + events from Mongo." After migration it
pulls with one SQL join — which also **removes the leakage-prone manual join**:

```python
import pandas as pd
from sqlalchemy import create_engine
engine = create_engine(DATABASE_URL)
readings = pd.read_sql("SELECT * FROM machine_readings ORDER BY session_id, seq", engine)
runs     = pd.read_sql("SELECT * FROM machine_runs", engine)
# label join (B.4) is now a clean SQL/pandas merge on session_id + failure_ts >= ts
```

Drop `truth_json` and ids before building `FEATURE_COLUMNS` (the §B.3 rule still holds).

## A.8 One-time data export (move the surviving ~500k rows)

Don't lose what's already in Atlas. Stream-copy before cutover:

```python
# scripts/mongo_to_mysql.py  (run once)
from pymongo import MongoClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Reading
# read each Mongo doc, flatten via the same helpers as A.4, batch-insert to MySQL.
# 500k rows in batches of 1000 → ~500 round trips, minutes.
```

> Because MySQL stores typed columns (no repeated BSON field names per document), the
> same 500k readings will occupy **substantially less** space than they did in Atlas.

## A.9 Migration checklist

1. Stand up MySQL 8 (local Docker `mysql:8` or a sized instance); create the
   `machine_telemetry` schema.
2. Add deps (A.0); add `DATABASE_URL` to `.env` (A.6).
3. Add `models.py` (A.2); create tables (`Base.metadata.create_all(engine)` or Alembic).
4. Rewrite `database.py` (A.3) keeping helper names.
5. Rewrite `consumer.py` ingest (A.4) — string→typed at write time.
6. Convert the 5 services + `session_name_map` to `select(...)` (A.5).
7. Point `publisher.py` / `backfill.py` checkpoint at the `generator_state` table.
8. Export surviving Atlas rows (A.8).
9. Update `ml/dataset.py` to read via SQL (A.7).
10. Smoke-test: dashboard endpoints return; consumer writes; restart stays idempotent.

---

# PART B — Sampling rate for XGBoost: seconds vs minutes vs hours

**Question:** for Phase 2 Part 1 (the generator's `GEN_DT_SECONDS` cadence that feeds
the XGBoost model in Plan 02), should readings be spaced in **seconds, minutes, or
hours** — what's best for processing and memory?

## B.1 The trade-off in one table

For **one machine**, the row count scales directly with the cadence:

| Cadence (`dt`) | Rows / machine / day | Rows / machine / 30 days | Verdict |
|---|---|---|---|
| **1 / second** | 86,400 | ~2.6 M | Massive; mostly redundant for PM |
| **1 / 10 s** | 8,640 | ~259 k | Still heavy |
| **1 / minute** | 1,440 | ~43 k | **Sweet spot for PM** |
| **1 / 5 min** | 288 | ~8.6 k | Good for long horizons |
| **1 / hour** | 24 | 720 | Too coarse — misses lead-time detail |

With the fleet of `GEN_MACHINES` machines and weeks of `backfill.py`, multiply
accordingly. Second-resolution across several machines for weeks is exactly the volume
that overflowed the cluster (Part C).

## B.2 Why finer ≠ better here

The model's job (Plan 02 §B.1) is **next-imminent failure within H = 24 sim-hours**.
The physics in Plan 01 that the model keys off are **slow processes**:

- Gamma-process wear, Weibull lifetimes → degrade over **hours-to-days**.
- First-order **thermal lag** (`τ` minutes) → temps ramp over minutes, never jump.
- OU/AR(1) flow dynamics → autocorrelated; consecutive **per-second** rows are
  near-identical.

So second-level rows are **highly autocorrelated and nearly redundant** for a 24-hour
horizon. They inflate storage, RAM, and training time **without adding signal** — and
they worsen the leakage risk in §B.5 (consecutive near-identical rows make a random
split leak the future). The predictive tells (rising vibration slope, winding-temp
ramp) are perfectly visible at **minute** resolution.

## B.3 Processing & memory impact

- **Memory** — XGBoost `tree_method="hist"` holds the feature matrix in RAM. At 1/min
  vs 1/s you carry **~60× fewer rows** for the same time span — the difference between
  a few hundred MB and tens of GB for a multi-machine, multi-week dataset.
- **Training time** — roughly linear in rows for `hist`; 60× fewer rows ≈ 60× faster
  fits, which makes the Optuna tuning loop (§B.6) actually feasible.
- **Feature engineering** — the rolling windows in §B.3 (`W ∈ {10,30,60 readings, 1h,
  6h}`) are cheaper and the windows mean **more in clock time** at coarse cadence
  (60 readings = 1 hour at 1/min, vs 1 minute at 1/s).
- **Storage** — directly ties to Part C: minute cadence is the difference between
  filling the cluster and not.

## B.4 Recommendation

- **Use minute-level cadence as the default** — `GEN_DT_SECONDS = 60` for the model
  training / backfill data. It captures all PM-relevant dynamics at a fraction of the
  cost. For very long horizons you can go to **5 minutes**.
- **Don't store seconds** for training. If you ever want fast transients (e.g. a
  vibration spike), capture them as **aggregated features over the minute**
  (min/max/std within the minute), not as raw per-second rows.
- **Don't use hours** as the base cadence — 24 rows/day is too coarse to estimate the
  lead-time distribution (§B.7) that proves predictive value. Hours are fine as a
  **bucket size for OEE/aggregation**, not as the raw sampling rate.
- **Live demo** can still tick faster (every few seconds) for a responsive dashboard,
  but **down-sample to minute rows before they hit the training table** — keep the
  serving window aligned with the training cadence to avoid train/serve skew.

> Bottom line: **minutes.** Seconds = wasted memory/time + redundancy + leakage risk;
> hours = too coarse for lead-time. Minute resolution is the best
> processing/memory/signal balance for this model and horizon.

---

# PART C — Observed data loss: ~500k of 1.2M readings

## C.1 What happened

A full run was expected to produce **~1.2 million readings**, but only **~500,000**
were persisted to MongoDB **before the cluster filled up** and writes started failing.
The remaining ~700k were effectively dropped (the cluster hit its storage ceiling — a
free/shared Atlas tier caps at 512 MB).

## C.2 Why

- **Storage cap**, not a logic bug. The current consumer does an unbounded
  `insert_one` per reading with **no cap, no TTL, no rotation** — once Atlas hit its
  limit, inserts errored and `write_reading()` just logged and moved on (the data was
  lost, not retried).
- **Per-document overhead** — every BSON doc repeats all nested keys
  (`plc`, `utility`, `health`, `quality`, `_truth` …) as strings, so each reading is
  storage-heavy. 500k rich docs ≈ the 512 MB ceiling.
- **Second-level cadence** (Part B) multiplies the row count, so the ceiling arrives
  far sooner than it would at minute cadence.

## C.3 How the changes in this doc fix it

1. **MySQL (Part A)** removes the 512 MB wall and stores the **same data more
   compactly** (typed columns, no repeated field names) → the full 1.2 M fits with
   room to spare.
2. **Minute cadence (Part B)** cuts the row count by ~60× vs seconds → the dataset is
   a fraction of the size for the same simulated time span.
3. **Idempotent ingest** (the `UNIQUE(session_id, seq)` + `ON DUPLICATE KEY UPDATE` in
   A.4) means a restart after a failure **resumes without duplicating or losing** rows.
4. **Backfill batching** (A.4) makes re-generating the missing ~700k fast once MySQL is
   in place.

## C.4 Action

- Treat the surviving ~500k as **disposable bootstrap data** — export it (A.8) so it
  isn't lost, but plan to **re-run `backfill.py` against MySQL** at minute cadence to
  get a clean, complete 1.2 M-reading dataset for Plan 02.
- Before training, re-check the **per-class failure counts** (Plan 01 §18 / Plan 02
  §B.10) on the new MySQL dataset — the truncated 500k run may be missing whole
  failure cycles for some components, which would silently bias the model.
```
