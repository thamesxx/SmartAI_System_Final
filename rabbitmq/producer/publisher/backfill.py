"""
backfill.py — Fast historical data bootstrap for Phase 2.

Runs the stochastic engine in sim-clock mode (no sleep), bulk-inserts
readings into machine_readings and events into machine_runs, then
saves a checkpoint in generator_state so the live publisher can pick
up seamlessly from where the backfill ended.

Usage:
  python backfill.py

Key env vars (all have defaults):
  MONGODB_URI, MONGODB_DB
  GEN_MACHINES   — number of machines (default 5)
  GEN_SEED       — master RNG seed (default 42)
  GEN_SIM_WEEKS  — sim weeks to generate (default 12)
  GEN_DT_SECONDS — sim step size in seconds (default 30)
  GEN_ACCEL      — wear acceleration multiplier (default 3.0)
  GEN_BATCH_SIZE — readings per insert_many call (default 500)
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING

from machine_data_generator import SyntheticMachineGenerator

load_dotenv()

# ─── Config ────────────────────────────────────────────────────────────────
MONGO_URI   = os.getenv("MONGODB_URI", os.getenv("MONGO_URI", "mongodb://localhost:27017"))
MONGO_DB    = os.getenv("MONGODB_DB",  os.getenv("MONGO_DB",  "machine_telemetry"))
N_MACHINES  = int(os.getenv("GEN_MACHINES",   "5"))
MASTER_SEED = int(os.getenv("GEN_SEED",        "42"))
SIM_WEEKS   = int(os.getenv("GEN_SIM_WEEKS",   "12"))
DT          = float(os.getenv("GEN_DT_SECONDS","30"))
ACCEL       = float(os.getenv("GEN_ACCEL",     "3.0"))
BATCH_SIZE  = int(os.getenv("GEN_BATCH_SIZE",  "500"))

# ─── MongoDB setup ─────────────────────────────────────────────────────────
client = MongoClient(MONGO_URI)
db     = client[MONGO_DB]
readings_col      = db["machine_readings"]
runs_col          = db["machine_runs"]
checkpoint_col    = db["generator_state"]

def _ensure_indexes() -> None:
    readings_col.create_index([("session_id", ASCENDING), ("seq", ASCENDING)])
    readings_col.create_index([("timestamp", ASCENDING)])
    readings_col.create_index([("machine_name", ASCENDING)])
    runs_col.create_index([("session_id", ASCENDING)])
    runs_col.create_index([("machine_name", ASCENDING)])
    runs_col.create_index([("failure_ts",  ASCENDING)])
    runs_col.create_index([("component",   ASCENDING)])
    checkpoint_col.create_index([("machine_id", ASCENDING)], unique=True)
    print("Indexes ensured.")

# ─── Helpers ───────────────────────────────────────────────────────────────

def _save_checkpoint(gen: SyntheticMachineGenerator) -> None:
    state = gen.serialize()
    checkpoint_col.replace_one(
        {"machine_id": gen.machine_id},
        {"machine_id": gen.machine_id, "state": state, "saved_at": datetime.now(timezone.utc).isoformat()},
        upsert=True,
    )

# ─── Main backfill loop ────────────────────────────────────────────────────

def run_backfill() -> None:
    _ensure_indexes()

    sim_end   = datetime.now(timezone.utc)
    sim_start = sim_end - timedelta(weeks=SIM_WEEKS)

    print(f"\n{'='*60}")
    print(f"BACKFILL  machines={N_MACHINES}  weeks={SIM_WEEKS}  dt={DT}s  accel={ACCEL}x")
    print(f"Sim range: {sim_start.date()} → {sim_end.date()}")
    ticks_per_machine = int(SIM_WEEKS * 7 * 24 * 3600 / DT)
    print(f"Ticks/machine: {ticks_per_machine:,}   (~{ticks_per_machine*N_MACHINES/1e6:.2f}M total readings)")
    print(f"{'='*60}\n")

    # ── Stagger machine starts so their component ages differ
    stagger_ticks_options = [0, ticks_per_machine // 7, ticks_per_machine // 5,
                              ticks_per_machine // 4, ticks_per_machine // 3]

    total_readings = 0
    total_events   = 0
    component_counts: Counter = Counter()

    wall_start = time.time()

    for mid in range(N_MACHINES):
        machine_seed = MASTER_SEED * 1000 + mid
        # Phase-stagger: each machine starts at a different point in its lifecycle
        stagger = stagger_ticks_options[mid % len(stagger_ticks_options)]
        # Adjust sim_start so machine has already been running for stagger*dt seconds
        machine_sim_start = sim_start - timedelta(seconds=stagger * DT)

        gen = SyntheticMachineGenerator(
            machine_id=mid,
            seed=machine_seed,
            accel=ACCEL,
            dt=DT,
            sim_start_ts=machine_sim_start,
        )

        print(f"  Machine {mid+1}/{N_MACHINES} — seed={machine_seed}  stagger={stagger} ticks")

        readings_buf: list[dict] = []
        events_buf:   list[dict] = []
        m_readings = 0
        m_events   = 0

        # Fast-forward through stagger ticks (discard output — just age components)
        for _ in range(stagger):
            gen.generate_one()
            gen.pop_events()  # discard pre-history events

        # Reset session so history starts cleanly at sim_start
        gen.sim_ts = sim_start
        gen.session_id = __import__("uuid").uuid4().__str__()
        gen.seq = 0
        gen.steam_total = 0.0
        gen.water_total = 0.0
        gen.em_energy_kwh = 0.0
        gen.length = 0.0
        gen.lot_time_s = 0.0
        gen._good_count = 0
        gen._reject_count = 0
        gen._current_run_start_ts = sim_start.isoformat()
        gen._current_run_start_machine_time = gen.machine_time_s
        gen._current_run_start_seq = 0

        for tick in range(ticks_per_machine):
            reading = gen.generate_one()
            readings_buf.append(reading)
            m_readings += 1

            evts = gen.pop_events()
            for evt in evts:
                events_buf.append(evt)
                component_counts[evt["component"]] += 1
                m_events += 1

            # Flush readings buffer
            if len(readings_buf) >= BATCH_SIZE:
                readings_col.insert_many(readings_buf, ordered=False)
                total_readings += len(readings_buf)
                readings_buf.clear()

            # Flush events buffer
            if events_buf:
                runs_col.insert_many(events_buf, ordered=False)
                total_events += len(events_buf)
                events_buf.clear()

            if tick % 50_000 == 0 and tick > 0:
                pct = tick / ticks_per_machine * 100
                elapsed = time.time() - wall_start
                rate = total_readings / elapsed if elapsed > 0 else 0
                print(f"    [{pct:5.1f}%] {total_readings:,} readings  {total_events} events  "
                      f"{rate:,.0f} readings/s")

        # Flush remainder
        if readings_buf:
            readings_col.insert_many(readings_buf, ordered=False)
            total_readings += len(readings_buf)
        if events_buf:
            runs_col.insert_many(events_buf, ordered=False)
            total_events += len(events_buf)

        # Save checkpoint for live handoff
        _save_checkpoint(gen)

        print(f"    Done: {m_readings:,} readings, {m_events} events")
        print()

    elapsed = time.time() - wall_start
    print(f"{'='*60}")
    print(f"BACKFILL COMPLETE in {elapsed:.1f}s")
    print(f"  Total readings : {total_readings:,}")
    print(f"  Total events   : {total_events}")
    print(f"  Generation rate: {total_readings/elapsed:,.0f} readings/s")
    print(f"\n  Failure counts by component:")
    for comp, count in sorted(component_counts.items()):
        print(f"    {comp:15s}: {count:4d} events")
    print(f"{'='*60}\n")

    client.close()


if __name__ == "__main__":
    run_backfill()
