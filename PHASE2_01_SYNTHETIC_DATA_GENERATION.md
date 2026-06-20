# Phase 2 — Plan 01: Synthetic Run-to-Failure Data Generation

> **Goal of this document.** A complete, executable methodology for replacing the
> current steady-state generator with a physically-plausible, **multi-machine,
> run-to-failure** simulator whose telemetry **accurately mimics real industrial
> machinery** and streams **continuously and naturally** (no "seams"), while
> producing ground-truth maintenance labels and everything OEE + the XGBoost model
> need.
>
> **Methodology decisions (locked):** calibration = **pure domain defaults**;
> method rigor = **full stochastic** (Gamma-process wear, Ornstein–Uhlenbeck/AR(1)
> noise, first-order thermal lag, Weibull lifetimes, Poisson shocks, physics-
> consistent integration); data realism = **configurable imperfections**.
>
> Companion docs: `PHASE2_02_MODEL_TRAINING_EVALUATION.md`,
> `PHASE2_03_ML_LIFECYCLE.md`.

---

## 1. Why this is needed

Phase 1 built a working live pipeline:

```
machine_data_generator.py  →  publisher.py  →  RabbitMQ (exchange "scada_data")
        →  consumer.py  →  MongoDB (machine_telemetry.machine_readings)
        →  FastAPI backend  →  React dashboard
```

But the **data** is useless for predictive maintenance:

| Problem in current generator                  | Consequence                                  |
|-----------------------------------------------|----------------------------------------------|
| Every reading is `state="running"`             | No downtime → Availability can't be computed |
| `speed` constant at 25.0                        | No performance variation                     |
| `EM_Power`/`EM_Energy` always `null`            | No load/energy signal                        |
| No vibration/temperature/current/pressure       | None of the classic PM signals exist         |
| No defect/quality data                          | OEE Quality can't be computed                |
| **Nothing degrades, nothing ever fails**        | **No target to predict — the model is impossible** |
| i.i.d. noise, no continuity model               | Looks synthetic; restarts reset everything   |

This plan fixes the root cause: a machine that **wears down and fails in different
ways**, emitting the sensors + maintenance log a real plant has, and **streaming
believably forever**.

**Guiding principle (how real PM data works):** a **hidden degradation state**
drives **observable sensors**; **failure** is a threshold crossing; **maintenance**
resets it. The downstream model only ever sees observable sensors + a maintenance
log — never the hidden health. We mirror that structure (as in NASA C-MAPSS,
AI4I-2020, Azure PdM).

---

## 2. Architecture — what changes, what stays

**Stays:** RabbitMQ topology, the publish/consume transport, MongoDB as the sink,
and the `generate_one()` calling convention used by `publisher.py`.

**Changes:**
1. Rewrite the *brain* of `machine_data_generator.py` as a **stochastic degradation
   engine** (Sections 3–10).
2. Extend the reading schema **additively** (Section 11) — frontend keeps working.
3. **Multi-machine** driver + a failure/maintenance **event stream** → populate the
   empty `machine_runs` collection (Section 12).
4. **Checkpoint/resume** + **two-clock** design for natural continuous streaming
   (Section 13).
5. Add `backfill.py` for fast historical bootstrap; keep the live publisher (Section 14).

---

## 3. Calibration — domain-default baselines (no real dataset assumed)

Healthy operating points + alarm limits from domain knowledge of an industrial
fabric dyeing/finishing range (~22 kW LV motor, 400 V 3-phase, pf ≈ 0.85, 6-bar air):

| Signal           | Healthy baseline    | Alarm / limit          | Basis                              |
|------------------|---------------------|------------------------|------------------------------------|
| `speed`          | ~25 m/min           | —                      | existing PLC value                 |
| `vibration_rms`  | 1.5–2.5 mm/s        | >4.5 unsat, >7.1 fail  | **ISO 10816** medium-machine zones |
| `motor_current`  | 25–35 A             | ~40 A FLC              | ~22 kW @ 400 V, pf 0.85            |
| `bearing_temp`   | 40–60 °C            | >85 °C                 | rolling-bearing norms              |
| `winding_temp`   | 60–85 °C            | >130 °C (class F)      | motor insulation class             |
| `air_pressure`   | 5.8–6.2 bar         | <5.0                   | shop compressed-air                |
| `SF_Flow`/`Tot`  | ~2.8 / accumulates  | —                      | existing PLC values                |
| `Wat_Flow`/`Tot` | ~12.2 / accumulates | —                      | existing PLC values                |
| `EM_Power`       | derived (Section 6) | —                      | √3·V·I·pf                          |
| `ambient`        | 25–35 °C (diurnal)  | —                      | plant environment                  |

Each machine instance draws its own setpoints from these ranges (Section 10).

---

## 4. Hidden degradation model (per component)

Four wearing components, each with a hidden `health ∈ [0,100]` and a **distinct,
separable** sensor fingerprint (so the multi-class model can tell them apart):

| Component       | Failure class | Primary signal as it wears               | Secondary effects                              |
|-----------------|---------------|------------------------------------------|------------------------------------------------|
| Bearing/motor   | `bearing`     | `vibration_rms` ↑ (accelerating)         | `bearing_temp` ↑, `motor_current` ↑, `speed` ↓ |
| Steam valve     | `steam_valve` | `SF_Flow` drifts below setpoint, erratic   | `SF_Tot` rate ↓, **reject_rate ↑**             |
| Heating element | `heater`      | `winding_temp` ↑                          | `EM_Power`/current ↑, **reject_rate ↑**        |
| Water pump      | `water_pump`  | `Wat_Flow` ↓, erratic                     | `air_pressure` ↓, `motor_current` ↑            |

### 4.1 Gamma-process cumulative wear
Each component accumulates **monotone** damage `D` via a **Gamma process** (the
standard reliability model for monotone degradation):

```
ΔD ~ Gamma(shape = α · stress · dt,  scale = β)      # non-negative, irregular increments
D(t) = D(t−1) + ΔD
health = 100 · (1 − clamp(D / D_fail, 0, 1))
```

### 4.2 Weibull component lifetimes (wear-out hazard)
Per instance, sample a target life `L ~ Weibull(k>1, λ)` (increasing hazard =
wear-out) and scale `α`/`D_fail` so the **mean** first-passage of `D` to `D_fail`
matches `L`. Failure *times* then follow the canonical Weibull distribution.

### 4.3 Poisson shocks (sudden damage)
Superimpose sudden damage events `N ~ Poisson(λ_shock · dt)`; each adds a jump to
`D` (bearing spall, valve blockage). Gradual wear **+** shocks = realistic mixed
failure modes.

### 4.4 Covariate (stress) acceleration
The Gamma drift scales with operating stress — an accelerated-degradation flavor:

```
stress = (speed/nominal_speed) · (1 + k_temp·(ambient−25)/10) · duty_factor
```

Pushing the machine harder, hotter, longer wears it faster.

---

## 5. Sensor synthesis pipeline

Every observable sensor is built as:

```
signal(t) = baseline(state, article)
          + degradation_transfer(health)     # monotone, convex: shifts mean AND inflates variance
          + dynamics(t)                       # temporal correlation (5.1 / 5.2)
          + measurement_noise                 # small Gaussian sensor white-noise
```

### 5.1 Ornstein–Uhlenbeck / AR(1) dynamics (flows, currents)
Mean-reverting wander around the setpoint — **smooth and autocorrelated**, not i.i.d.
jitter (the single biggest "looks natural" lever):

```
x(t) = x(t−1) + θ · (μ − x(t−1)) · dt + σ · sqrt(dt) · ε,   ε ~ N(0,1)
```

### 5.2 First-order thermal lag (temperatures)
Thermal mass ⇒ `bearing_temp`/`winding_temp` **ramp** toward a load/health-driven
target and **lag** load changes; they never jump:

```
T(t) = T(t−1) + (T_target(load, health) − T(t−1)) · dt / τ
```

### 5.3 Degradation transfer + variance inflation
Monotone convex maps from `degradation_c = (1 − health_c/100)` to each sensor's
**mean shift** and **σ growth** (e.g. as the valve wears, `SF_Flow` mean sags *and*
its variance rises → erratic). Rising **std** is itself a predictive tell.

---

## 6. Physics-consistent coupling (internal consistency)

Enforce the identities a real plant obeys, so **no row is physically impossible**
and cross-sensor structure is genuine:

```
EM_Power  = sqrt(3) · V · I · pf                         # power from current/voltage/pf
EM_Energy += EM_Power · dt                                # energy = ∫ power
SF_Tot    += SF_Flow · dt   ;  Wat_Tot += Wat_Flow · dt  # totalizers = ∫ flow
length    += speed · dt      ;  machine_time += dt        # while running
good, reject ~ Binomial(units_in_window, reject_rate)    # reject_rate from valve+heater wear
```

---

## 7. Operating context — semi-Markov states + production calendar

### 7.1 Semi-Markov state machine
States `{running, idle, changeover, maintenance, error}` with **sojourn-time
distributions** (lognormal/Weibull dwell), *not* fixed durations ⇒ realistically
irregular stops. Transitions gated by the calendar, lot completion, and failures.

```
            lot done / break                  health_c < fail_threshold
 running ───────────────────► idle ─┐        ┌──────────────────────────► error
    ▲                                │        │                              │
    │        break over             ▼        │   downtime ~ lognormal        ▼
    └───────────────── changeover ◄──┘        └─ maintenance (imperfect repair) ◄─┘
```

- **idle / changeover** = planned stops (excluded from planned production time).
- **error** = unplanned downtime (Availability loss) → emits a failure event.
- **maintenance** = repair window → emits a repair event → back to `running`.

### 7.2 Production calendar
Shifts (e.g. 2×8 h), breaks, weekends, occasional holidays ⇒ diurnal + weekly
**utilization seasonality** — the rhythm that makes long streams read as real.

### 7.3 Lot / batch lifecycle
Each lot has an article (different setpoints) + a length target; lot ends →
changeover → next lot; a new campaign starts a new `session_id`.

---

## 8. Failure → repair cycle

- **Failure** = first component whose `D` crosses `D_fail` (first-passage) →
  `state=error`, event logged to `machine_runs`.
- **Imperfect repair (Kijima model)** — maintenance restores the failed component to
  ~95–100% (not always perfect) and may slightly age others ⇒ realistic recurrence,
  not a clean "as-good-as-new" every time.
- **Downtime** duration `~ lognormal` (repair-time variability).

---

## 9. Imperfection layer (configurable data-quality artifacts)

Real telemetry is messy; all of these sit behind config flags so you can train
clean first, then harden:

- **Dropouts** — with prob `p_drop`, emit `null` (like `EM_Power` already is);
  bursty via a 2-state Markov for realistic gaps.
- **Outliers/spikes** — rare large transient on a single sensor.
- **Stuck sensor** — small prob a sensor freezes at its last value for a few reads.
- **Timestamp jitter** — emit time ± a few hundred ms.

---

## 10. Per-machine heterogeneity + reproducibility

Sample each machine's Weibull/Gamma params, setpoints, noise σ, and thermal τ from
the Section 3 ranges (manufacturing variation / "lemons"). A **master seed** with
per-machine sub-seeds ⇒ fully reproducible simulations. The RNG state is part of the
checkpoint (Section 13) so live runs stay reproducible across restarts.

---

## 11. Extended document schema (additive — frontend unaffected)

Keep every existing `plc`/`utility` field. Add `health` + `quality`, populate
`EM_Power`/`EM_Energy`, let `state` vary. Hidden truth lives under `_truth` (dev/debug
only; **the feature pipeline must exclude it** — see Plan 02).

```jsonc
{
  "session_id": "…", "seq": 1234, "machine_name": "Machine 1",
  "state": "running",                         // running|idle|changeover|maintenance|error
  "timestamp": "2026-06-20T10:00:00+00:00",
  "plc": {
    "lot_1": 2923740, "lot_2": 2923741, "article": 5896,
    "speed": "24.81", "length": "318.44", "runmemory": true,
    "lot_time": "1:02:11", "machine_time": "7051:33:23",
    "steam_consumed_lot": "…", "water_consumed_lot": "…",
    "power_consumed_lot": 0, "air_consumed_lot": 0
  },
  "utility": {
    "SF_Flow": "2.71", "SF_Tot": "39110.221",
    "Wat_Flow": "12.10", "Wat_Tot": "171476.880",
    "EM_Power": 41.7, "EM_Energy": 138204.5          // NOW POPULATED (kW / kWh)
  },
  "health": {                                         // NEW — PM sensors
    "vibration_rms": 2.41,   // mm/s
    "motor_current": 14.3,   // A
    "bearing_temp": 58.7,    // °C
    "winding_temp": 71.2,    // °C
    "air_pressure": 6.1      // bar
  },
  "quality": { "good_count": 196, "reject_count": 4 },// NEW — feeds OEE Quality
  "_truth": {                                          // NEW — hidden; EXCLUDE from features
    "health": { "bearing": 71.0, "steam_valve": 88.2, "heater": 95.1, "water_pump": 60.4 },
    "active_fault": null, "rul_hours": 41.5, "ambient": 28.4
  }
}
```

---

## 12. `machine_runs` event documents (the simulated maintenance log)

One document per run-to-failure cycle — the **source of truth for labelling**:

```jsonc
{
  "session_id": "…", "machine_name": "Machine 1",
  "component": "bearing",            // which component failed
  "severity": "failure",             // failure | scheduled_maintenance
  "run_start_ts": "…", "failure_ts": "…", "repair_ts": "…",
  "run_hours_to_failure": 312.5, "seq_at_failure": 9981
}
```

---

## 13. Continuous / live structure (no seams over weeks)

The answer to "generation is not one-time — keep it natural forever."

### 13.1 One engine, two clocks
A `Clock` abstraction. **Backfill** = sim-clock, advance `dt` fast, no sleep.
**Live** = wall-clock, advance by real elapsed time, real timestamps. Identical
physics ⇒ history and live tail are statistically indistinguishable.

### 13.2 Checkpoint / resume — THE key to no seams
Persist the generator's **full** state — per-component `D`/health, all totalizers,
`EM_Energy`, `machine_time`, `length`, current lot/article, state-machine state +
remaining sojourn, `seq`, **and the RNG state** — to a `generator_state` collection
every `CHECKPOINT_EVERY` readings. On startup, **resume** so totalizers/health/time
*continue*. Without this, a restart resets everything and instantly looks fake.

### 13.3 Seamless backfill → live handoff
Backfill produces history ending at "now" and saves its final state; live boots from
that same state ⇒ contiguous `seq`/timestamps, continuous totalizers/health. No
discontinuity at the boundary.

### 13.4 Fleet staggering & session lifecycle
N machines with **phase-staggered** lifecycles (different lot offsets, component
ages) so failures/repairs spread over time, not synchronized ⇒ organic fleet. New
`session_id` at campaign/major-maintenance boundaries (totalizers reset per run).

### 13.5 Anti-periodicity (kill the tells)
Jitter shift boundaries ± minutes, dwell times from distributions, randomized
article order, and a **slow OU drift on setpoints** (calibration drift). Nothing
repeats exactly ⇒ no spottable period.

### 13.6 Long-run invariants
- **Monotone**: `machine_time`, totalizers (within session), `EM_Energy`.
- **Sawtooth**: component health (decay → repair → decay).
- **Bounded-but-wandering**: flows/temps/currents around evolving setpoints.
- **Rare**: failures (Weibull-spaced), dropouts.

---

## 14. Data-volume strategy (two run modes, one engine)

Multi-class training needs **many failures of each type**; a live 1-reading/few-sec
stream alone would take months. Same engine:

1. **`backfill.py` (bootstrap)** — sim-clock, bulk-insert directly to Mongo;
   *weeks–months* across *many* machines ⇒ ≥ a few hundred examples per class. Knobs:
   machines, sim weeks, `dt`, `GEN_ACCEL` (accelerated wear for bootstrap).
2. **Live stream (demo)** — existing `publisher.py` path, real-time.

---

## 15. Module design / file-by-file

### 15.1 `rabbitmq/producer/publisher/machine_data_generator.py` (rewrite brain)
Keep the public surface (`SyntheticMachineGenerator`, `generate_one()`); decompose:

- `Clock` (sim | wall) — §13.1
- Process helpers: `GammaWearProcess`, `OUProcess`, `ThermalLag`, `ShockProcess` — §4/§5
- `ComponentState` — health/`D`, Weibull/Gamma params, threshold — §4
- `SensorModel` — baseline + degradation transfer + dynamics + noise + physics coupling — §3/§5/§6
- `SemiMarkovStateMachine` — states + sojourn distributions — §7.1
- `ProductionCalendar` — shifts/breaks/weekends — §7.2
- `ImperfectionLayer` — dropouts/outliers/stuck/jitter — §9
- `SyntheticMachineGenerator` — orchestrates `generate_one()`; `serialize()/restore()`
  for checkpoint (§13.2); `pop_events()`

### 15.2 `rabbitmq/producer/publisher/backfill.py` (new)
Loop sim-clock across N machines; buffer readings → `insert_many` (batched); collect
`pop_events()` → `machine_runs`; create indexes `(session_id, seq)`, `timestamp`;
print per-class failure counts; save final checkpoint for handoff.

### 15.3 `publisher.py` (extend)
N generators (multi-machine), wall-clock; publish telemetry on `scada.tag.data` and
events on a second routing key `scada.machine.event`; checkpoint periodically;
graceful shutdown → final checkpoint.

### 15.4 `consumer.py` (extend)
Telemetry → `machine_readings` (unchanged); add a handler bound to
`scada.machine.event` → `machine_runs`; ensure indexes on both.

---

## 16. Configuration (env)

`GEN_MACHINES`, `GEN_SEED`, `GEN_CLOCK` (sim|wall), `GEN_LIVE_INTERVAL`,
`GEN_SIM_WEEKS`, `GEN_DT_SECONDS`, `GEN_ACCEL`; per-component `WEIBULL_K/LAMBDA`,
`GAMMA_ALPHA/BETA`, `SHOCK_LAMBDA`, `FAIL_THRESHOLD`; per-sensor `OU_THETA/SIGMA`,
`THERMAL_TAU`, `NOISE_SIGMA`; calendar `SHIFTS/BREAKS/WEEKEND`; imperfections
`P_DROP/P_OUTLIER/P_STUCK/TS_JITTER`; `CHECKPOINT_EVERY`. Section 3 domain ranges ship
as defaults. (Reuse `MONGODB_URI`/`MONGODB_DB` from `backend/app/config.py`.)

---

## 17. Step-by-step execution

1. Snapshot the current `machine_readings` count (it's disposable dummy data).
2. Implement §15.1 engine; sanity-run one machine standalone: health decays then
   resets on repair, sensors track health, events emit at failure, physics
   identities hold.
3. Implement §15.2 `backfill.py`. (Confirm with the user before any wipe of
   `machine_readings`/`machine_runs` — see Plan 03 for a safe wipe.)
4. Run backfill (env per §16); watch the per-class failure summary.
5. Implement §15.3/§15.4; run `consumer.py` then `publisher.py`; confirm live
   readings + events land in both collections and a restart resumes cleanly.
6. Run the §18 verification.

---

## 18. Verification & acceptance

**Structural (from the dataset):**
- `machine_readings.distinct("state")` returns all five states.
- `utility.EM_Power == null` is ~0 during `running` (aside from configured dropouts).
- `machine_runs` grouped by `component` shows every class with ≥ a few hundred events.
- `quality.reject_count > 0` exists and correlates with degraded valve/heater runs.

**Realism (the methodology checks):**
- **Smoothness** — autocorrelation of `SF_Flow`/`Wat_Flow` clearly > 0 (OU working).
- **Thermal lag** — temps ramp and lag step changes in load (no instant jumps).
- **Physics identities** — `EM_Energy ≈ Σ EM_Power·dt`, `length ≈ Σ speed·dt`,
  totalizers ≈ ∫ flow (within noise).
- **Reliability** — failure inter-arrival times fit a Weibull; `vibration_rms` sits
  in ISO-10816 bands and rises before `bearing` failures (slope > 0 in the lead-up).
- **Signal precedes failure** — for several events per class, the component's primary
  signal is clearly trending before `seq_at_failure`.

**Continuity:**
- **Restart test** — kill mid-run, restart → totalizers/health/`machine_time`
  continue with **no jump** (checkpoint/resume works).
- **Handoff test** — no seam at the backfill→live boundary (continuous seq/totalizers).
- **Imperfections** — dropouts/outliers/stuck/jitter appear at configured rates.

**Compatibility:**
- Existing dashboard endpoints still return without error (additive schema).

**Acceptance:** all of the above hold, and the dataset is large/varied enough for
Plan 02 to train a multi-class model with real per-class signal.

---

## 19. Pitfalls to avoid

- **Leakage** — never let `_truth.*` reach the feature pipeline; keep it namespaced
  and exclude it explicitly in Plan 02.
- **Too-clean signal** — if sensors track health perfectly the model hits 100% and
  looks fake; keep OU noise, variance inflation, imperfections, and per-machine
  variability.
- **Seams** — skipping checkpoint/resume makes every restart reset totalizers/health
  (the #1 tell of a simulation). Persist full state incl. RNG.
- **Periodicity tells** — fixed shift/dwell durations look robotic; jitter everything
  (§13.5).
- **Totalizer/time bugs** — resets and monotonicity are easy to get wrong and quietly
  poison totalizer-rate features.
- **Class imbalance baked too hard** — ensure each component fails enough; check the
  §18 per-class counts before training.
- **Don't break the contract** — `publisher.py` still calls `generate_one()`; the
  schema stays additive so the live dashboard keeps working throughout.
