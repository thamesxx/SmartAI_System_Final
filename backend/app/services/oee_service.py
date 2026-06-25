"""OEE service — real pillars from machine_readings (MySQL / SQLAlchemy 2.0).

OEE = Availability × Performance × Quality

  Availability = running_time / (running + error + maintenance)
                 i.e. run-time fraction of planned production time.
                 idle/changeover are planned stops — excluded from denominator.

  Performance  = mean(speed | state=running) / NOMINAL_SPEED  [clamped 0-1]

  Quality      = Σ good_count / (Σ good_count + Σ reject_count)

OEE for a bucket = A × P × Q  (reported as %, each pillar also returned).
"""
from __future__ import annotations

from datetime import timedelta

from sqlalchemy import case, func, select

import ml.config as ml_cfg
from app.database import get_session, map_status, session_name_map
from app.models import Reading

NOMINAL_SPEED = ml_cfg.NOMINAL_SPEED


def _oee_from_agg(running, downtime, total, avg_run_speed, good, reject) -> dict:
    """Compute OEE pillars from pre-aggregated counts."""
    prod_time = (running or 0) + (downtime or 0)
    if prod_time == 0:
        return None  # nothing to compute — skip this bucket

    availability  = (running or 0) / prod_time
    run_spd       = float(avg_run_speed or 0)
    performance   = min(run_spd / NOMINAL_SPEED, 1.0) if NOMINAL_SPEED > 0 else 0.0
    g, r          = float(good or 0), float(reject or 0)
    quality       = g / (g + r) if (g + r) > 0 else 1.0

    oee = round(availability * performance * quality * 100, 2)
    return {
        "availability": round(availability * 100, 2),
        "performance":  round(performance  * 100, 2),
        "quality":      round(quality      * 100, 2),
        "oee":          oee,
    }


# ── Snapshot (whole history per machine) ─────────────────────────────────────

def get_oee_snapshot() -> list[dict]:
    """Return one OEE summary row per physical machine over its full history."""
    names = session_name_map()

    with get_session() as session:
        rows = session.execute(
            select(
                Reading.machine_name,
                func.sum(case((Reading.state == "running", 1), else_=0)).label("running"),
                func.sum(
                    case((Reading.state.in_(["error", "maintenance"]), 1), else_=0)
                ).label("downtime"),
                func.count().label("total"),
                func.avg(
                    case((Reading.state == "running", Reading.speed), else_=None)
                ).label("avg_run_speed"),
                func.sum(Reading.good_count).label("good"),
                func.sum(Reading.reject_count).label("reject"),
            )
            .group_by(Reading.machine_name)
            .order_by(Reading.machine_name)
        ).all()

    out = []
    for row in rows:
        pillars = _oee_from_agg(
            row.running, row.downtime, row.total,
            row.avg_run_speed, row.good, row.reject,
        )
        if pillars is None:
            continue
        out.append({
            "machine_id":   row.machine_name,
            "machine_name": names.get(row.machine_name, row.machine_name),
            **pillars,
        })
    return out


# ── Timeseries (per bucket) ──────────────────────────────────────────────────

_BUCKET_FMTS = {
    "shift": "%Y-%m-%d %H:00",
    "day":   "%Y-%m-%d %H:00",
    "week":  "%Y-%m-%d",
    "month": "%Y-%m-%d",
}


def get_oee_timeseries(
    machine: str | None = None,
    range: str = "shift",
) -> list[dict]:
    """Return OEE bucketed over time for charting.

    Parameters
    ----------
    machine : machine_name filter (None → all machines).
    range   : 'shift' | 'day' | 'week' | 'month'.
    """
    fmt = _BUCKET_FMTS.get(range, "%Y-%m-%d %H:00")
    names = session_name_map()

    stmt = select(
        Reading.machine_name,
        func.date_format(Reading.ts, fmt).label("bucket"),
        func.sum(case((Reading.state == "running", 1), else_=0)).label("running"),
        func.sum(
            case((Reading.state.in_(["error", "maintenance"]), 1), else_=0)
        ).label("downtime"),
        func.count().label("total"),
        func.avg(
            case((Reading.state == "running", Reading.speed), else_=None)
        ).label("avg_run_speed"),
        func.sum(Reading.good_count).label("good"),
        func.sum(Reading.reject_count).label("reject"),
    )

    if machine:
        stmt = stmt.where(Reading.machine_name == machine)

    stmt = stmt.group_by(
        Reading.machine_name,
        func.date_format(Reading.ts, fmt),
    ).order_by(Reading.machine_name, "bucket")

    with get_session() as session:
        rows = session.execute(stmt).all()

    out = []
    for row in rows:
        pillars = _oee_from_agg(
            row.running, row.downtime, row.total,
            row.avg_run_speed, row.good, row.reject,
        )
        if pillars is None:
            continue
        out.append({
            "machine_name": names.get(row.machine_name, row.machine_name),
            "time":         row.bucket,
            **pillars,
        })
    return out
