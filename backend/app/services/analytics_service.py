"""Analytics + OEE, derived from machine_readings (MySQL / SQLAlchemy 2.0)."""
from sqlalchemy import case, func, select

from app.database import get_session, session_name_map, to_float, map_status
from app.models import Reading


# ── Speed vs Lot length (grouped by lot) ─────────────────────────────────────

def get_lot_analytics():
    with get_session() as session:
        rows = session.execute(
            select(
                Reading.lot_1,
                func.avg(Reading.speed).label("avg_speed"),
                func.max(Reading.length).label("max_length"),
            )
            .where(Reading.lot_1.is_not(None))
            .group_by(Reading.lot_1)
        ).all()

    return [
        {
            "lot": str(row.lot_1),
            "speed": round(to_float(row.avg_speed), 1),
            "totalLength": round(to_float(row.max_length), 1),
        }
        for row in rows
    ]


# Legacy alias kept for /analytics/temperature route.
def get_temperature_analytics():
    return get_lot_analytics()


# ── Production rate vs target (per hour bucket) ───────────────────────────────

def get_production_analytics():
    with get_session() as session:
        rows = session.execute(
            select(
                func.date_format(Reading.ts, "%m-%d %H:00").label("hour"),
                func.min(Reading.length).label("min_len"),
                func.max(Reading.length).label("max_len"),
            )
            .where(Reading.ts.is_not(None))
            .group_by(func.date_format(Reading.ts, "%m-%d %H:00"))
            .order_by(func.date_format(Reading.ts, "%m-%d %H:00"))
        ).all()

    result = [
        {"hour": row.hour, "rate": round(to_float(row.max_len) - to_float(row.min_len), 1)}
        for row in rows
    ]
    if result:
        avg_target = round(sum(r["rate"] for r in result) / len(result), 1)
        for r in result:
            r["target"] = avg_target
    return result


# ── Utilities consumption (totalizer deltas per session, summed) ───────────────

def get_utilities_analytics():
    with get_session() as session:
        # Totalizer resets each session → (max − min) per session = session consumption
        sf_total = session.scalar(
            select(func.sum(Reading.sf_tot))
            .select_from(
                select(
                    Reading.session_id,
                    func.max(Reading.sf_tot).label("sf_tot"),
                ).group_by(Reading.session_id).subquery()
            )
        ) or 0.0

        wat_total = session.scalar(
            select(func.sum(Reading.wat_tot))
            .select_from(
                select(
                    Reading.session_id,
                    func.max(Reading.wat_tot).label("wat_tot"),
                ).group_by(Reading.session_id).subquery()
            )
        ) or 0.0

        air_total = session.scalar(
            select(func.sum(Reading.air_consumed_lot))
        ) or 0.0

        power_total = session.scalar(
            select(func.sum(Reading.power_consumed_lot))
        ) or 0.0

    return [
        {"utility": "SF",    "usage": round(sf_total,    3), "cost": 0.0},
        {"utility": "Water", "usage": round(wat_total,   3), "cost": 0.0},
        {"utility": "Air",   "usage": round(air_total,   3), "cost": 0.0},
        {"utility": "Power", "usage": round(power_total, 3), "cost": 0.0},
    ]


# ── OEE per machine ───────────────────────────────────────────────────────────

def get_oee_list():
    names = session_name_map()
    with get_session() as session:
        rows = session.execute(
            select(
                Reading.machine_name,
                func.count().label("total"),
                func.sum(
                    case((Reading.state == "running", 1), else_=0)
                ).label("running_count"),
                func.avg(
                    case((Reading.speed > 0, Reading.speed), else_=None)
                ).label("avg_speed"),
                func.max(Reading.speed).label("max_speed"),
            )
            .group_by(Reading.machine_name)
            .order_by(Reading.machine_name)
        ).all()

    out = []
    for row in rows:
        total = row.total or 1
        running = int(row.running_count or 0)
        avg_spd = to_float(row.avg_speed)
        max_spd = to_float(row.max_speed)

        availability = round(running / total * 100, 1)
        performance = round(avg_spd / max_spd * 100, 1) if max_spd else 0.0
        quality = 100.0
        oee = round(availability * performance * quality / 10_000, 1)

        out.append({
            "machine_id": row.machine_name,
            "machine_name": names.get(row.machine_name, row.machine_name),
            "availability": availability,
            "performance": performance,
            "quality": quality,
            "oee": oee,
        })
    return out
