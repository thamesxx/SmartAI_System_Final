"""Database table rows — every reading mapped to a DataRecord (MySQL / SQLAlchemy 2.0)."""
from sqlalchemy import select

from app.database import get_session, session_name_map, to_float, hms_to_minutes, map_status
from app.models import Reading


def get_records(search: str | None = None, status: str | None = None):
    names = session_name_map()

    stmt = select(Reading).order_by(Reading.ts.desc()).limit(5_000)
    if status and status != "all":
        stmt = stmt.where(Reading.state == status)

    with get_session() as session:
        readings = session.scalars(stmt).all()

    out = []
    for r in readings:
        record = {
            "id": str(r.id),
            "timestamp": r.ts.isoformat() if r.ts else None,
            "machineId": names.get(r.machine_name, r.machine_name),
            "lot1": str(r.lot_1 or ""),
            "lot2": str(r.lot_2 or ""),
            "articleNumber": str(r.article or ""),
            "totalLength": to_float(r.length),
            "speed": to_float(r.speed),
            "lotTime": hms_to_minutes(r.lot_time_s),
            "machineRunningTime": hms_to_minutes(r.machine_time_s),
            "sfConsumption": to_float(r.steam_consumed_lot),
            "waterConsumption": to_float(r.water_consumed_lot),
            "airConsumption": to_float(r.air_consumed_lot),
            "gasConsumption": 0.0,
            "powerConsumption": to_float(r.power_consumed_lot),
            "status": map_status(r.state),
        }
        out.append(record)

    if search:
        term = search.lower()
        out = [
            rec for rec in out
            if term in rec["lot1"].lower()
            or term in rec["lot2"].lower()
            or term in rec["articleNumber"].lower()
            or term in str(rec["machineId"]).lower()
        ]

    return out
