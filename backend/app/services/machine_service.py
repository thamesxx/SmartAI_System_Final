"""Machine cards and status timeline (MySQL / SQLAlchemy 2.0)."""
from sqlalchemy import case, func, select

from app.database import get_session, session_name_map, to_float, hms_to_minutes, map_status
from app.models import Reading


def get_machine_data():
    """One card per physical machine, built from its latest reading."""
    names = session_name_map()

    with get_session() as session:
        machine_names = session.scalars(
            select(Reading.machine_name).distinct().order_by(Reading.machine_name)
        ).all()

        out = []
        for mname in machine_names:
            r = session.scalars(
                select(Reading)
                .where(Reading.machine_name == mname)
                .order_by(Reading.ts.desc())
                .limit(1)
            ).first()
            if r is None:
                continue
            out.append({
                "id": mname,
                "name": names.get(mname, mname),
                "lot1": str(r.lot_1 or ""),
                "lot2": str(r.lot_2 or ""),
                "articleNumber": str(r.article or ""),
                "totalLength": to_float(r.length),
                "status": map_status(r.state),
                "lotTime": hms_to_minutes(r.lot_time_s),
                "machineRunningTime": hms_to_minutes(r.machine_time_s),
                "speed": to_float(r.speed),
            })
    return out


def get_machine_timeline(range: str = "shift"):
    """Bucket every reading into running/stopped counts.

    Hourly buckets for shift/day; daily buckets for week/month.
    """
    daily = range in ("week", "month")
    fmt = "%m-%d" if daily else "%H:00"

    with get_session() as session:
        rows = session.execute(
            select(
                func.date_format(Reading.ts, fmt).label("bucket"),
                func.sum(case((Reading.state == "running", 1), else_=0)).label("running"),
                func.sum(case((Reading.state != "running", 1), else_=0)).label("stopped"),
            )
            .where(Reading.ts.is_not(None))
            .group_by(func.date_format(Reading.ts, fmt))
            .order_by(func.date_format(Reading.ts, fmt))
        ).all()

    return [
        {"time": row.bucket, "running": int(row.running or 0), "stopped": int(row.stopped or 0)}
        for row in rows
    ]
