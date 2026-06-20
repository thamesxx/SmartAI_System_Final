"""Machine cards and status timeline, sourced from machine_readings."""
from app.database import (
    readings_collection,
    session_name_map,
    to_float,
    hms_to_minutes,
    map_status,
    parse_timestamp,
)


def get_machine_data():
    """One card per session (machine run), built from its latest reading."""
    col = readings_collection()
    names = session_name_map()
    out = []
    for sid, name in sorted(names.items(), key=lambda kv: kv[1]):
        doc = col.find_one({"session_id": sid}, sort=[("seq", -1)])
        if not doc:
            continue
        plc = doc.get("plc", {})
        out.append({
            "id": sid,
            "name": name,
            "lot1": str(plc.get("lot_1", "")),
            "lot2": str(plc.get("lot_2", "")),
            "articleNumber": str(plc.get("article", "")),
            "totalLength": to_float(plc.get("length")),
            "status": map_status(doc.get("state")),
            "lotTime": hms_to_minutes(plc.get("lot_time")),
            "machineRunningTime": hms_to_minutes(plc.get("machine_time")),
            "speed": to_float(plc.get("speed")),
        })
    return out


def get_machine_timeline(range: str = "shift"):
    """Bucket every reading into running/stopped counts.

    Hourly buckets for shift/day, daily buckets for week/month. The counts are
    a proxy for minutes spent in each state (one reading ≈ one sample).
    """
    col = readings_collection()
    daily = range in ("week", "month")
    buckets: dict[str, list[int]] = {}
    for doc in col.find({}, projection={"timestamp": 1, "state": 1}):
        ts = parse_timestamp(doc.get("timestamp"))
        if ts is None:
            continue
        key = ts.strftime("%m-%d") if daily else ts.strftime("%H:00")
        bucket = buckets.setdefault(key, [0, 0])
        if map_status(doc.get("state")) == "running":
            bucket[0] += 1
        else:
            bucket[1] += 1
    return [
        {"time": key, "running": running, "stopped": stopped}
        for key, (running, stopped) in sorted(buckets.items())
    ]
