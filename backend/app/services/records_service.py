"""Database table rows — every reading mapped to a DataRecord."""
from app.database import (
    readings_collection,
    session_name_map,
    to_float,
    hms_to_minutes,
    map_status,
)


def get_records(search: str | None = None, status: str | None = None):
    col = readings_collection()
    names = session_name_map()

    query = {}
    if status and status != "all":
        query["state"] = status

    out = []
    for doc in col.find(query).sort("timestamp", -1):
        plc = doc.get("plc", {})
        sid = doc.get("session_id", "")
        record = {
            "id": str(doc.get("_id")),
            "timestamp": doc.get("timestamp"),
            "machineId": names.get(sid, sid),
            "lot1": str(plc.get("lot_1", "")),
            "lot2": str(plc.get("lot_2", "")),
            "articleNumber": str(plc.get("article", "")),
            "totalLength": to_float(plc.get("length")),
            "speed": to_float(plc.get("speed")),
            "lotTime": hms_to_minutes(plc.get("lot_time")),
            "machineRunningTime": hms_to_minutes(plc.get("machine_time")),
            "sfConsumption": to_float(plc.get("steam_consumed_lot")),
            "waterConsumption": to_float(plc.get("water_consumed_lot")),
            "airConsumption": to_float(plc.get("air_consumed_lot")),
            "gasConsumption": 0.0,
            "powerConsumption": to_float(plc.get("power_consumed_lot")),
            "status": map_status(doc.get("state")),
        }
        out.append(record)

    if search:
        term = search.lower()
        out = [
            r for r in out
            if term in r["lot1"].lower()
            or term in r["lot2"].lower()
            or term in r["articleNumber"].lower()
            or term in str(r["machineId"]).lower()
        ]

    return out
