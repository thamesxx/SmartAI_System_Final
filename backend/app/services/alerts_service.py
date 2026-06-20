"""Alerts derived from the latest reading of each machine run.

No alert collection exists in the source data, so alerts are inferred from real
conditions in the readings (non-running state, missing power metering, idle
utilities). If nothing is wrong, an empty list is returned.
"""
from app.database import (
    readings_collection,
    session_name_map,
    map_status,
    to_float,
)


def get_alerts():
    col = readings_collection()
    names = session_name_map()
    alerts = []
    counter = 0

    for sid, name in sorted(names.items(), key=lambda kv: kv[1]):
        doc = col.find_one({"session_id": sid}, sort=[("seq", -1)])
        if not doc:
            continue
        plc = doc.get("plc", {})
        u = doc.get("utility", {})
        ts = doc.get("timestamp")
        status = map_status(doc.get("state"))

        if status != "running":
            counter += 1
            alerts.append({
                "id": f"alert-{counter}",
                "timestamp": ts,
                "type": f"{name} Not Running",
                "message": f"{name} is currently in '{status}' state.",
                "severity": "warning",
                "iconName": "Activity",
            })

        if u.get("EM_Power") is None:
            counter += 1
            alerts.append({
                "id": f"alert-{counter}",
                "timestamp": ts,
                "type": "Power Metering Offline",
                "message": f"{name}: EM power meter is not reporting a value.",
                "severity": "warning",
                "iconName": "AlertTriangle",
            })

        if status == "running" and to_float(u.get("SF_Flow")) == 0:
            counter += 1
            alerts.append({
                "id": f"alert-{counter}",
                "timestamp": ts,
                "type": "No Steam Flow",
                "message": f"{name} is running with zero SF flow.",
                "severity": "info",
                "iconName": "ThermometerSun",
            })

    return alerts
