"""Analytics + OEE, derived from machine_readings.

Notes on fields that do not exist in the source data:
  - OEE 'quality': no defect/quality data is recorded, so quality is reported
    as 100%. OEE = availability × performance × quality.
  - Production 'target': the dataset has no target, so the average production
    rate across all buckets is used as the reference target line.
  - Utilities 'cost': no tariff/cost data exists, so cost is reported as 0.
"""
from app.database import (
    readings_collection,
    session_name_map,
    to_float,
    map_status,
    parse_timestamp,
)


# ── Speed vs Lot length (grouped by lot) ─────────────────────────────────────
def get_lot_analytics():
    col = readings_collection()
    out = []
    for lot in col.distinct("plc.lot_1"):
        docs = list(col.find({"plc.lot_1": lot}, projection={"plc": 1}))
        speeds = [to_float(d.get("plc", {}).get("speed")) for d in docs]
        lengths = [to_float(d.get("plc", {}).get("length")) for d in docs]
        avg_speed = round(sum(speeds) / len(speeds), 1) if speeds else 0.0
        out.append({
            "lot": str(lot),
            "speed": avg_speed,
            "totalLength": round(max(lengths), 1) if lengths else 0.0,
        })
    return out


# Legacy alias kept for the /analytics/temperature route.
def get_temperature_analytics():
    return get_lot_analytics()


# ── Production rate vs target (per time bucket) ──────────────────────────────
def get_production_analytics():
    col = readings_collection()
    buckets: dict[str, list[float]] = {}  # hour -> [min_len, max_len]
    for doc in col.find({}, projection={"timestamp": 1, "plc.length": 1}):
        ts = parse_timestamp(doc.get("timestamp"))
        if ts is None:
            continue
        length = to_float(doc.get("plc", {}).get("length"))
        key = ts.strftime("%m-%d %H:00")
        b = buckets.get(key)
        if b is None:
            buckets[key] = [length, length]
        else:
            b[0] = min(b[0], length)
            b[1] = max(b[1], length)

    rows = [
        {"hour": key, "rate": round(hi - lo, 1)}
        for key, (lo, hi) in sorted(buckets.items())
    ]
    if rows:
        target = round(sum(r["rate"] for r in rows) / len(rows), 1)
        for r in rows:
            r["target"] = target
    return rows


# ── Utilities consumption (totalizer deltas) ─────────────────────────────────
def get_utilities_analytics():
    col = readings_collection()

    def _read(doc, field):
        node = doc
        for part in field.split("."):
            node = node.get(part, {}) if isinstance(node, dict) else None
        return to_float(node)

    def totalizer_delta(field: str) -> float:
        """Sum each session's (max - min) totalizer; totalizers reset per run."""
        total = 0.0
        for sid in col.distinct("session_id"):
            docs = list(col.find({"session_id": sid}, projection={field: 1}))
            vals = [_read(d, field) for d in docs]
            if vals:
                total += max(vals) - min(vals)
        return round(total, 3)

    def consumption_sum(field: str) -> float:
        total = 0.0
        for doc in col.find({}, projection={field: 1}):
            total += _read(doc, field)
        return round(total, 3)

    return [
        {"utility": "SF", "usage": totalizer_delta("utility.SF_Tot"), "cost": 0.0},
        {"utility": "Water", "usage": totalizer_delta("utility.Wat_Tot"), "cost": 0.0},
        {"utility": "Air", "usage": consumption_sum("plc.air_consumed_lot"), "cost": 0.0},
        {"utility": "Power", "usage": consumption_sum("plc.power_consumed_lot"), "cost": 0.0},
    ]


# ── OEE per machine run ──────────────────────────────────────────────────────
def get_oee_list():
    col = readings_collection()
    names = session_name_map()
    out = []
    for sid, name in sorted(names.items(), key=lambda kv: kv[1]):
        docs = list(col.find({"session_id": sid}, projection={"state": 1, "plc.speed": 1}))
        if not docs:
            continue
        total = len(docs)
        running = sum(1 for d in docs if map_status(d.get("state")) == "running")
        speeds = [to_float(d.get("plc", {}).get("speed")) for d in docs]
        speeds = [s for s in speeds if s > 0]

        availability = round(running / total * 100, 1) if total else 0.0
        max_speed = max(speeds) if speeds else 0.0
        avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
        performance = round(avg_speed / max_speed * 100, 1) if max_speed else 0.0
        quality = 100.0  # no quality/defect data in source
        oee = round(availability * performance * quality / 10000, 1)

        out.append({
            "machine_id": sid,
            "machine_name": name,
            "availability": availability,
            "performance": performance,
            "quality": quality,
            "oee": oee,
        })
    return out
