"""Utility gauge cards, built from the most recent reading.

The source document exposes these utility streams:
  utility.SF_Flow / SF_Tot      -> Steam/SF flow + totalizer
  utility.Wat_Flow / Wat_Tot    -> Water flow + totalizer
  utility.EM_Power / EM_Energy  -> Electrical power/energy (currently null)
  plc.steam/water/air/power_consumed_lot -> per-lot consumption
There is no gas stream in the data, so no gas card is produced.
"""
from app.database import readings_collection, to_float


def get_utilities():
    col = readings_collection()
    doc = col.find_one({}, sort=[("timestamp", -1)])
    if not doc:
        return []

    plc = doc.get("plc", {})
    u = doc.get("utility", {})

    return [
        {
            "id": "SF",
            "name": "SF Flowrate",
            "type": "sf",
            "processValue": to_float(u.get("SF_Flow")),
            "processUnit": "m³/h",
            "totalizer": to_float(u.get("SF_Tot")),
            "totalizerUnit": "m³",
            "lotConsumption": to_float(plc.get("steam_consumed_lot")),
            "lotConsumptionUnit": "/lot",
            "status": "normal",
        },
        {
            "id": "WATER",
            "name": "Water Flowrate",
            "type": "water",
            "processValue": to_float(u.get("Wat_Flow")),
            "processUnit": "m³/h",
            "totalizer": to_float(u.get("Wat_Tot")),
            "totalizerUnit": "m³",
            "lotConsumption": to_float(plc.get("water_consumed_lot")),
            "lotConsumptionUnit": "/lot",
            "status": "normal",
        },
        {
            "id": "AIR",
            "name": "Air Flowrate",
            "type": "air",
            "processValue": 0.0,
            "processUnit": "Nm³/h",
            "totalizer": 0.0,
            "totalizerUnit": "Nm³",
            "lotConsumption": to_float(plc.get("air_consumed_lot")),
            "lotConsumptionUnit": "/lot",
            "status": "normal",
        },
        {
            "id": "POWER",
            "name": "EM Power",
            "type": "power",
            "processValue": to_float(u.get("EM_Power")),
            "processUnit": "kW",
            "totalizer": 0.0,
            "totalizerUnit": "kWh",
            "lotConsumption": to_float(plc.get("power_consumed_lot")),
            "lotConsumptionUnit": "kWh/lot",
            "energy": to_float(u.get("EM_Energy")),
            "energyUnit": "kWh",
            "status": "normal",
        },
    ]
