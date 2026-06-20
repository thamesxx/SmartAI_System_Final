"""MongoDB access layer + shared parsing helpers.

All backend data is read from the `machine_telemetry` database, collection
`machine_readings`. Each document is one time-stamped reading of a machine run
(identified by `session_id`). See README/config for connection details.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pymongo import MongoClient

from app.config import MONGO_URI, DB_NAME

_client: Optional[MongoClient] = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
    return _client


def get_db():
    return get_client()[DB_NAME]


def readings_collection():
    """The machine_readings collection (the only populated collection)."""
    return get_db()["machine_readings"]


# ── Parsing helpers ──────────────────────────────────────────────────────────
# PLC/utility values arrive as strings ("25.00"), numbers, or null.

def to_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def hms_to_minutes(value) -> int:
    """Convert an "H:M:S" string (e.g. "7051:33:23") to whole minutes."""
    if not value:
        return 0
    try:
        nums = [int(p) for p in str(value).split(":")]
    except ValueError:
        return 0
    while len(nums) < 3:
        nums.insert(0, 0)
    hours, minutes, seconds = nums[0], nums[1], nums[2]
    return int(hours * 60 + minutes + round(seconds / 60))


def parse_timestamp(value) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


_VALID_STATUS = {"running", "idle", "maintenance", "error"}


def map_status(state) -> str:
    s = str(state or "").lower()
    return s if s in _VALID_STATUS else "idle"


def session_name_map() -> dict[str, str]:
    """Map each session_id to a stable display name ("Machine 1", "Machine 2"…),
    ordered by the session's first reading timestamp."""
    col = readings_collection()
    sessions = []
    for sid in col.distinct("session_id"):
        first = col.find_one(
            {"session_id": sid}, sort=[("seq", 1)], projection={"timestamp": 1}
        )
        sessions.append((str(first["timestamp"]) if first else "", sid))
    sessions.sort()
    return {sid: f"Machine {i + 1}" for i, (_, sid) in enumerate(sessions)}
