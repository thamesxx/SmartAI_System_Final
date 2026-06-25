from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger, Index, Integer, JSON, String, Float,
    DateTime, UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Reading(Base):
    """One row per simulation tick written by the consumer."""
    __tablename__ = "machine_readings"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(36), nullable=False)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    machine_name: Mapped[str] = mapped_column(String(64), nullable=False)
    state: Mapped[str] = mapped_column(String(16), nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # PLC block
    lot_1: Mapped[Optional[int]] = mapped_column(Integer)
    lot_2: Mapped[Optional[int]] = mapped_column(Integer)
    article: Mapped[Optional[str]] = mapped_column(String(32))
    speed: Mapped[Optional[float]] = mapped_column(Float)
    length: Mapped[Optional[float]] = mapped_column(Float)
    lot_time_s: Mapped[Optional[int]] = mapped_column(Integer)       # stored as seconds
    machine_time_s: Mapped[Optional[int]] = mapped_column(Integer)   # stored as seconds
    steam_consumed_lot: Mapped[Optional[float]] = mapped_column(Float)
    water_consumed_lot: Mapped[Optional[float]] = mapped_column(Float)
    power_consumed_lot: Mapped[Optional[float]] = mapped_column(Float)
    air_consumed_lot: Mapped[Optional[float]] = mapped_column(Float)

    # Utility block
    sf_flow: Mapped[Optional[float]] = mapped_column(Float)
    sf_tot: Mapped[Optional[float]] = mapped_column(Float)
    wat_flow: Mapped[Optional[float]] = mapped_column(Float)
    wat_tot: Mapped[Optional[float]] = mapped_column(Float)
    em_power: Mapped[Optional[float]] = mapped_column(Float)
    em_energy: Mapped[Optional[float]] = mapped_column(Float)

    # Sensor readings (from "health" sub-dict in generator JSON)
    vibration_rms: Mapped[Optional[float]] = mapped_column(Float)
    motor_current: Mapped[Optional[float]] = mapped_column(Float)
    bearing_temp: Mapped[Optional[float]] = mapped_column(Float)
    winding_temp: Mapped[Optional[float]] = mapped_column(Float)
    air_pressure: Mapped[Optional[float]] = mapped_column(Float)

    # Quality block
    good_count: Mapped[Optional[int]] = mapped_column(Integer)
    reject_count: Mapped[Optional[int]] = mapped_column(Integer)

    # Hidden ground-truth (dev only, never used as ML feature)
    truth_json: Mapped[Optional[dict]] = mapped_column(JSON)

    __table_args__ = (
        UniqueConstraint("session_id", "seq", name="uq_reading_session_seq"),
        Index("idx_reading_ts", "ts"),
        Index("idx_reading_machine", "machine_name"),
        Index("idx_reading_lot", "lot_1"),
    )


class MachineRun(Base):
    """One row per failure/repair event emitted by the generator."""
    __tablename__ = "machine_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(36), nullable=False)
    machine_name: Mapped[str] = mapped_column(String(64), nullable=False)
    component: Mapped[Optional[str]] = mapped_column(String(32))
    severity: Mapped[Optional[str]] = mapped_column(String(16))
    run_start_ts: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    failure_ts: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    repair_ts: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    run_hours_to_failure: Mapped[Optional[float]] = mapped_column(Float)
    seq_at_failure: Mapped[Optional[int]] = mapped_column(Integer)

    __table_args__ = (
        Index("idx_run_session", "session_id"),
        Index("idx_run_machine", "machine_name"),
        Index("idx_run_failure_ts", "failure_ts"),
        Index("idx_run_component", "component"),
    )


class GeneratorState(Base):
    """Checkpoint table — one row per physical machine, upserted on save."""
    __tablename__ = "generator_state"

    machine_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    state_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    saved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class FeatureSnapshot(Base):
    """Catalog of sealed Parquet feature shards (written by ml/retention.py)."""
    __tablename__ = "feature_snapshots"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    pipeline_version: Mapped[str] = mapped_column(String(64), nullable=False)
    shard_path: Mapped[str] = mapped_column(String(512), nullable=False)
    range_start_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    range_end_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    class_counts: Mapped[dict] = mapped_column(JSON, nullable=False)
    ref_dist: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("idx_snap_version", "pipeline_version"),
        Index("idx_snap_range", "range_start_ts", "range_end_ts"),
    )
