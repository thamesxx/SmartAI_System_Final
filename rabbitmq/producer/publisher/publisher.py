"""
publisher.py — Multi-machine live stream producer.

Runs N SyntheticMachineGenerator instances on wall-clock time,
publishing telemetry on 'scada.tag.data' and failure/maintenance
events on 'scada.machine.event'. Checkpoints state to MongoDB every
CHECKPOINT_EVERY readings so a restart resumes cleanly.

On startup, attempts to load each machine's last checkpoint from
generator_state collection; falls back to fresh generator.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from datetime import datetime, timezone

import pika
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING

from machine_data_generator import SyntheticMachineGenerator

load_dotenv()

# ─── RabbitMQ ──────────────────────────────────────────────────────────────
RABBIT_USER     = os.getenv("RABBIT_MQ_USER")
RABBIT_PASS     = os.getenv("RABBIT_MQ_PASSWORD")
RABBIT_HOST     = os.getenv("RABBIT_MQ_HOST")
RABBIT_PORT     = int(os.getenv("RABBIT_MQ_PORT", "5672"))
EXCHANGE        = os.getenv("RABBIT_MQ_EXCHANGE", "scada_data")
ROUTING_DATA    = os.getenv("RABBIT_MQ_ROUTING_KEY",       "scada.tag.data")
ROUTING_EVENT   = os.getenv("RABBIT_MQ_EVENT_ROUTING_KEY", "scada.machine.event")

# ─── MongoDB (for checkpoints) ─────────────────────────────────────────────
MONGO_URI       = os.getenv("MONGODB_URI", os.getenv("MONGO_URI", "mongodb://localhost:27017"))
MONGO_DB        = os.getenv("MONGODB_DB",  os.getenv("MONGO_DB",  "machine_telemetry"))

# ─── Generator config ──────────────────────────────────────────────────────
N_MACHINES      = int(os.getenv("GEN_MACHINES",        "3"))
MASTER_SEED     = int(os.getenv("GEN_SEED",            "42"))
LIVE_INTERVAL_S = float(os.getenv("GEN_LIVE_INTERVAL", "3.0"))
DT              = float(os.getenv("GEN_DT_SECONDS",    "3.0"))
CHECKPOINT_EVERY= int(os.getenv("CHECKPOINT_EVERY",    "200"))

# ─── MongoDB checkpoint store ─────────────────────────────────────────────
mongo_client   = MongoClient(MONGO_URI)
mongo_db       = mongo_client[MONGO_DB]
checkpoint_col = mongo_db["generator_state"]
checkpoint_col.create_index([("machine_id", ASCENDING)], unique=True)


def _load_checkpoint(machine_id: int) -> dict | None:
    doc = checkpoint_col.find_one({"machine_id": machine_id})
    if doc and "state" in doc:
        return doc["state"]
    return None


def _save_checkpoint(gen: SyntheticMachineGenerator) -> None:
    state = gen.serialize()
    checkpoint_col.replace_one(
        {"machine_id": gen.machine_id},
        {"machine_id": gen.machine_id, "state": state,
         "saved_at": datetime.now(timezone.utc).isoformat()},
        upsert=True,
    )


# ─── RabbitMQ connection with retry ────────────────────────────────────────

def _connect_rabbitmq() -> tuple[pika.BlockingConnection, pika.channel.Channel]:
    creds = pika.PlainCredentials(RABBIT_USER, RABBIT_PASS)
    for attempt in range(1, 11):
        try:
            print(f"Connecting to RabbitMQ (attempt {attempt}/10)...")
            conn = pika.BlockingConnection(
                pika.ConnectionParameters(host=RABBIT_HOST, port=RABBIT_PORT, credentials=creds,
                                          heartbeat=600, blocked_connection_timeout=300)
            )
            ch = conn.channel()
            ch.exchange_declare(exchange=EXCHANGE, exchange_type="topic", durable=True)
            print("Connected to RabbitMQ.")
            return conn, ch
        except pika.exceptions.AMQPConnectionError:
            if attempt < 10:
                print(f"  Failed — retrying in 3s...")
                time.sleep(3)
            else:
                print("Max retries reached. Exiting.")
                sys.exit(1)


def _publish(ch, routing_key: str, payload: dict) -> None:
    ch.basic_publish(
        exchange=EXCHANGE,
        routing_key=routing_key,
        body=json.dumps(payload),
        properties=pika.BasicProperties(delivery_mode=2),
    )


# ─── Generator initialisation ──────────────────────────────────────────────

def _init_generators() -> list[SyntheticMachineGenerator]:
    generators = []
    for mid in range(N_MACHINES):
        seed = MASTER_SEED * 1000 + mid
        cp = _load_checkpoint(mid)
        gen = SyntheticMachineGenerator(machine_id=mid, seed=seed, dt=DT)
        if cp:
            try:
                gen.restore(cp)
                print(f"  Machine {mid+1}: resumed from checkpoint (seq={gen.seq}, state={gen.state})")
            except Exception as e:
                print(f"  Machine {mid+1}: checkpoint restore failed ({e}), starting fresh.")
        else:
            print(f"  Machine {mid+1}: no checkpoint — starting fresh (session={gen.session_id[:8]}…)")
        generators.append(gen)
    return generators


# ─── Graceful shutdown ─────────────────────────────────────────────────────

_running = True

def _handle_shutdown(sig, frame):
    global _running
    print("\nShutdown signal received — checkpointing and exiting.")
    _running = False

signal.signal(signal.SIGINT,  _handle_shutdown)
signal.signal(signal.SIGTERM, _handle_shutdown)


# ─── Main loop ─────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\nPublisher starting — {N_MACHINES} machine(s), interval={LIVE_INTERVAL_S}s, dt={DT}s\n")
    conn, ch = _connect_rabbitmq()
    generators = _init_generators()
    print()

    total_published = 0
    total_events    = 0

    try:
        while _running:
            tick_start = time.time()

            for gen in generators:
                if not _running:
                    break

                reading = gen.generate_one()
                try:
                    _publish(ch, ROUTING_DATA, reading)
                    total_published += 1
                except pika.exceptions.AMQPError:
                    print("RabbitMQ error — reconnecting...")
                    conn, ch = _connect_rabbitmq()
                    _publish(ch, ROUTING_DATA, reading)

                # Publish any accumulated events
                for evt in gen.pop_events():
                    try:
                        _publish(ch, ROUTING_EVENT, evt)
                        total_events += 1
                    except pika.exceptions.AMQPError:
                        conn, ch = _connect_rabbitmq()
                        _publish(ch, ROUTING_EVENT, evt)

                # Periodic checkpoint
                if gen.seq % CHECKPOINT_EVERY == 0 and gen.seq > 0:
                    _save_checkpoint(gen)

            # Progress log every 100 ticks
            if total_published % (100 * N_MACHINES) == 0 and total_published > 0:
                statuses = "  ".join(f"M{g.machine_id+1}:{g.state[:4]}(seq={g.seq})" for g in generators)
                print(f"[{total_published:>7} readings | {total_events} events]  {statuses}")

            # Sleep for the remainder of the interval
            elapsed = time.time() - tick_start
            sleep_s = max(0.0, LIVE_INTERVAL_S - elapsed)
            time.sleep(sleep_s)

    finally:
        # Final checkpoint on exit
        for gen in generators:
            try:
                _save_checkpoint(gen)
                print(f"  Checkpoint saved: Machine {gen.machine_id+1} seq={gen.seq}")
            except Exception as e:
                print(f"  Checkpoint failed for Machine {gen.machine_id+1}: {e}")
        try:
            conn.close()
        except Exception:
            pass
        mongo_client.close()
        print(f"Publisher stopped. Published {total_published} readings, {total_events} events.")


if __name__ == "__main__":
    main()
