# consumer.py
import pika
import json
import logging
import os
import time
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("consumer")

# ===== MongoDB Config =====
MONGO_URI = os.getenv("MONGO_URI", os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
MONGO_DB  = os.getenv("MONGO_DB",  os.getenv("MONGODB_DB",  "machine_db"))

mongo_client = MongoClient(MONGO_URI)
mongo_db     = mongo_client[MONGO_DB]

readings_col   = mongo_db["machine_readings"]
machine_runs_col = mongo_db["machine_runs"]

readings_col.create_index([("session_id", ASCENDING), ("seq", ASCENDING)])
readings_col.create_index([("timestamp",  ASCENDING)])
readings_col.create_index([("machine_name", ASCENDING)])

machine_runs_col.create_index([("session_id",   ASCENDING)])
machine_runs_col.create_index([("machine_name", ASCENDING)])
machine_runs_col.create_index([("failure_ts",   ASCENDING)])
machine_runs_col.create_index([("component",    ASCENDING)])

# ===== RabbitMQ Config =====
RABBIT_USER    = os.getenv("RABBIT_MQ_USER")
RABBIT_PASS    = os.getenv("RABBIT_MQ_PASSWORD")
RABBIT_HOST    = os.getenv("RABBIT_MQ_HOST")
RABBIT_PORT    = int(os.getenv("RABBIT_MQ_PORT", "5672"))

EXCHANGE       = os.getenv("RABBIT_MQ_EXCHANGE",             "scada_data")
QUEUE          = os.getenv("RABBIT_MQ_QUEUE",                "scada_data_queue")
ROUTING        = os.getenv("RABBIT_MQ_ROUTING_KEY",          "scada.tag.data")

# New event queue for machine_runs
EVENT_QUEUE    = os.getenv("RABBIT_MQ_EVENT_QUEUE",          "machine_event_queue")
EVENT_ROUTING  = os.getenv("RABBIT_MQ_EVENT_ROUTING_KEY",    "scada.machine.event")

HEALTH_QUEUE   = os.getenv("RABBIT_MQ_HEALTH_DATA_QUEUE",    "sensor_health_data_queue")
HEALTH_ROUTING = os.getenv("RABBIT_MQ_HEALTH_DATA_ROUTING_KEY", "scada.sensor.health")


def write_reading(data: dict):
    try:
        readings_col.insert_one(data)
        logger.info(
            f"reading → session={data.get('session_id','?')[:8]}  "
            f"machine={data.get('machine_name','?')}  seq={data.get('seq','?')}  "
            f"state={data.get('state','?')}"
        )
    except Exception as e:
        logger.error(f"ERROR writing reading: {e}")


def write_event(data: dict):
    """Write a machine_runs event document."""
    try:
        machine_runs_col.insert_one(data)
        logger.info(
            f"event   → machine={data.get('machine_name','?')}  "
            f"component={data.get('component','?')}  "
            f"severity={data.get('severity','?')}"
        )
    except Exception as e:
        logger.error(f"ERROR writing event: {e}")


def on_message(ch, method, props, body):
    try:
        data = json.loads(body)
        write_reading(data)
    except Exception as e:
        logger.error(f"Error processing reading message: {e}")
    ch.basic_ack(method.delivery_tag)


def on_event_message(ch, method, props, body):
    """Handler for scada.machine.event → machine_runs collection."""
    try:
        data = json.loads(body)
        write_event(data)
    except Exception as e:
        logger.error(f"Error processing event message: {e}")
    ch.basic_ack(method.delivery_tag)


def on_health_message(ch, method, props, body):
    try:
        data = json.loads(body)
        logger.info(f"Health message received: {data}")
    except Exception as e:
        logger.error(f"Error processing health message: {e}")
    ch.basic_ack(method.delivery_tag)


def connect_and_consume():
    creds = pika.PlainCredentials(RABBIT_USER, RABBIT_PASS)

    while True:
        try:
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host=RABBIT_HOST,
                    port=RABBIT_PORT,
                    credentials=creds,
                    heartbeat=600,
                    blocked_connection_timeout=300,
                )
            )
            logger.info("Connected to RabbitMQ")
            ch = connection.channel()
            ch.exchange_declare(exchange=EXCHANGE, exchange_type="topic", durable=True)

            # Telemetry queue → machine_readings
            ch.queue_declare(queue=QUEUE, durable=True)
            ch.queue_bind(exchange=EXCHANGE, queue=QUEUE, routing_key=ROUTING)
            ch.basic_consume(queue=QUEUE, on_message_callback=on_message)
            logger.info(f"Listening: '{QUEUE}' ({ROUTING})")

            # Event queue → machine_runs
            ch.queue_declare(queue=EVENT_QUEUE, durable=True)
            ch.queue_bind(exchange=EXCHANGE, queue=EVENT_QUEUE, routing_key=EVENT_ROUTING)
            ch.basic_consume(queue=EVENT_QUEUE, on_message_callback=on_event_message)
            logger.info(f"Listening: '{EVENT_QUEUE}' ({EVENT_ROUTING})")

            # Health queue (legacy — kept wired, no-op handler)
            ch.queue_declare(queue=HEALTH_QUEUE, durable=True)
            ch.queue_bind(exchange=EXCHANGE, queue=HEALTH_QUEUE, routing_key=HEALTH_ROUTING)
            ch.basic_consume(queue=HEALTH_QUEUE, on_message_callback=on_health_message)
            logger.info(f"Listening: '{HEALTH_QUEUE}' ({HEALTH_ROUTING})")

            logger.info("Waiting for messages...")
            ch.start_consuming()

        except Exception as e:
            logger.warning(f"RabbitMQ not available, retrying in 5s: {e}")
            time.sleep(5)


if __name__ == "__main__":
    connect_and_consume()
