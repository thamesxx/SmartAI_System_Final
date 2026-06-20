from __future__ import annotations
import os
from dotenv import load_dotenv
load_dotenv()

class Settings:
    PROJECT_NAME: str = "Manufacturing Dashboard API"
    VERSION: str = "0.1"
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./test.db")

settings = Settings()



MONGO_URI = os.getenv("MONGODB_URI", "mongodb+srv://aliabdullah121:bingoman@cluster0.aid5c3g.mongodb.net")
DB_NAME = os.getenv("MONGODB_DB", "machine_telemetry")
ROWS = int(os.getenv("GEN_ROWS", "600"))
SESSIONS = int(os.getenv("GEN_SESSIONS", "1"))
SEED = os.getenv("GEN_SEED")
SEED = int(SEED) if SEED not in (None, "") else None
NO_IDLE = os.getenv("GEN_NO_IDLE", "0").strip().lower() in {"1", "true", "yes", "y"}