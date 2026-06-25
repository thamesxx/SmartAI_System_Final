from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# config.py adds project root to sys.path (required before ml/ imports)
import app.config  # noqa: F401 — side-effect import

from app.database import create_tables
from app.routes import machine, analytics, utilities, alerts, records
from app.routes import oee, prediction

app = FastAPI(title="Manufacturing Dashboard API", version="0.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(machine.router,     prefix="/api")
app.include_router(analytics.router,   prefix="/api")
app.include_router(utilities.router,   prefix="/api")
app.include_router(alerts.router,      prefix="/api")
app.include_router(records.router,     prefix="/api")
app.include_router(oee.router,         prefix="/api")
app.include_router(prediction.router,  prefix="/api")


@app.on_event("startup")
def on_startup() -> None:
    create_tables()


@app.get("/")
def root():
    return {"status": "API Running", "version": "0.2"}
