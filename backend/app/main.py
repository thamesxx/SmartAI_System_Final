from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import machine, analytics, utilities, alerts, records

app = FastAPI(title="Manufacturing Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(machine.router, prefix="/api")
app.include_router(analytics.router, prefix="/api")
app.include_router(utilities.router, prefix="/api")
app.include_router(alerts.router, prefix="/api")
app.include_router(records.router, prefix="/api")


@app.get("/")
def root():
    return {"status": "API Running"}