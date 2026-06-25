from fastapi import APIRouter
from app.services.analytics_service import (
    get_lot_analytics,
    get_temperature_analytics,
    get_production_analytics,
    get_utilities_analytics,
)

router = APIRouter()

@router.get("/analytics/lot")
def lot():
    return get_lot_analytics()

@router.get("/analytics/temperature")
def temperature():
    return get_temperature_analytics()

@router.get("/analytics/production")
def production():
    return get_production_analytics()

@router.get("/analytics/utilities")
def utilities():
    return get_utilities_analytics()
# /api/oee is now handled by app.routes.oee (oee_service.py)
