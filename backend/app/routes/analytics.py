from fastapi import APIRouter
from app.services.analytics_service import (
    get_lot_analytics,
    get_temperature_analytics,
    get_production_analytics,
    get_utilities_analytics,
    get_oee_list,
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

# OEE endpoint expected by the frontend (/api/oee)
@router.get("/oee")
def oee():
    return get_oee_list()
