from fastapi import APIRouter
from app.services.prediction_service import (
    predict_machine,
    predict_all,
    current_model_summary,
)

router = APIRouter()


@router.get("/predict/{machine}")
def predict(machine: str):
    """Score one machine by machine_name. Returns predicted component + probabilities."""
    return predict_machine(machine)


@router.get("/maintenance")
def maintenance():
    """Score all machines, sorted by risk (highest first)."""
    return predict_all()


@router.get("/model/info")
def model_info():
    """Summary of the currently loaded model artifact."""
    return current_model_summary()
