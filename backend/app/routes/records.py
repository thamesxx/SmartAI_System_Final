from fastapi import APIRouter, Query
from app.services.records_service import get_records

router = APIRouter()

@router.get("/database-records")
def records(search: str | None = Query(None), status: str | None = Query(None)):
    return get_records(search=search, status=status)
