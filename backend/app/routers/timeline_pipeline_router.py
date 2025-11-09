# =======================================================
# timeline_pipeline_route.py
# =======================================================
from fastapi import APIRouter, HTTPException
from app.services.timeline_pipeline_service import run_timeline_pipeline_service

router = APIRouter()

@router.post("/pipeline/{region}/{match_id}/{puuid}")
async def run_full_timeline_pipeline(region: str, match_id: str, puuid: str):
    """
    Fetch the timeline, analyze it, and build player summary in one step.
    """
    try:
        result = await run_timeline_pipeline_service(region, match_id, puuid)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
