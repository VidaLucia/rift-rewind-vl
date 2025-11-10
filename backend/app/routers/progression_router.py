from fastapi import APIRouter, HTTPException
from backend.app.services import progression_service

router = APIRouter()

@router.post("/progress/{player_name}")
def update_progress(player_name: str):
    """
    Runs progression.py for a player, grants EXP and items, and triggers level-up if applicable.
    """
    result = progression_service.run_progression(player_name)
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    return result
