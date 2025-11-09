from fastapi import APIRouter, HTTPException
from app.services import levelup_service

router = APIRouter()

@router.post("/levelup/{player_name}")
def level_up_player(player_name: str):
    """
    Triggers the level-up process for a player and updates their character sheet.
    """
    result = levelup_service.run_level_up(player_name)
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    return result
