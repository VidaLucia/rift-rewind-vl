from fastapi import APIRouter, HTTPException
from backend.app.services import sheet_service

router = APIRouter()

@router.post("/generate")
def generate_sheet(player_name: str):
    """
    Runs the character_sheet.py script and returns the populated character sheet JSON.
    """
    result = sheet_service.generate_character_sheet(player_name)
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    return result

from backend.app.services import character_sheet_reader_service

@router.get("/character-sheet/{player_name}")
def get_character_sheet(player_name: str):
    """
    Returns the saved D&D character sheet JSON for the given player.
    """
    result = character_sheet_reader_service.get_character_sheet(player_name)
    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result["message"])
    return result