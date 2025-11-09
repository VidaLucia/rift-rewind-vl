from fastapi import APIRouter, HTTPException
from pathlib import Path
from urllib.parse import unquote
import json

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[2]
SHEET_DIR = BASE_DIR / "app" / "data" / "populated"


@router.get("/character-sheet/{player_name}")
def get_character_sheet(player_name: str):
    """
    Return a saved D&D sheet JSON for a player.
    Handles URL-encoded characters like %23 for '#'.
    """
    # Decode URL-encoded string
    decoded_name = unquote(player_name)
    print(f"[DEBUG] Raw player_name: {player_name}")
    print(f"[DEBUG] Decoded player_name: {decoded_name}")

    # Prepare possible file paths
    safe_name = decoded_name.replace("#", "_")
    possible_files = [
        SHEET_DIR / f"{decoded_name}_sheet.json",
        SHEET_DIR / f"{safe_name}_sheet.json",
        SHEET_DIR / f"{decoded_name}.json",
        SHEET_DIR / f"{safe_name}.json",
    ]

    print(f"[DEBUG] Checking these files:")
    for p in possible_files:
        print(f"  - {p}")

    for sheet_path in possible_files:
        if sheet_path.exists():
            print(f"[FOUND] Loaded: {sheet_path}")
            try:
                with open(sheet_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return {"status": "success", "sheet": data}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to read character sheet: {e}")

    print(f"[ERROR] No file found for {decoded_name} in {SHEET_DIR}")
    raise HTTPException(status_code=404, detail=f"No character sheet found for {decoded_name}")
