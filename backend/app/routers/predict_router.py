from fastapi import APIRouter, HTTPException
from backend.app.utils.sanitize import sanitize_nans
from backend.app.services import (
    predict_service,
    character_sheet_service,
)

router = APIRouter()


@router.post("/player/{player_name}")
def predict_player_class(player_name: str):
    """
    Runs the full player pipeline:
    1. Predict clusters and compute inline D&D class mapping
    2. Generate the D&D character sheet
    """
    try:
        print(f"[INFO] Starting pipeline for {player_name}")

        # Step 1 — Predict + Inline D&D Mapping
        predict_result = predict_service.predict_player(player_name)

        # Step 2 — Generate character sheet
        sheet_result = character_sheet_service.run_sheet_generation(player_name)

        return sanitize_nans({
            "status": "success",
            "player": player_name,
            "prediction_output": predict_result,
            "character_sheet": sheet_result
        })

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
