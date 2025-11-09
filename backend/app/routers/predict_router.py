from fastapi import APIRouter, HTTPException
from app.utils.sanitize import sanitize_nans


from app.services import (
    predict_service,
    class_mapping_service,
    character_sheet_service,
)

router = APIRouter()


@router.post("/player/{player_name}")
def predict_player_class(player_name: str):
    """
    Runs the full player pipeline:
    Predicts clusters
    Assigns D&D classes
    Generates the character sheet
    """
    try:
        # Step 1 — Predict clusters
        predict_result = predict_service.predict_player(player_name)
        # Step 2 — D&D mapping
        mapping_result = class_mapping_service.run_class_mapping(player_name)
        # Step 3 — Generate character sheet (only if mapping succeeded)
        if mapping_result["status"] == "success":
            sheet_result = character_sheet_service.run_sheet_generation(player_name)
        else:
            sheet_result = {"status": "skipped", "message": "D&D mapping failed"}

        return sanitize_nans({
            "status": "success",
            "player": player_name,
            "prediction_output": predict_result,
            "dnd_class_mapping": mapping_result,
            "character_sheet": sheet_result
        })

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
