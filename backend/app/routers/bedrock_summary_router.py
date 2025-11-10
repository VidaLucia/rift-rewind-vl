# =======================================================
# bedrock_summarizer_router.py
# =======================================================
from fastapi import APIRouter, HTTPException, Query
from backend.app.services.bedrock_summarizer_service import run_bedrock_summarizer_service

router = APIRouter()

@router.post("/timeline/bedrock/{region}/{match_id}/{puuid}")
def summarize_with_bedrock(
    region: str,
    match_id: str,
    puuid: str,
    player_class: str 
):
    """
    Generate an advanced analytical summary for the player using AWS Bedrock (Claude Sonnet 4).
    Accepts `player_class` as a simple string query parameter (not JSON body).
    """
    try:
        result = run_bedrock_summarizer_service(region, match_id, puuid, player_class)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
