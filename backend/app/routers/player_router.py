from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services import player_service
import re
router = APIRouter()

class PlayerRequest(BaseModel):
    region: str
    summoner_name: str
    tag: str
    max_matches: int = 200

@router.post("/find")
async def find_player(req: PlayerRequest):
    """
    Fetches a player's match history from Riot API and exports a CSV.
    Returns both the CSV path and the player's PUUID for timeline analysis.
    """
    result = player_service.find_and_export_player(
        region=req.region,
        summoner_name=req.summoner_name,
        tag=req.tag,
        max_matches=req.max_matches,
    )

    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])

    # ✅ Extract PUUID from the log (since player_finder prints it)
    log_text = result.get("log", "")
    puuid_match = re.search(r"PUUID for .*?: ([A-Za-z0-9\-_]+)", log_text)
    puuid = puuid_match.group(1) if puuid_match else None

    # ✅ Add the PUUID into the response payload
    return {
        **result,
        "puuid": puuid,
        "message": f"Exported {req.summoner_name}#{req.tag} successfully."
                   + (f" Found PUUID: {puuid}" if puuid else " (no PUUID found in log).")
    }
from urllib.parse import unquote

@router.get("/history/{player_name}")
def get_player_history(player_name: str):
    import pandas as pd, os
    from pathlib import Path

    # Decode URL-encoded characters (e.g. %23 → #)
    player_name = unquote(player_name)

    BASE_DIR = Path(__file__).resolve().parents[3]
    DATA_DIR = BASE_DIR / "data" / "players"

    # Try both file variants
    primary_csv = DATA_DIR / f"{player_name}_data.csv"
    clustered_csv = DATA_DIR / f"{player_name}_data_clustered.csv"

    csv_path = primary_csv if primary_csv.exists() else clustered_csv
    print(f"[DEBUG] Looking for file: {csv_path}")
    print(f"[DEBUG] Directory listing: {list(DATA_DIR.glob('*'))}")
    if not csv_path.exists():
        raise HTTPException(404, f"No player data found for {player_name}")

    df = pd.read_csv(csv_path, on_bad_lines="skip")

    for col in ["date", "duration"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.fillna(0)

    cols = [
        "match_id", "date", "duration", "champion", "role_str", "win",
        "kills", "deaths", "assists", "ch_kda", "damage", "cs",
        "gold_earned", "vision_score", "ch_teamDamagePercentage",
        "ch_killParticipation", "ch_goldPerMinute",
        "ch_turretTakedowns", "ch_dragonTakedowns", "ch_baronTakedowns",
        "cluster"
    ]
    df = df[[c for c in cols if c in df.columns]]

    print(f"[DEBUG] Returning {len(df)} matches for {player_name}")
    return df.to_dict(orient="records")
