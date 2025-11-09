# =======================================================
# timeline_pipeline_service.py
# =======================================================
import asyncio
from scripts.timeline_pipeline import grab_timeline

async def run_timeline_pipeline_service(region: str, match_id: str, puuid: str):
    """
    Run the full end-to-end pipeline:
    Riot API  Timeline JSON  Deep Analysis  Player Summary
    """
    result = await grab_timeline(region, match_id, puuid)
    return {"status": "success", "match_id": match_id, "region": region, "puuid": puuid, "result": result}
