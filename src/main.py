import asyncio
import aiohttp
from riot_client import (
    get_puuids_from_rank,
    get_matches,
    get_match_details,
    get_match_timeline,
    get_master_players,
    get_grandmaster_players,
    get_challenger_players,
)
from db_helpers import save_player, save_match, save_player_match
from db_setup import init_db
from utils import load_progress, save_progress

async def process_region(session, region, queue, tier, division, max_players=25,max_matches=5):
    print(f"\n=== Processing {region} {tier} {division} ===")
    
    players = await get_puuids_from_rank(session, region, queue, tier, division)
    
    # Limit to 50 players
    players = players[:max_players]

    for player in players:
        puuid = player["puuid"]
        save_player(puuid, region)

        matches = await get_matches(session, puuid, region)
        for match_id in matches[:max_matches]:
            details = await get_match_details(session, match_id, region)
            timeline = await get_match_timeline(session, match_id, region)

            save_match(match_id, region, details, timeline)
            save_player_match(puuid, match_id)

            print(f"[{region}] Saved {match_id} for {puuid}")



async def run_pipeline():
    regions = ["na1", "euw1", "kr"]

    async with aiohttp.ClientSession() as session:
        tasks = [process_all_tiers(session, region) for region in regions]
        await asyncio.gather(*tasks)


async def process_all_tiers(session, region, max_players=25, max_matches=20):
    progress = load_progress()
    if region not in progress:
        progress[region] = {}

    # Normal tiers
    tiers_with_divs = {
        "IRON": ["I", "II", "III", "IV"],
        "BRONZE": ["I", "II", "III", "IV"],
        "SILVER": ["I", "II", "III", "IV"],
        "GOLD": ["I", "II", "III", "IV"],
        "PLATINUM": ["I", "II", "III", "IV"],
        "DIAMOND": ["I", "II", "III", "IV"],
    }

    for tier, divs in tiers_with_divs.items():
        for division in divs:
            if tier in progress[region] and division in progress[region][tier]:
                print(f"Skipping {region} {tier} {division} (already done)")
                continue

            await process_region(session, region, "RANKED_SOLO_5x5", tier, division, 
                                 max_players=max_players, max_matches=max_matches)

            # save checkpoint
            progress.setdefault(region, {}).setdefault(tier, []).append(division)
            save_progress(progress)
            print(f"💾 Saved checkpoint after {region} {tier} {division}")

    # Masters+
    for special_tier, getter in [
        ("MASTER", get_master_players),
        ("GRANDMASTER", get_grandmaster_players),
        ("CHALLENGER", get_challenger_players),
    ]:
        if special_tier in progress[region]:
            print(f"Skipping {region} {special_tier} (already done)")
            continue

        print(f"\n=== Processing {region} {special_tier} ===")
        data = await getter(session, region, "RANKED_SOLO_5x5")
        players = data.get("entries", [])[:max_players]

        for player in players:
            puuid = player["puuid"]
            save_player(puuid, region)
            matches = await get_matches(session, puuid, region)
            for match_id in matches[:max_matches]:
                details = await get_match_details(session, match_id, region)
                timeline = await get_match_timeline(session, match_id, region)
                save_match(match_id, region, details, timeline)
                save_player_match(puuid, match_id)
                print(f"[{region}] Saved {match_id} for {puuid}")

        # save checkpoint
        progress.setdefault(region, {})[special_tier] = ["I"]
        save_progress(progress)
        print(f"Saved checkpoint after {region} {special_tier}")

if __name__ == "__main__":
    init_db()
    asyncio.run(run_pipeline())
