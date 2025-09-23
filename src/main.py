import asyncio
import aiohttp
import time
from riot_client import (
    get_puuids_from_rank,
    get_matches,
    get_match_details,
    get_match_timeline,
    get_master_players,
    get_grandmaster_players,
    get_challenger_players,
)
from db_helpers import save_player, save_match, save_player_match, save_league
from db_setup import init_db
from utils import load_progress, save_progress

region_stats = {}

async def process_region(session, region, queue, tier, division, max_players=10, max_matches=20, with_timeline=False):
    start_time = time.perf_counter()
    print(f"\n=== Processing {region} {tier} {division} ===")

    league_data = await get_puuids_from_rank(session, region, queue, tier, division)
    update_region_progress(region, calls=1)

    if isinstance(league_data, dict) and "entries" in league_data:
        save_league(league_data, region)
        players = league_data.get("entries", [])
    else:
        players = league_data

    players = players[:max_players]

    for player in players:
        puuid = player["puuid"]
        save_player(puuid, region)

        matches = await get_matches(session, puuid, region)
        for match_id in matches[:max_matches]:
            try:
                details = await get_match_details(session, match_id, region)

                timeline = None
                if with_timeline:
                    try:
                        timeline = await get_match_timeline(session, match_id, region)
                    except Exception as e:
                        if "404" in str(e):
                            print(f"[{region}] Timeline missing for {match_id} (skipping)")
                        else:
                            raise

                save_match(match_id, region, details, timeline)
                save_player_match(puuid, match_id)
                print(f"[{region}] Saved {match_id} for {puuid}")

            except Exception as e:
                if "404" in str(e):
                    print(f"[{region}] Match {match_id} not found (skipping)")
                    continue
                else:
                    raise

    elapsed = time.perf_counter() - start_time
    print(f"⏱️ Finished {region} {tier} {division} in {elapsed:.2f} seconds")


async def run_pipeline():
    regions = ["na1", "euw1", "kr"]

    async with aiohttp.ClientSession() as session:
        tasks = [process_all_tiers(session, region) for region in regions]
        await asyncio.gather(*tasks)


async def process_all_tiers(session, region, max_players=25, max_matches=20, with_timeline=False):
    progress = load_progress()
    if region not in progress:
        progress[region] = {}

    tiers_with_divs = {
        "IRON": ["I", "II", "III", "IV"],
        "BRONZE": ["I", "II", "III", "IV"],
        "SILVER": ["I", "II", "III", "IV"],
        "GOLD": ["I", "II", "III", "IV"],
        "PLATINUM": ["I", "II", "III", "IV"],
        "EMERALD": ["I", "II", "III", "IV"],
        "DIAMOND": ["I", "II", "III", "IV"],
    }

    for tier, divs in tiers_with_divs.items():
        for division in divs:
            if tier in progress[region] and division in progress[region][tier]:
                print(f"Skipping {region} {tier} {division} (already done)")
                continue

            await process_region(
                session, region, "RANKED_SOLO_5x5", tier, division,
                max_players=max_players, max_matches=max_matches, with_timeline=with_timeline
            )

            progress = load_progress()
            progress.setdefault(region, {}).setdefault(tier, [])
            if division not in progress[region][tier]:
                progress[region][tier].append(division)
            save_progress(progress)
            print(f"Saved checkpoint after {region} {tier} {division}")

    # Masters+
    for special_tier, getter in [
        ("MASTER", get_master_players),
        ("GRANDMASTER", get_grandmaster_players),
        ("CHALLENGER", get_challenger_players),
    ]:
        if special_tier in progress[region]:
            print(f"Skipping {region} {special_tier} (already done)")
            continue

        start_time = time.perf_counter()
        print(f"\n=== Processing {region} {special_tier} ===")
        data = await getter(session, region, "RANKED_SOLO_5x5")

        save_league(data, region)

        players = data.get("entries", [])[:max_players]
        for player in players:
            puuid = player["puuid"]
            save_player(puuid, region)
            matches = await get_matches(session, puuid, region)
            for match_id in matches[:max_matches]:
                try:
                    details = await get_match_details(session, match_id, region)

                    timeline = None
                    if with_timeline:
                        try:
                            timeline = await get_match_timeline(session, match_id, region)
                        except Exception as e:
                            if "404" in str(e):
                                print(f"[{region}] Timeline missing for {match_id} (skipping)")
                            else:
                                raise

                    save_match(match_id, region, details, timeline)
                    save_player_match(puuid, match_id)
                    print(f"[{region}] Saved {match_id} for {puuid}")

                except Exception as e:
                    if "404" in str(e):
                        print(f"[{region}] Match {match_id} not found (skipping)")
                        continue
                    else:
                        raise

        elapsed = time.perf_counter() - start_time
        print(f"⏱️ Finished {region} {special_tier} in {elapsed:.2f} seconds")

        progress = load_progress()
        progress.setdefault(region, {})[special_tier] = ["I"]
        save_progress(progress)
        print(f"Saved checkpoint after {region} {special_tier}")


def update_region_progress(region, calls=0, matches=0, players=0):
    if region not in region_stats:
        region_stats[region] = {"calls": 0, "matches": 0, "players": 0}
    region_stats[region]["calls"] += calls
    region_stats[region]["matches"] += matches
    region_stats[region]["players"] += players

    stats = region_stats[region]
    print(f"[{region}] Progress -> {stats['players']} players, "
          f"{stats['matches']} matches, {stats['calls']} API calls")


if __name__ == "__main__":
    init_db()
    asyncio.run(run_pipeline())
