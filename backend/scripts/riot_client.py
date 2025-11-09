import requests, time,os,json,asyncio, aiohttp
from collections import defaultdict
from asyncio import Lock

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("API_KEY")
print(API_KEY)
HEADERS = {"X-Riot-Token": API_KEY}
REGION_ROUTING = {
    "na1": "americas",
    "br1": "americas",
    "lan": "americas",
    "las": "americas",
    "oce": "americas",
    "euw1": "europe",
    "eun1": "europe",
    "tr1": "europe",
    "ru": "europe",
    "kr": "asia",
    "jp1": "asia"
}
req_count = 0
window_start = time.time()

# Track requests per routing cluster
rate_limits = defaultdict(lambda: {
    "timestamps": [],       # store recent call times
    "lock": asyncio.Lock()  # lock per cluster
})
for key in list(REGION_ROUTING.keys()) + list(set(REGION_ROUTING.values())):
    rate_limits[key]
async def riot_request(session, url, headers, params=None, routing=None):
    """
    Riot API request with per-region rate limiting.
    - 20 req/s
    - 100 req/2min
    """
    rl = rate_limits[routing]

    async with rl["lock"]:
        now = time.time()

        # Remove timestamps older than 2 minutes
        rl["timestamps"] = [t for t in rl["timestamps"] if now - t < 120]

        # Enforce 100 requests / 2 min
        while len(rl["timestamps"]) >= 100:
            sleep_time = 120 - (now - rl["timestamps"][0])
            print(f"[{routing}] Hit 100/2min limit -> sleeping {sleep_time:.1f}s")
            await asyncio.sleep(sleep_time)
            now = time.time()
            rl["timestamps"] = [t for t in rl["timestamps"] if now - t < 120]

        # Enforce 20 requests / 1 sec
        one_sec_window = [t for t in rl["timestamps"] if now - t < 1]
        if len(one_sec_window) >= 20:
            sleep_time = 1 - (now - one_sec_window[0])
            print(f"[{routing}] Hit 20/sec limit -> sleeping {sleep_time:.2f}s")
            await asyncio.sleep(sleep_time)
            now = time.time()
            rl["timestamps"] = [t for t in rl["timestamps"] if now - t < 120]

        # Record this request
        rl["timestamps"].append(time.time())

    # Outside lock → make the request
    async with session.get(url, headers=headers, params=params) as resp:
        if resp.status == 429:
            #print(f"[{routing}] 429 Too Many Requests — backoff 2s")
            await asyncio.sleep(2)
            return await riot_request(session, url, headers, params, routing)

        if resp.status in (401, 403):
            print(f" Riot API auth error {resp.status} for {url}")
            raise RuntimeError("API key invalid or expired. Refresh it at https://developer.riotgames.com/")

        resp.raise_for_status()
        return await resp.json()
async def get_puuids_from_rank(session, region: str, queue: str, tier: str, division: str):
    """
    Fetch summoner info for players in a given rank (one page).
    """
    url = f"https://{region}.api.riotgames.com/lol/league/v4/entries/{queue}/{tier}/{division}?page=1"
    #routing = REGION_ROUTING.get(region)
    data = await riot_request(session, url, HEADERS, routing=region)
    #with open(f"{region}_{division}_{tier}_{queue}.json", "w") as f:
        #json.dump(data, f, indent=4)
    return data
async def get_matches(session, puuid: str, region: str):
    """Fetch recent matches for a player by PUUID."""
    routing = REGION_ROUTING.get(region)
    all_matches = []
    for start in range(0, 1000, 100):
        url = f"https://{routing}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
        params = {"start": start, "count": 100}
        batch = await riot_request(session, url, HEADERS, params=params, routing=routing)
        if not batch:
            break
        all_matches.extend(batch)
    return all_matches


async def get_match_details(session, match_id: str, region: str):
    """Fetch detailed match info by match ID."""
    routing = REGION_ROUTING.get(region)
    url = f"https://{routing}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    return await riot_request(session, url, HEADERS, routing=routing)


async def get_match_timeline(session, match_id: str, region: str):
    """Fetch match timeline by match ID."""
    routing = REGION_ROUTING.get(region)
    url = f"https://{routing}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
    return await riot_request(session, url, HEADERS, routing=routing)

async def get_master_players(session, region: str, queue: str):
    url = f"https://{region}.api.riotgames.com/lol/league/v4/masterleagues/by-queue/{queue}"
    return await riot_request(session, url, HEADERS, routing=region)

async def get_grandmaster_players(session, region: str, queue: str):
    url = f"https://{region}.api.riotgames.com/lol/league/v4/grandmasterleagues/by-queue/{queue}"
    return await riot_request(session, url, HEADERS, routing=region)

async def get_challenger_players(session, region: str, queue: str):
    url = f"https://{region}.api.riotgames.com/lol/league/v4/challengerleagues/by-queue/{queue}"
    return await riot_request(session, url, HEADERS, routing=region)

