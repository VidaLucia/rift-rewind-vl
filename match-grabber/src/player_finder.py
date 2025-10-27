import asyncio
import aiohttp
import pandas as pd
import numpy as np
from riot_client import get_matches, get_match_details, riot_request, REGION_ROUTING, HEADERS

#  CONFIG 
DDRAGON_VERSION_URL = "https://ddragon.leagueoflegends.com/api/versions.json"
DDRAGON_CHAMP_URL = "https://ddragon.leagueoflegends.com/cdn/{}/data/en_US/champion.json"

ROLE_MAP = {
    "TOP": 1,
    "JUNGLE": 2,
    "MIDDLE": 3,
    "BOTTOM": 4,
    "SUPPORT": 5,
    "UTILITY": 5,  # Merge UTILITY = SUPPORT
}
def resolve_position(row):
    """Resolve role from available Riot fields with fallback logic."""
    if row.get("teamPosition"):
        return row["teamPosition"].upper()
    elif row.get("lane"):
        return row["lane"].upper()
    elif row.get("role"):
        return row["role"].upper()
    else:
        return "UNKNOWN"

async def get_puuid(session, region: str, summoner_name: str, tag: str):
    """Fetch PUUID for a given summoner name and tag."""
    routing = REGION_ROUTING.get(region, "americas")
    url = f"https://{routing}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{summoner_name}/{tag}"
    data = await riot_request(session, url, HEADERS, routing=routing)
    return data["puuid"]


async def get_champion_data(session):
    """Fetch champion metadata (attack, defense, magic, difficulty, tags)."""
    versions = await riot_request(session, DDRAGON_VERSION_URL, headers=None, routing="global")
    latest = versions[0]
    url = DDRAGON_CHAMP_URL.format(latest)
    champ_data = (await riot_request(session, url, headers=None, routing="global"))["data"]

    champs = []
    for key, data in champ_data.items():
        champs.append({
            "champion": key,
            "attack": data["info"]["attack"],
            "defense": data["info"]["defense"],
            "magic": data["info"]["magic"],
            "difficulty": data["info"]["difficulty"],
            "tags": ",".join(data.get("tags", []))
        })
    champ_df = pd.DataFrame(champs)
    return champ_df


async def export_match_history(
    region: str,
    summoner_name: str,
    tag: str,
    output_path: str = "../../data/player_cleaned.csv",
    max_matches: int | None = None
):
    async with aiohttp.ClientSession() as session:
        print(f"Fetching PUUID for {summoner_name}#{tag} in {region}")
        puuid = await get_puuid(session, region, summoner_name, tag)
        print(f"PUUID for {summoner_name}#{tag}: {puuid}")

        print("Fetching champion metadata...")
        champ_df = await get_champion_data(session)
        print(f"Loaded {len(champ_df)} champion records from DDragon")

        # Add numeric champion ID (_id) from DDragon
        print("Adding numeric champion IDs...")
        champ_df["champion"] = champ_df["champion"].astype(str)
        champ_df["_id"] = champ_df.index  # numeric ID surrogate
        if "key" in champ_df.columns:
            champ_df["_id"] = champ_df["key"].astype(int)

        print("Fetching match IDs...")
        match_ids = await get_matches(session, puuid, region)
        print(f"Found {len(match_ids)} matches")

        if max_matches:
            match_ids = match_ids[:max_matches]
            print(f"Limiting to {len(match_ids)} matches")

        rows = []

        for i, match_id in enumerate(match_ids):
            if max_matches and i >= max_matches:
                print(f"Reached match limit ({max_matches}), stopping early.")
                break

            print(f"[{i+1}/{len(match_ids)}] Getting match {match_id}")
            try:
                match_data = await get_match_details(session, match_id, region)
                participants = match_data.get("info", {}).get("participants", [])

                for p in participants:
                    if p.get("puuid") != puuid:
                        continue

                    resolved_role = resolve_position(p)
                    numeric_role = ROLE_MAP.get(resolved_role, 0)

                    row = {
                        "match_id": match_id,
                        "duration": p.get("gameDuration", 0),
                        "date": p.get("gameCreation", 0),
                        "champion": p.get("championName", ""),
                        "role_str": resolved_role,
                        "role": numeric_role,
                        "kills": p.get("kills", 0),
                        "deaths": p.get("deaths", 0),
                        "assists": p.get("assists", 0),
                        "damage": p.get("totalDamageDealtToChampions", 0),
                        "cs": p.get("totalMinionsKilled", 0) + p.get("neutralMinionsKilled", 0),
                        "gold_earned": p.get("goldEarned", 0),
                        "vision_score": p.get("visionScore", 0),
                        "win": int(p.get("win", False)),
                        # Challenge metrics
                        "ch_kda": p.get("challenges", {}).get("kda", 0),
                        "ch_killingSprees": p.get("challenges", {}).get("killingSprees", 0),
                        "ch_damagePerMinute": p.get("challenges", {}).get("damagePerMinute", 0),
                        "ch_teamDamagePercentage": p.get("challenges", {}).get("teamDamagePercentage", 0),
                        "ch_killParticipation": p.get("challenges", {}).get("killParticipation", 0),
                        "ch_turretTakedowns": p.get("challenges", {}).get("turretTakedowns", 0),
                        "ch_baronTakedowns": p.get("challenges", {}).get("baronTakedowns", 0),
                        "ch_dragonTakedowns": p.get("challenges", {}).get("dragonTakedowns", 0),
                        "ch_enemyJungleMonsterKills": p.get("challenges", {}).get("enemyJungleMonsterKills", 0),
                        "ch_goldPerMinute": p.get("challenges", {}).get("goldPerMinute", 0),
                        "ch_laningPhaseGoldExpAdvantage": p.get("challenges", {}).get("laningPhaseGoldExpAdvantage", 0),
                        "ch_maxCsAdvantageOnLaneOpponent": p.get("challenges", {}).get("maxCsAdvantageOnLaneOpponent", 0),
                        "ch_deathsByEnemyChamps": p.get("challenges", {}).get("deathsByEnemyChamps", 0),
                        "ch_damageTakenOnTeamPercentage": p.get("challenges", {}).get("damageTakenOnTeamPercentage", 0),
                        "ch_survivedSingleDigitHpCount": p.get("challenges", {}).get("survivedSingleDigitHpCount", 0),
                        "ch_effectiveHealAndShielding": p.get("challenges", {}).get("effectiveHealAndShielding", 0),
                        "ch_saveAllyFromDeath": p.get("challenges", {}).get("saveAllyFromDeath", 0),
                        "ch_immobilizeAndKillWithAlly": p.get("challenges", {}).get("immobilizeAndKillWithAlly", 0),
                        "ch_visionScorePerMinute": p.get("challenges", {}).get("visionScorePerMinute", 0),
                        "ch_wardTakedowns": p.get("challenges", {}).get("wardTakedowns", 0),
                        # Runes
                        "primarystyle_id": p.get("perks", {}).get("styles", [{}])[0].get("style", 0),
                        "substyle_id": p.get("perks", {}).get("styles", [{}])[-1].get("style", 0),
                        "primarystyle_perk1": p.get("perks", {}).get("styles", [{}])[0].get("selections", [{}])[0].get("perk", 0),
                        "primarystyle_perk2": p.get("perks", {}).get("styles", [{}])[0].get("selections", [{}])[1].get("perk", 0),
                        "primarystyle_perk3": p.get("perks", {}).get("styles", [{}])[0].get("selections", [{}])[2].get("perk", 0),
                        "substyle_perk1": p.get("perks", {}).get("styles", [{}])[-1].get("selections", [{}])[0].get("perk", 0),
                        "substyle_perk2": p.get("perks", {}).get("styles", [{}])[-1].get("selections", [{}])[1].get("perk", 0),
                        # Items
                        "item0": p.get("item0"), "item1": p.get("item1"), "item2": p.get("item2"),
                        "item3": p.get("item3"), "item4": p.get("item4"),
                        "item5": p.get("item5"), "item6": p.get("item6"),
                    }
                    rows.append(row)

            except Exception as e:
                print(f"Failed match {match_id}: {e}")

        if not rows:
            print("No matches retrieved.")
            return

        df = pd.DataFrame(rows)
        print(f"Collected {len(df)} rows")

        # --- Derived Features (Trainer-Exact) ---
        print("Computing derived metrics (trainer-compatible)...")

        # Ensure percentage metrics scaled to 0–1
        df["kp_rate"] = df["ch_killParticipation"] / 100
        df["damage_share"] = df["ch_teamDamagePercentage"] / 100

        # Derived ratios identical to trainer script
        df["gold_efficiency"] = df["ch_goldPerMinute"] / df["duration"].clip(lower=1)
        df["objective_focus"] = (
            df["ch_turretTakedowns"] + df["ch_baronTakedowns"] + df["ch_dragonTakedowns"]
        ) / 3
        df["survivability_ratio"] = df["kills"] / df["deaths"].clip(lower=1)
        df["vision_efficiency"] = df["ch_visionScorePerMinute"] / (df["vision_score"] + 1)

        # Per-minute features (trainer-matched)
        df["kpm"] = df["kills"] / df["duration"].clip(lower=1)
        df["dpm"] = df["damage"] / df["duration"].clip(lower=1)
        df["apm"] = df["assists"] / df["duration"].clip(lower=1)
        df["cspm"] = df["cs"] / df["duration"].clip(lower=1)

        # Prune and merge champion info
        before = len(df)
        df = df[~df["champion"].str.startswith("Strawberry_", na=False)]
        df = df[df["primarystyle_id"] != 0]
        after = len(df)
        print(f"Pruned {before - after} invalid rows")

        # Merge champion metadata
        df = df.merge(champ_df[["champion", "_id", "attack", "defense", "magic", "difficulty", "tags"]], on="champion", how="left")

        # One-hot encode tags (trainer expects tag_ prefix)
        tag_dummies = df["tags"].fillna("").str.get_dummies(sep=",").add_prefix("tag_")
        df = pd.concat([df, tag_dummies], axis=1)

        # Fill missing values and export
        df = df.fillna(0)
        df.to_csv(output_path, index=False)
        print(f"Saved cleaned feature-rich data to {output_path}")

if __name__ == "__main__":
    region = "na1"
    players = [
        #("aquatick", "001"),
        #("vida", "lucia"),
        ("aixwy", "cham"),
        ("pyropiller167", "na1"),
        ("lulululu04", "lulu"),
    ]

    for summoner_name, tag in players:
        print(f"Exporting match history for {summoner_name}#{tag}...")
        asyncio.run(
            export_match_history(
                region,
                summoner_name,
                tag,
                output_path=f"../../data/players/{summoner_name}#{tag}_data.csv",
                max_matches=200,
            )
        )
        print(f"Done: {summoner_name}#{tag}\n")