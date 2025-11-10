# =======================================================
# timeline_pipeline.py
# Full end-to-end pipeline:
# Riot API → Timeline JSON → Deep Analysis → Player Summary
# =======================================================

import asyncio, aiohttp, json, os, sys
import pandas as pd
from collections import defaultdict
from typing import Dict, Any, List
from statistics import mean
from backend.scripts.riot_client import get_match_timeline


# =======================================================
# CONFIG
# =======================================================
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/timeline"))
os.makedirs(DATA_DIR, exist_ok=True)


# =======================================================
# FETCH TIMELINE AND RUN PIPELINE
# =======================================================
async def grab_timeline(region: str, match_id: str, puuid: str):
    """Fetch timeline JSON from Riot API, run analysis, and generate player summaries."""
    raw_path = os.path.join(DATA_DIR, f"{match_id}_timeline.json")

    async with aiohttp.ClientSession() as session:
        timeline = await get_match_timeline(session, match_id, region)
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(timeline, f, indent=4)

    print(f"[Grabber]  Saved timeline to {raw_path}")

    # --- Run analysis (pass puuid) ---
    result = analyze_timeline(timeline, puuid=puuid)
    analyzed_path = os.path.join(DATA_DIR, f"{match_id}_analyzed.json")
    with open(analyzed_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[Analyzer]  Saved analyzed data to {analyzed_path}")

    # --- Generate per-player summaries ---
    participants = timeline["info"]["participants"]
    pid_map = {p["puuid"]: p["participantId"] for p in participants}
    player_id = pid_map.get(puuid)
    if not player_id:
        print(f"[Warning] PUUID {puuid} not found in match participants.")
        return result

    out_path = os.path.join(DATA_DIR, f"{match_id}_player_{puuid}_summary.json")
    build_player_summary(analyzed_path, player_id=player_id, output_path=out_path)
    print(f"[Pipeline]  Completed full pipeline for match {match_id}")
    return result
# =======================================================
# ANALYZER (from timeline_analyzer.py)
# =======================================================
def analyze_timeline(data: Dict[str, Any], puuid: str = None) -> Dict[str, Any]:
    info = data["info"]
    frames = info["frames"]
    participants = info["participants"]
    pid_map = {p["puuid"]: p["participantId"] for p in participants}
    team_map = {p["participantId"]: (100 if p["participantId"] <= 5 else 200) for p in participants}

    pid = pid_map.get(puuid) if puuid else None
    player_team = team_map[pid] if pid else None

    # Frame-level collection
    frame_rows = []
    for frame in frames:
        ts = frame["timestamp"]
        for pid_str, pdata in frame["participantFrames"].items():
            pid_int = int(pid_str)
            frame_rows.append({
                "timestamp": ts,
                "participantId": pid_int,
                "team": team_map[pid_int],
                "level": pdata["level"],
                "xp": pdata["xp"],
                "gold": pdata["totalGold"],
                "cs": pdata["minionsKilled"] + pdata["jungleMinionsKilled"]
            })
    df = pd.DataFrame(frame_rows)

    # Compose analysis results
    player_timeline = per_player_minute_stats(info)
    gold_adv = team_gold_advantage(df)
    momentum = detect_momentum(player_timeline, gold_adv)

    return {
        "matchId": info.get("gameId", "unknown"),
        "solo_kills": detect_solo_kills(info, pid),
        "teamfights": detect_teamfights(info),
        "lane_advantage": lane_advantage(df, pid),
        "gold_advantage": gold_adv,
        "per_player_minute": player_timeline,
        "momentum": momentum,
        "deaths": detect_deaths(info, pid),
        "objectives": detect_objectives(info, player_team)
    }


# (All detect_* and helper functions below are directly pasted from timeline_analyzer.py)
# =======================================================
# SOLO KILLS / TEAMFIGHTS / LANE ADVANTAGE / etc.
# =======================================================
def analyze_timeline(data: Dict[str, Any], puuid: str = None) -> Dict[str, Any]:
    """
    Deep timeline analysis for a Riot match.
    Consumes timeline JSON and returns per-player, per-minute stats,
    lane and team advantages, events, and momentum swings.
    """

    info = data["info"]
    frames = info["frames"]
    participants = info["participants"]
    pid_map = {p["puuid"]: p["participantId"] for p in participants}
    team_map = {p["participantId"]: (100 if p["participantId"] <= 5 else 200) for p in participants}

    pid = pid_map.get(puuid) if puuid else None
    player_team = team_map[pid] if pid else None
    enemy_team = 200 if player_team == 100 else 100

    # Frame-level data
    frame_rows = []
    for frame in frames:
        ts = frame["timestamp"]
        for pid_str, pdata in frame["participantFrames"].items():
            pid_int = int(pid_str)
            frame_rows.append({
                "timestamp": ts,
                "participantId": pid_int,
                "team": team_map[pid_int],
                "level": pdata["level"],
                "xp": pdata["xp"],
                "gold": pdata["totalGold"],
                "cs": pdata["minionsKilled"] + pdata["jungleMinionsKilled"]
            })
    df = pd.DataFrame(frame_rows)

    # Build results
    player_timeline = per_player_minute_stats(info)
    gold_adv = team_gold_advantage(df)
    momentum = detect_momentum(player_timeline, gold_adv)

    results = {
        "solo_kills": detect_solo_kills(info, pid),
        "teamfights": detect_teamfights(info),
        "lane_advantage": lane_advantage(df, pid),
        "gold_advantage": gold_adv,
        "per_player_minute": player_timeline,
        "momentum": momentum,
        "deaths": detect_deaths(info, pid),
        "objectives": detect_objectives(info, player_team)
    }

    return results


# =======================================================
# SOLO KILLS
# =======================================================
def detect_solo_kills(info: Dict[str, Any], pid: int = None) -> List[Dict[str, Any]]:
    """
    Detect solo kills done by the specified player only.
    If pid is None, returns all solo kills in the match.
    """
    solo_kills = []
    for frame in info["frames"]:
        for event in frame.get("events", []):
            if event["type"] != "CHAMPION_KILL":
                continue

            assisters = event.get("assistingParticipantIds", [])
            killer_id = event.get("killerId")
            victim_id = event.get("victimId")

            # Only count if no assists (solo)
            if not assisters:
                if pid is None or killer_id == pid:
                    solo_kills.append({
                        "timestamp": event["timestamp"],
                        "victimId": victim_id,
                        "killerId": killer_id
                    })
    return solo_kills


# =======================================================
# TEAMFIGHTS
# =======================================================
def detect_teamfights(info: Dict[str, Any], window_ms: int = 10000) -> List[Dict[str, Any]]:
    """
    Detects clusters of CHAMPION_KILL events close in time (within window_ms)
    and classifies them as teamfights.
    Adds spatial labels like 'top river', 'mid lane', 'bot jungle'.
    """

    # ---------------------------
    # Helper: classify a position
    # ---------------------------
    def get_region(x: int, y: int) -> str:
        if x is None or y is None:
            return "unknown"

        # approximate zone classification (SR map)
        if x > 11000 and y < 4000:
            return "top lane"
        elif x < 4000 and y > 11000:
            return "bot lane"
        elif 5000 <= x <= 10000 and 5000 <= y <= 10000:
            return "mid lane"
        elif 9000 <= x <= 13000 and 4000 <= y <= 9000:
            return "top river"
        elif 4000 <= x <= 9000 and 9000 <= y <= 13000:
            return "bot river"
        else:
            return "jungle"

    # ---------------------------
    # Collect all CHAMPION_KILL events
    # ---------------------------
    kills = []
    for frame in info["frames"]:
        for e in frame.get("events", []):
            if e["type"] == "CHAMPION_KILL":
                pos = e.get("position", {})
                kills.append({
                    "timestamp": e["timestamp"],
                    "killer": e.get("killerId"),
                    "victim": e.get("victimId"),
                    "assisters": e.get("assistingParticipantIds", []),
                    "position": pos,
                    "x": pos.get("x"),
                    "y": pos.get("y"),
                    "region": get_region(pos.get("x"), pos.get("y"))
                })

    kills.sort(key=lambda x: x["timestamp"])
    if not kills:
        return []

    teamfights = []
    cluster = [kills[0]]

    # Helper to finalize a fight cluster
    def finalize_cluster(cluster):
        all_participants = set()
        for c in cluster:
            all_participants.update([c["killer"], c["victim"], *c["assisters"]])

        blue_team = [p for p in all_participants if p and p <= 5]
        red_team = [p for p in all_participants if p and p >= 6]

        blue_kills = sum(1 for c in cluster if c["killer"] and c["killer"] <= 5)
        red_kills = sum(1 for c in cluster if c["killer"] and c["killer"] >= 6)
        blue_deaths = sum(1 for c in cluster if c["victim"] and c["victim"] <= 5)
        red_deaths = sum(1 for c in cluster if c["victim"] and c["victim"] >= 6)

        # Average fight location
        avg_x = sum(c["x"] or 0 for c in cluster if c["x"]) / max(1, sum(1 for c in cluster if c["x"]))
        avg_y = sum(c["y"] or 0 for c in cluster if c["y"]) / max(1, sum(1 for c in cluster if c["y"]))
        fight_region = get_region(avg_x, avg_y)

        winner_team = 100 if blue_kills > red_kills else 200 if red_kills > blue_kills else None

        return {
            "start": cluster[0]["timestamp"],
            "end": cluster[-1]["timestamp"],
            "duration_ms": cluster[-1]["timestamp"] - cluster[0]["timestamp"],
            "region": fight_region,
            "kills": [
                {
                    "timestamp": k["timestamp"],
                    "killer": k["killer"],
                    "victim": k["victim"],
                    "assisters": k["assisters"],
                    "region": k["region"],
                    "position": k["position"]
                } for k in cluster
            ],
            "participants": list(all_participants),
            "blue_team_participants": blue_team,
            "red_team_participants": red_team,
            "blue_kills": blue_kills,
            "red_kills": red_kills,
            "blue_deaths": blue_deaths,
            "red_deaths": red_deaths,
            "winner_team": winner_team
        }

    # ---------------------------
    # Cluster kills into fights
    # ---------------------------
    for i in range(1, len(kills)):
        if kills[i]["timestamp"] - kills[i - 1]["timestamp"] <= window_ms:
            cluster.append(kills[i])
        else:
            if len(cluster) >= 2:
                teamfights.append(finalize_cluster(cluster))
            cluster = [kills[i]]

    if len(cluster) >= 2:
        teamfights.append(finalize_cluster(cluster))

    return teamfights

    
# =======================================================
# LANE ADVANTAGE (simple early-game diff)
# =======================================================
def lane_advantage(df: pd.DataFrame, pid: int, minute_cutoff: int = 10) -> Dict[str, Any]:
    if pid is None or df.empty:
        return {}
    player_team = df[df["participantId"] == pid]["team"].iloc[0]
    opp_pid = pid + 5 if player_team == 100 else pid - 5
    early = df[df["timestamp"] <= minute_cutoff * 60000]
    if early.empty:
        return {}
    player = early[early["participantId"] == pid]
    opp = early[early["participantId"] == opp_pid]
    if player.empty or opp.empty:
        return {}
    return {
        "gold_diff_10": int(player["gold"].iloc[-1] - opp["gold"].iloc[-1]),
        "xp_diff_10": int(player["xp"].iloc[-1] - opp["xp"].iloc[-1]),
        "cs_diff_10": int(player["cs"].iloc[-1] - opp["cs"].iloc[-1])
    }


# =======================================================
# TEAM GOLD ADVANTAGE
# =======================================================
def team_gold_advantage(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    gold_summary = (
        df.groupby(["timestamp", "team"])["gold"]
        .sum()
        .unstack()
        .fillna(0)
    )
    output = []
    for ts, row in gold_summary.iterrows():
        diff = row.get(100, 0) - row.get(200, 0)
        output.append({
            "timestamp": ts,
            "team100": int(row.get(100, 0)),
            "team200": int(row.get(200, 0)),
            "diff": int(diff)
        })
    return output


# =======================================================
# DEATHS
# =======================================================
def detect_deaths(info: Dict[str, Any], pid: int = None) -> List[Dict[str, Any]]:
    deaths = []
    for frame in info["frames"]:
        for e in frame.get("events", []):
            if e["type"] == "CHAMPION_KILL" and (pid is None or e.get("victimId") == pid):
                deaths.append({
                    "timestamp": e["timestamp"],
                    "killerId": e.get("killerId"),
                    "victimId": e.get("victimId")
                })
    return deaths


# =======================================================
# OBJECTIVE TAKEDOWNS (EPIC OBJECTIVES)
# =======================================================
def detect_objectives(info: Dict[str, Any], team_id: int = None) -> List[Dict[str, Any]]:
    """
    Detect all major objective takedowns in the timeline.
    Notes Dragon, Rift Herald, Baron, Grub, and Atakhan kills,
    with killer team and timestamp.
    """
    objs = []
    for frame in info["frames"]:
        for e in frame.get("events", []):
            # Only consider elite monster kills
            if e["type"] == "ELITE_MONSTER_KILL":
                monster = e.get("monsterType", "")
                sub = e.get("monsterSubType", None)
                killer_team = e.get("killerTeamId")

                # Normalize "epic" monsters only
                if monster in {"DRAGON", "RIFTHERALD", "BARON_NASHOR", "ATAKHAN", "GRUB"}:
                    objs.append({
                        "timestamp": e["timestamp"],
                        "team": killer_team,
                        "monsterType": monster,
                        "monsterSubType": sub,
                        "position": e.get("position", {}),
                        "killerId": e.get("killerId"),
                        "region": classify_objective_region(e.get("monsterType"), e.get("position", {}))
                    })

            # Also count BUILDING_KILL for inhibitors/towers if desired
            elif e["type"] == "BUILDING_KILL":
                if e.get("buildingType") in {"TOWER_BUILDING", "INHIBITOR_BUILDING"}:
                    objs.append({
                        "timestamp": e["timestamp"],
                        "team": e.get("killerTeamId"),
                        "monsterType": e["buildingType"],
                        "laneType": e.get("laneType"),
                        "position": e.get("position", {}),
                        "region": classify_objective_region(e["buildingType"], e.get("position", {}))
                    })

    return objs


# =======================================================
# HELPER: CLASSIFY OBJECTIVE REGION
# =======================================================
def classify_objective_region(monster_type: str, pos: Dict[str, Any]) -> str:
    """
    Label objectives by map region for clarity (e.g., 'bot river', 'top river', etc.)
    """
    if not pos:
        pos = {}
    x = pos.get("x", 0)
    y = pos.get("y", 0)

    # Explicit markers for known objective positions
    if monster_type == "BARON_NASHOR" or monster_type == "ATAKHAN":
        return "baron pit (top river)"
    if monster_type == "RIFTHERALD":
        return "rift herald pit (top river)"
    if monster_type == "DRAGON" or monster_type == "GRUB":
        return "dragon pit (bot river)"
    if monster_type in {"TOWER_BUILDING", "INHIBITOR_BUILDING"}:
        if y < 7000:
            return "top lane"
        elif y > 9000:
            return "bot lane"
        else:
            return "mid lane"

    # Fallback: approximate region by coordinates
    if x > 11000 and y < 4000:
        return "top side"
    elif x < 4000 and y > 11000:
        return "bot side"
    elif 5000 <= x <= 10000 and 5000 <= y <= 10000:
        return "mid"
    else:
        return "jungle"

# =======================================================
# PER-PLAYER PER-MINUTE PERFORMANCE
# =======================================================
def per_player_minute_stats(info: Dict[str, Any]) -> List[Dict[str, Any]]:
    frames = info["frames"]

    # --- frame-based stats ---
    frame_rows = []
    for frame in frames:
        ts = frame["timestamp"]
        minute = int(ts / 60000)
        for pid_str, pdata in frame["participantFrames"].items():
            pid_int = int(pid_str)
            frame_rows.append({
                "minute": minute,
                "participantId": pid_int,
                "gold": pdata["totalGold"],
                "cs": pdata["minionsKilled"] + pdata["jungleMinionsKilled"],
                "xp": pdata["xp"]
            })
    df = pd.DataFrame(frame_rows)
    df = df.groupby(["minute", "participantId"]).mean().reset_index()

    # --- event-based kills/deaths/assists ---
    kills = defaultdict(lambda: defaultdict(int))
    deaths = defaultdict(lambda: defaultdict(int))
    assists = defaultdict(lambda: defaultdict(int))
    for frame in frames:
        minute = int(frame["timestamp"] / 60000)
        for e in frame.get("events", []):
            if e["type"] != "CHAMPION_KILL":
                continue
            killer = e.get("killerId")
            victim = e.get("victimId")
            assist_list = e.get("assistingParticipantIds", [])
            if killer:
                kills[killer][minute] += 1
            if victim:
                deaths[victim][minute] += 1
            for a in assist_list:
                assists[a][minute] += 1

    df["kills"] = df.apply(lambda r: kills[r["participantId"]].get(r["minute"], 0), axis=1)
    df["deaths"] = df.apply(lambda r: deaths[r["participantId"]].get(r["minute"], 0), axis=1)
    df["assists"] = df.apply(lambda r: assists[r["participantId"]].get(r["minute"], 0), axis=1)

    # --- lane-relative diffs ---
    diffs = []
    for lane_pair in range(1, 6):
        blue_id = lane_pair
        red_id = lane_pair + 5
        blue_lane = df[df["participantId"] == blue_id].set_index("minute")
        red_lane = df[df["participantId"] == red_id].set_index("minute")
        merged = blue_lane.join(red_lane, lsuffix="_b", rsuffix="_r", how="outer").fillna(method="ffill").fillna(0)
        for minute, row in merged.iterrows():
            gold_diff = row["gold_b"] - row["gold_r"]
            cs_diff = row["cs_b"] - row["cs_r"]
            leader = 100 if gold_diff > 0 else 200 if gold_diff < 0 else None
            diffs.append({
                "minute": int(minute),
                "lane": lane_pair,
                "blue_id": blue_id,
                "red_id": red_id,
                "gold_diff": int(gold_diff),
                "cs_diff": int(cs_diff),
                "leader_team": leader
            })

    # --- merge back ---
    all_rows = []
    lane_map = {1: 6, 2: 7, 3: 8, 4: 9, 5: 10}
    for _, row in df.iterrows():
        pid = int(row["participantId"])
        minute = int(row["minute"])
        lane = pid if pid <= 5 else pid - 5
        lane_diff = next((d for d in diffs if d["minute"] == minute and (d["blue_id"] == pid or d["red_id"] == pid)), None)
        all_rows.append({
            "minute": minute,
            "participantId": pid,
            "gold": int(row["gold"]),
            "cs": int(row["cs"]),
            "xp": int(row["xp"]),
            "kills": int(row["kills"]),
            "deaths": int(row["deaths"]),
            "assists": int(row["assists"]),
            "gold_diff_vs_lane": int(lane_diff["gold_diff"]) if lane_diff else 0,
            "cs_diff_vs_lane": int(lane_diff["cs_diff"]) if lane_diff else 0,
            "leader_team_lane": lane_diff["leader_team"] if lane_diff else None
        })
    return all_rows


# =======================================================
# MOMENTUM / COMEBACK DETECTION
# =======================================================
def detect_momentum(per_player_minute: List[Dict[str, Any]], gold_advantage: List[Dict[str, Any]],
                    lane_threshold: int = 200, team_threshold: int = 1000) -> Dict[str, Any]:
    momentum = {
        "lane_swings": [],
        "team_swings": []
    }

    # --- lane momentum ---
    lane_history = defaultdict(lambda: None)
    for record in per_player_minute:
        lane = record["participantId"] if record["participantId"] <= 5 else record["participantId"] - 5
        diff = record["gold_diff_vs_lane"]
        leader = record["leader_team_lane"]
        minute = record["minute"]

        prev = lane_history[lane]
        if prev is not None:
            prev_diff = prev["gold_diff_vs_lane"]
            if (diff > 0 > prev_diff) or (diff < 0 < prev_diff) or abs(diff - prev_diff) > lane_threshold:
                momentum["lane_swings"].append({
                    "minute": minute,
                    "lane": lane,
                    "from_diff": int(prev_diff),
                    "to_diff": int(diff),
                    "new_leader": leader
                })
        lane_history[lane] = record

    # --- team momentum ---
    if gold_advantage:
        prev_diff = gold_advantage[0]["diff"]
        for i in range(1, len(gold_advantage)):
            diff = gold_advantage[i]["diff"]
            minute = int(gold_advantage[i]["timestamp"] / 60000)
            if (diff > 0 > prev_diff) or (diff < 0 < prev_diff) or abs(diff - prev_diff) > team_threshold:
                momentum["team_swings"].append({
                    "minute": minute,
                    "from_diff": int(prev_diff),
                    "to_diff": int(diff),
                    "new_leader_team": 100 if diff > 0 else 200 if diff < 0 else None
                })
            prev_diff = diff

    return momentum



# =======================================================
# PLAYER SUMMARY BUILDER (from build_player_summary)
# =======================================================
def build_player_summary(full_analysis_path: str, player_id: int = 1, output_path: str = "player_summary.json"):
    with open(full_analysis_path, "r") as f:
        data = json.load(f)

    lane_stats = [
        {
            "minute": p["minute"],
            "gold": p["gold"],
            "cs": p["cs"],
            "xp": p["xp"],
            "kills": p["kills"],
            "deaths": p["deaths"],
            "assists": p["assists"],
            "gold_diff_vs_lane": p["gold_diff_vs_lane"],
            "cs_diff_vs_lane": p["cs_diff_vs_lane"],
            "leader_team_lane": p["leader_team_lane"]
        }
        for p in data.get("per_player_minute", [])
        if p["participantId"] == player_id
    ]

    avg_gold_diff = int(mean([p["gold_diff_vs_lane"] for p in lane_stats])) if lane_stats else 0
    avg_cs_diff = int(mean([p["cs_diff_vs_lane"] for p in lane_stats])) if lane_stats else 0
    lane_adv_10 = next((p["gold_diff_vs_lane"] for p in lane_stats if p["minute"] == 10), None)
    cs_adv_10 = next((p["cs_diff_vs_lane"] for p in lane_stats if p["minute"] == 10), None)

    relevant_objectives = [
        {
            "minute": round(o["timestamp"] / 60000, 1),
            "type": o["monsterType"],
            "team": o["team"],
            "region": o["region"]
        }
        for o in data.get("objectives", [])
        if "river" in o.get("region", "") or "lane" in o.get("region", "")
    ]

    lane_index = player_id if player_id <= 5 else player_id - 5
    momentum = [
        s for s in data.get("momentum", {}).get("lane_swings", [])
        if s["lane"] == lane_index
    ]

    kills = sum(p["kills"] for p in lane_stats)
    deaths = sum(p["deaths"] for p in lane_stats)
    assists = sum(p["assists"] for p in lane_stats)

    summary = {
        "match_id": data.get("matchId", "unknown"),
        "player_id": player_id,
        "class": "Fighter",   # Replace later with D&D mapping
        "lane": lane_index,
        "totals": {"kills": kills, "deaths": deaths, "assists": assists},
        "lane_summary": {
            "avg_gold_diff": avg_gold_diff,
            "avg_cs_diff": avg_cs_diff,
            "lane_advantage_10": lane_adv_10,
            "cs_advantage_10": cs_adv_10
        },
        "lane_trends": [
            {"minute": p["minute"], "gold_diff": p["gold_diff_vs_lane"], "cs_diff": p["cs_diff_vs_lane"]}
            for p in lane_stats
        ],
        "objectives": relevant_objectives,
        "momentum": momentum
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Summary]  Saved player {player_id} summary → {output_path}")


# =======================================================
# MAIN ENTRY POINT
# =======================================================
if __name__ == "__main__":
    if len(sys.argv) >= 3:
        region, match_id, puuid= sys.argv[1], sys.argv[2], sys.argv[3]
    else:
        region, match_id, puuid = "euw1", "EUW1_7553099692", "qi_odtdjcRV0HwE29sGPLEO17niJg-_8LV8P1HAYbmrZ1BG4BcVcLmjqV8WNHq433qP6V6KQ8EP4aA"

    asyncio.run(grab_timeline(region, match_id,puuid))
