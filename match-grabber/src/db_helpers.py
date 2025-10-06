import sqlite3, json
from db_setup import DB_FILE

def get_conn():
    return sqlite3.connect(DB_FILE)

# =
# LEAGUE STORAGE
# =
def save_league(league_data, region):
    """
    Save league + all entries (LeagueItemDTO + MiniSeriesDTO) into DB.
    """
    conn = get_conn()
    c = conn.cursor()

    league_id = league_data.get("leagueId")
    tier = league_data.get("tier")
    name = league_data.get("name")
    queue = league_data.get("queue")

    # Insert league metadata
    c.execute("""
    INSERT OR IGNORE INTO leagues (league_id, tier, name, queue, region)
    VALUES (?, ?, ?, ?, ?)
    """, (league_id, tier, name, queue, region))

    # Insert entries
    for entry in league_data.get("entries", []):
        mini_series = entry.get("miniSeries", {})
        c.execute("""
        INSERT INTO league_entries
        (league_id, region, puuid, summoner_id, summoner_name, rank, league_points,
         wins, losses, veteran, inactive, hot_streak, fresh_blood, mini_series)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            league_id,
            region,
            entry.get("puuid"),
            entry.get("summonerId"),
            entry.get("summonerName"),
            entry.get("rank"),
            entry.get("leaguePoints"),
            entry.get("wins"),
            entry.get("losses"),
            int(entry.get("veteran", False)),
            int(entry.get("inactive", False)),
            int(entry.get("hotStreak", False)),
            int(entry.get("freshBlood", False)),
            json.dumps(mini_series)
        ))

    conn.commit()
    conn.close()

# =
# PLAYER STORAGE
# =
def save_player(puuid, region):
    conn = get_conn()
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO players (puuid, region) VALUES (?, ?)", (puuid, region))
    conn.commit()
    conn.close()

# =
# MATCH STORAGE
# =
def save_match(match_id, region, details, timeline):
    conn = get_conn()
    c = conn.cursor()

    meta = details.get("metadata", {})
    info = details.get("info", {})

    # Insert match metadata
    c.execute("""
    INSERT OR IGNORE INTO matches
    (match_id, region, data_version, game_creation, game_duration,
     game_end_timestamp, game_mode, game_type, game_version,
     map_id, platform_id, queue_id, tournament_code)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        meta.get("matchId"),
        region,
        meta.get("dataVersion"),
        info.get("gameCreation"),
        info.get("gameDuration"),
        info.get("gameEndTimestamp"),
        info.get("gameMode"),
        info.get("gameType"),
        info.get("gameVersion"),
        info.get("mapId"),
        info.get("platformId"),
        info.get("queueId"),
        info.get("tournamentCode"),
    ))

    # Insert participants
    for p in info.get("participants", []):
        items = json.dumps([p.get(f"item{i}") for i in range(7)])
        spells = json.dumps([p.get("summoner1Id"), p.get("summoner2Id")])
        perks = json.dumps(p.get("perks", {}))
        challenges = json.dumps(p.get("challenges", {}))

        total_minions = p.get("totalMinionsKilled", 0) or 0
        neutral_minions = p.get("neutralMinionsKilled", 0) or 0
        enemy_jungle = p.get("enemyJungleMonsterKills", 0) or 0
        champ_level = p.get("champLevel", 0) or 0
        cs_total = total_minions + neutral_minions

        c.execute("""
        INSERT INTO participants
        (match_id, puuid, summoner_id, summoner_name, champion_id, champion_name,
        team_id, participant_id, role, lane, team_position, win,
        kills, deaths, assists, gold_earned, gold_spent,
        total_damage_dealt, total_damage_dealt_to_champions,
        total_damage_taken, vision_score, wards_placed, wards_killed,
        total_minions_killed, neutral_minions_killed, enemy_jungle_monster_kills,
        cs, champ_level,
        items, spells, perks, challenges)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            match_id,
            p.get("puuid"),
            p.get("summonerId"),
            p.get("summonerName"),
            p.get("championId"),
            p.get("championName"),
            p.get("teamId"),
            p.get("participantId"),
            p.get("role"),
            p.get("lane"),
            p.get("teamPosition"),
            int(p.get("win", False)),
            p.get("kills"),
            p.get("deaths"),
            p.get("assists"),
            p.get("goldEarned"),
            p.get("goldSpent"),
            p.get("totalDamageDealt"),
            p.get("totalDamageDealtToChampions"),
            p.get("totalDamageTaken"),
            p.get("visionScore"),
            p.get("wardsPlaced"),
            p.get("wardsKilled"),
            total_minions,
            neutral_minions,
            enemy_jungle,
            cs_total,
            champ_level,
            items,
            spells,
            perks,
            challenges
        ))
    # Insert teams
    for t in info.get("teams", []):
        c.execute("""
        INSERT INTO teams (match_id, team_id, win, bans, objectives)
        VALUES (?, ?, ?, ?, ?)
        """, (
            match_id,
            t.get("teamId"),
            int(t.get("win", False)),
            json.dumps(t.get("bans", [])),
            json.dumps(t.get("objectives", {}))
        ))

    # Insert timeline data (optional)
    if timeline:
        frames = timeline.get("info", {}).get("frames", [])
        for f in frames:
            ts = f.get("timestamp")
            c.execute("INSERT INTO frames (match_id, timestamp, frame_json) VALUES (?, ?, ?)",
                      (match_id, ts, json.dumps(f)))
            for event in f.get("events", []):
                c.execute("INSERT INTO events (match_id, timestamp, type, event_json) VALUES (?, ?, ?, ?)",
                          (match_id, event.get("timestamp"), event.get("type"), json.dumps(event)))

    conn.commit()
    conn.close()

# =
# PLAYER ↔ MATCH LINK
# =
def save_player_match(puuid, match_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute("INSERT INTO player_matches (puuid, match_id) VALUES (?, ?)", (puuid, match_id))
    conn.commit()
    conn.close()
