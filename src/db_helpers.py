import sqlite3, json
from db_setup import DB_PATH

def save_player(puuid, region):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT OR IGNORE INTO players (puuid, region)
        VALUES (?, ?)
    """, (puuid, region))
    conn.commit()
    conn.close()

def save_match(matchId, region, details, timeline):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO matches (matchId, region, details_json, timeline_json)
        VALUES (?, ?, ?, ?)
    """, (matchId, region, json.dumps(details), json.dumps(timeline)))
    conn.commit()
    conn.close()

def save_player_match(puuid, matchId):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT OR IGNORE INTO player_matches (puuid, matchId)
        VALUES (?, ?)
    """, (puuid, matchId))
    conn.commit()
    conn.close()
