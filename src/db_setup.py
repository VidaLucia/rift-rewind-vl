import sqlite3

DB_PATH = "riot_data.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Players table
    c.execute("""
    CREATE TABLE IF NOT EXISTS players (
        puuid TEXT PRIMARY KEY,
        region TEXT
    )
    """)

    # Matches table
    c.execute("""
    CREATE TABLE IF NOT EXISTS matches (
        matchId TEXT PRIMARY KEY,
        region TEXT,
        details_json TEXT,
        timeline_json TEXT
    )
    """)

    # Player-Match link table
    c.execute("""
    CREATE TABLE IF NOT EXISTS player_matches (
        puuid TEXT,
        matchId TEXT,
        PRIMARY KEY (puuid, matchId),
        FOREIGN KEY (puuid) REFERENCES players(puuid),
        FOREIGN KEY (matchId) REFERENCES matches(matchId)
    )
    """)

    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")