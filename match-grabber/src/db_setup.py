import sqlite3

DB_FILE = "matches.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # Players
    c.execute("""
    CREATE TABLE IF NOT EXISTS players (
        puuid TEXT PRIMARY KEY,
        region TEXT NOT NULL
    )
    """)

    # League tables
    c.execute("""
    CREATE TABLE IF NOT EXISTS leagues (
        league_id TEXT,
        tier TEXT,
        name TEXT,
        queue TEXT,
        region TEXT,
        PRIMARY KEY (league_id, region)
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS league_entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        league_id TEXT,
        region TEXT NOT NULL,
        puuid TEXT,
        summoner_id TEXT,
        summoner_name TEXT,
        rank TEXT,
        league_points INT,
        wins INT,
        losses INT,
        veteran BOOLEAN,
        inactive BOOLEAN,
        hot_streak BOOLEAN,
        fresh_blood BOOLEAN,
        mini_series TEXT, -- JSON MiniSeriesDTO
    FOREIGN KEY (league_id, region) REFERENCES leagues(league_id, region)
)
    """)

    # Matches metadata
    c.execute("""
    CREATE TABLE IF NOT EXISTS matches (
        match_id TEXT PRIMARY KEY,
        region TEXT NOT NULL,
        data_version TEXT,
        game_creation INTEGER,
        game_duration INTEGER,
        game_end_timestamp INTEGER,
        game_mode TEXT,
        game_type TEXT,
        game_version TEXT,
        map_id INT,
        platform_id TEXT,
        queue_id INT,
        tournament_code TEXT
    )
    """)

    # Participants
    c.execute("""
    CREATE TABLE IF NOT EXISTS participants (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id TEXT NOT NULL,
        puuid TEXT NOT NULL,
        summoner_id TEXT,
        summoner_name TEXT,
        champion_id INT,
        champion_name TEXT,
        team_id INT,
        participant_id INT,
        role TEXT,
        lane TEXT,
        team_position TEXT,
        win BOOLEAN,
        kills INT,
        deaths INT,
        assists INT,
        gold_earned INT,
        gold_spent INT,
        total_damage_dealt INT,
        total_damage_dealt_to_champions INT,
        total_damage_taken INT,
        vision_score INT,
        wards_placed INT,
        wards_killed INT,
        items TEXT,   -- JSON
        spells TEXT,  -- JSON
        perks TEXT,   -- JSON
        challenges TEXT, -- JSON
        FOREIGN KEY(match_id) REFERENCES matches(match_id),
        FOREIGN KEY(puuid) REFERENCES players(puuid)
    )
    """)

    # Teams
    c.execute("""
    CREATE TABLE IF NOT EXISTS teams (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id TEXT,
        team_id INT,
        win BOOLEAN,
        bans TEXT,       -- JSON of bans
        objectives TEXT, -- JSON of objectives (baron, dragon, tower, etc.)
        FOREIGN KEY(match_id) REFERENCES matches(match_id)
    )
    """)

    # Timeline frames
    c.execute("""
    CREATE TABLE IF NOT EXISTS frames (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id TEXT NOT NULL,
        timestamp INTEGER,
        frame_json TEXT,
        FOREIGN KEY(match_id) REFERENCES matches(match_id)
    )
    """)

    # Timeline events
    c.execute("""
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id TEXT NOT NULL,
        timestamp INTEGER,
        type TEXT,
        event_json TEXT,
        FOREIGN KEY(match_id) REFERENCES matches(match_id)
    )
    """)

    # Link table
    c.execute("""
    CREATE TABLE IF NOT EXISTS player_matches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        puuid TEXT NOT NULL,
        match_id TEXT NOT NULL,
        FOREIGN KEY(puuid) REFERENCES players(puuid),
        FOREIGN KEY(match_id) REFERENCES matches(match_id)
    )
    """)

    conn.commit()
    conn.close()