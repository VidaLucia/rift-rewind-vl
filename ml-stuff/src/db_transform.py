import sqlite3
import pandas as pd
import json
import os

DB_FILE = "../../data/matches.db"   # one level up if DB is in project root
OUTPUT_DIR = "../../data"      # saves into data folder at project root

def build_feature():
    conn = sqlite3.connect(DB_FILE)

    query = """
    SELECT p.*, m.game_duration
    FROM participants p
    JOIN matches m ON p.match_id = m.match_id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # --- CLEANING ---

    # Drop inconsistent name fields if not needed
    df = df.drop(columns=["summoner_name"], errors="ignore")

    # Remove swarm mode champs
    df = df[~df["champion_name"].str.startswith("Strawberry_", na=False)]

    # Normalize blanks
    def normalize_field(val):
        if pd.isna(val):
            return None
        val = str(val).strip().upper()
        if val in ["NONE", ""]:
            return None
        return val

    df["team_position"] = df["team_position"].apply(normalize_field)
    df["lane"] = df["lane"].apply(normalize_field)
    df["role"] = df["role"].apply(normalize_field)

    # Resolve consistent position
    def resolve_position(row):
        if row["team_position"]:
            return row["team_position"]
        elif row["lane"]:
            return row["lane"]
        elif row["role"]:
            return row["role"]
        else:
            return "UNKNOWN"

    df["resolved_position"] = df.apply(resolve_position, axis=1)

    # --- EXPORTS ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_path = os.path.join(OUTPUT_DIR, "cleaned_participants.csv")
    json_path = os.path.join(OUTPUT_DIR, "players_matches.json")

    df.to_csv(csv_path, index=False)

    grouped = {}
    for puuid, rows in df.groupby("puuid"):
        grouped[puuid] = {"matches": rows.to_dict(orient="records")}

    with open(json_path, "w") as f:
        json.dump(grouped, f, indent=2)

    print(f" Exported {len(df)} rows")
    print(f"   CSV:  {csv_path}")
    print(f"   JSON: {json_path}")

if __name__ == "__main__":
    build_feature()