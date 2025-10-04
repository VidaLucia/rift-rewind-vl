import json
import pandas as pd
import os


def safe_json_parse(raw):
    """Parse JSON string safely, return {} if fails."""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    elif isinstance(raw, dict):
        return raw
    return {}


def load_json_to_df(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    rows = []
    # Structure: {puuid: {"matches": [match1, match2, ...]}}
    for puuid, pdata in data.items():
        matches = pdata.get("matches", [])
        for match in matches:
            row = {
                "puuid": puuid,
                "match_id": match.get("match_id"),
                "champion": match.get("champion_name"),
                "kills": match.get("kills", 0),
                "deaths": match.get("deaths", 0),
                "assists": match.get("assists", 0),
                "gold_earned": match.get("gold_earned", 0),
                "vision_score": match.get("vision_score", 0),
                "damage": match.get("total_damage_dealt_to_champions", 0),
                "cs": match.get("total_damage_dealt", 0),  # can replace with minionsKilled+neutral
                "win": match.get("win", 0),
                "duration": match.get("game_duration", 0),
                "role": match.get("resolved_position"),
            }

            # Items and spells (lists)
            row["items"] = safe_json_parse(match.get("items", "[]"))
            row["spells"] = safe_json_parse(match.get("spells", "[]"))

            # Perks
            perks_data = safe_json_parse(match.get("perks"))
            row.update(flatten_perks(perks_data))

            # Challenges
            challenges_data = safe_json_parse(match.get("challenges"))
            row.update(flatten_challenges(challenges_data))

            rows.append(row)

    return pd.DataFrame(rows)


def flatten_perks(perks):
    """Flatten perks JSON into simple columns."""
    if not perks:
        return {}

    flat = {}
    stats = perks.get("statPerks", {})
    flat["perk_offense"] = stats.get("offense", 0)
    flat["perk_flex"] = stats.get("flex", 0)
    flat["perk_defense"] = stats.get("defense", 0)

    styles = perks.get("styles", [])
    for style in styles:
        desc = style.get("description", "").lower()
        style_id = style.get("style", 0)
        flat[f"{desc}_id"] = style_id

        for i, sel in enumerate(style.get("selections", [])):
            flat[f"{desc}_perk{i+1}"] = sel.get("perk", 0)
            flat[f"{desc}_perk{i+1}_var1"] = sel.get("var1", 0)
            flat[f"{desc}_perk{i+1}_var2"] = sel.get("var2", 0)
            flat[f"{desc}_perk{i+1}_var3"] = sel.get("var3", 0)
    return flat


def flatten_challenges(challenges):
    """Flatten challenges JSON into simple columns."""
    if not challenges:
        return {}
    flat = {}
    for key, val in challenges.items():
        if isinstance(val, (int, float)):
            flat[f"ch_{key}"] = val
    return flat


def transform_features(df):
    if df.empty:
        return df
    df["kda"] = (df["kills"] + df["assists"]) / df["deaths"].clip(lower=1)
    df["gpm"] = df["gold_earned"] / (df["duration"].clip(lower=1) / 60)
    df["dpm"] = df["damage"] / (df["duration"].clip(lower=1) / 60)
    return df

# TODO: MAYBE STORE THIS AS A CSV???
# DATA MODS  
OUTPUT_DIR = "../../data"
json_path = os.path.join(OUTPUT_DIR, "players_matches.json")

if __name__ == "__main__":
    df = load_json_to_df(json_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    df = transform_features(df)
    output_path = os.path.join(OUTPUT_DIR, "normalized_matches.csv")
    df.to_csv(output_path, index=False)
    print(df.head())
