import requests
import pandas as pd
import os

#  Paths 
DATA_PATH = "../../data/normalized_matches.csv"

#  Load normalized match data 
print(" Loading normalized match data")
df = pd.read_csv(DATA_PATH)
print(f" Loaded {len(df)} rows with {len(df.columns)} columns")

# Ensure champion column exists
if "champion" not in df.columns:
    raise ValueError(" Missing 'champion' column in normalized_matches.csv")

#  Fetch latest patch version 
version_url = "https://ddragon.leagueoflegends.com/api/versions.json"
latest_patch = requests.get(version_url).json()[0]
print(f" Latest DDragon version: {latest_patch}")

#  Get champion data 
champ_url = f"https://ddragon.leagueoflegends.com/cdn/{latest_patch}/data/en_US/champion.json"
champ_data = requests.get(champ_url).json()["data"]

#  Extract relevant champion info 
champ_rows = []
for champ_name, champ_info in champ_data.items():
    info = champ_info.get("info", {})
    tags = champ_info.get("tags", [])
    champ_rows.append({
        "champion": champ_info.get("id", champ_name),
        "champ_key": champ_info.get("key"),
        "attack": info.get("attack", 0),
        "defense": info.get("defense", 0),
        "magic": info.get("magic", 0),
        "difficulty": info.get("difficulty", 0),
        "tags": ",".join(tags) if tags else None
    })

champ_df = pd.DataFrame(champ_rows)
print(f" Parsed {len(champ_df)} champions from DDragon")

#  Normalize names for join 
champ_df["champion"] = champ_df["champion"].astype(str).str.strip()
df["champion"] = df["champion"].astype(str).str.strip()

#  Merge using INNER JOIN (only champions present in match data) 
merged_df = pd.merge(df, champ_df, on="champion", how="left")
print(f" Merged dataset shape: {merged_df.shape}")

#  Save merged dataset 
merged_df.to_csv(DATA_PATH, index=False)
print(f" Saved enriched dataset with champion info to {DATA_PATH}")

# Quick sanity check
print(merged_df[["champion", "champ_key", "attack", "defense", "magic", "difficulty", "tags"]].head(10))
