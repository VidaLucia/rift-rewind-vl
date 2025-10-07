import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import umap.umap_ as umap
from joblib import dump
import os

# === CONFIG ===
DATA_PATH = "../../data/normalized_matches.csv"
SCALER_PATH = "../../models/scaler.pkl"
UMAP_PATH = "../../models/umap_reducer.pkl"
MODEL_PATH = "../../models/kmeans_model.pkl"

# === LOAD DATA ===
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows with {len(df.columns)} columns.")

# === CLEAN NaNs / INF ===
df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

# === TAG ENCODING ===
if "tags" in df.columns:
    print("Encoding champion tags...")
    tag_dummies = (
        df["tags"]
        .fillna("")
        .str.get_dummies(sep=",")
        .add_prefix("tag_")
    )
    df = pd.concat([df, tag_dummies], axis=1)
    print(f"✅ Added {len(tag_dummies.columns)} tag columns: {list(tag_dummies.columns)}")

# === ROLE ENCODING ===
if "role" in df.columns:
    print("Encoding role column numerically...")
    df["role"] = (
        df["role"]
        .astype(str)
        .str.upper()
        .replace({"UTILITY": "SUPPORT"})
    )
    role_map = {"TOP": 1, "JUNGLE": 2, "MIDDLE": 3, "BOTTOM": 4, "SUPPORT": 5}
    invalid_roles = ~df["role"].isin(role_map.keys())
    if invalid_roles.any():
        print(f"⚠️ Dropping {invalid_roles.sum()} rows with invalid roles")
        df = df[~invalid_roles]
    df["role"] = df["role"].map(role_map).astype(int)
    print(f"✅ Role encoding applied: {role_map}")

# === PERK FILTER ===
if {"perk_offense", "perk_flex", "perk_defense"}.issubset(df.columns):
    before = len(df)
    df = df[
        ~((df["perk_offense"] == 0) &
          (df["perk_flex"] == 0) &
          (df["perk_defense"] == 0))
    ]
    print(f"Filtered {before - len(df)} rows with all-zero perks; remaining: {len(df)}")

# === NORMALIZE SOME QUANTITATIVE METRICS ===
if "ch_teamDamagePercentage" in df.columns:
    df["ch_teamDamagePercentage"] /= 100
if "ch_killParticipation" in df.columns:
    df["ch_killParticipation"] /= 100
if "ch_visionScorePerMinute" in df.columns:
    df["ch_visionScorePerMinute"] = df["ch_visionScorePerMinute"].clip(upper=10)
if "duration" in df.columns:
    df["duration"] = df["duration"].clip(lower=60)

# === DERIVED FEATURES ===
print("\nGenerating derived features...")
df["kp_rate"] = df["ch_killParticipation"]
df["damage_share"] = df["ch_teamDamagePercentage"]
df["gold_efficiency"] = df["ch_goldPerMinute"] / df["duration"].clip(lower=1)
df["survivability_ratio"] = df["kills"] / df["deaths"].clip(lower=1)
df["objective_focus"] = (
    df["ch_turretTakedowns"] + df["ch_baronTakedowns"] + df["ch_dragonTakedowns"]
) / 3
df["kpm"] = df["kills"] / df["duration"].clip(lower=1)
df["dpm"] = df["damage"] / df["duration"].clip(lower=1)
df["apm"] = df["assists"] / df["duration"].clip(lower=1)
df["cspm"] = df["cs"] / df["duration"].clip(lower=1)
df["vision_efficiency"] = df["ch_visionScorePerMinute"] / (df.get("ch_wardTakedowns", 0) + 1)

# === FEATURES ===
features = [
    "kills", "deaths", "assists", "damage",
    "cs", "role",
    "kp_rate", "damage_share", "gold_efficiency",
    "objective_focus", "survivability_ratio", "vision_efficiency",
    "primarystyle_id", "primarystyle_perk1", "primarystyle_perk2", "primarystyle_perk3",
    "substyle_id", "substyle_perk1", "substyle_perk2",
    "ch_kda", "ch_killingSprees", "ch_damagePerMinute", "ch_teamDamagePercentage",
    "ch_killParticipation", "ch_turretTakedowns", "ch_baronTakedowns", "ch_dragonTakedowns",
    "ch_enemyJungleMonsterKills", "ch_goldPerMinute", "ch_laningPhaseGoldExpAdvantage",
    "ch_maxCsAdvantageOnLaneOpponent", "ch_deathsByEnemyChamps",
    "ch_damageTakenOnTeamPercentage", "ch_survivedSingleDigitHpCount",
    "ch_effectiveHealAndShielding", "ch_saveAllyFromDeath", "ch_immobilizeAndKillWithAlly",
    "ch_visionScorePerMinute", "kpm", "dpm", "apm", "cspm",
    "_id", "attack", "defense", "magic", "difficulty"
]

# Fill missing features
for feat in features:
    if feat not in df.columns:
        df[feat] = 0
        print(f"⚠️ Added missing feature '{feat}' (filled with 0)")

X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
print(f"\n✅ Prepared training matrix: {X.shape}")

# === SCALER ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✅ Scaled features")

# === UMAP REDUCTION ===
print("Running UMAP dimensionality reduction...")
reducer = umap.UMAP(
    n_components=10,
    n_neighbors=30,
    min_dist=0.0,
    spread=1.5,
    random_state=42
)
X_embed = reducer.fit_transform(X_scaled)
print(f"✅ UMAP complete. Shape: {X_embed.shape}")

# === KMEANS TRAINING ===
print("Training KMeans clustering model...")
kmeans = KMeans(
    n_clusters=18,
    n_init=50,
    max_iter=600,
    random_state=42
)
labels = kmeans.fit_predict(X_embed)
print("✅ KMeans training done.")

# === CLUSTER SUMMARY ===
unique, counts = np.unique(labels, return_counts=True)
print("\n=== Cluster Distribution ===")
for c, n in zip(unique, counts):
    pct = n / len(labels) * 100
    print(f"Cluster {c:<2}: {n:>5} samples ({pct:.2f}%)")

# === SAVE MODELS ===
os.makedirs("../../models", exist_ok=True)
dump(scaler, SCALER_PATH)
dump(reducer, UMAP_PATH)
dump(kmeans, MODEL_PATH)
print("\n💾 Saved models to ../../models")

# === OPTIONAL QUALITY METRIC ===
try:
    from sklearn.metrics import silhouette_score
    sil = silhouette_score(X_embed, labels)
    print(f"🧭 Silhouette Score: {sil:.3f}")
except Exception as e:
    print(f"Silhouette calculation skipped: {e}")
