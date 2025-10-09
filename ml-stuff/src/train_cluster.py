import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import json
from joblib import dump
import umap.umap_ as umap
import os
import time

# =====================================================
# CONFIG
# =====================================================
DATA_PATH = "../../data/normalized_matches.csv"
MODEL_DIR = "../../models"
OUT_DIR = "../../data/player_analysis"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

best_k = 26
DISCOVERY_UMAP = False
DISCOVERY = False

# =====================================================
# LOAD DATA
# =====================================================
print(f"Loading data from {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows × {len(df.columns)} columns.")

# CLEANING
df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

# EXCLUDE ITEMS
item_cols = [c for c in df.columns if c.lower().startswith("item")]
if item_cols:
    print(f"Dropping {len(item_cols)} item columns: {item_cols[:6]}...")
    df = df.drop(columns=item_cols, errors="ignore")

# ENCODE TAGS
if "tags" in df.columns:
    print("Encoding champion tags...")
    tag_dummies = df["tags"].fillna("").str.get_dummies(sep=",").add_prefix("tag_")
    df = pd.concat([df, tag_dummies], axis=1)
    print(f"Added {len(tag_dummies.columns)} tag columns.")
else:
    print("No tag column found.")

# ENCODE ROLE
if "role" in df.columns:
    print("Encoding role column numerically...")
    df["role"] = df["role"].astype(str).str.upper().replace({"UTILITY": "SUPPORT"})
    role_map = {"TOP": 1, "JUNGLE": 2, "MIDDLE": 3, "BOTTOM": 4, "SUPPORT": 5}
    invalid = ~df["role"].isin(role_map.keys())
    print(f"Dropping {invalid.sum()} invalid roles.")
    df = df[~invalid]
    df["role"] = df["role"].map(role_map)
else:
    df["role"] = 0
    print("Role column missing, filled with 0.")

# FILTER EMPTY PERK DATA
before = len(df)
df = df[~((df["perk_offense"] == 0) & (df["perk_flex"] == 0) & (df["perk_defense"] == 0))]
print(f"Filtered {before - len(df)} rows with all-zero perks; remaining {len(df)}.")

# =====================================================
# BALANCE DATA BY ROLE
# =====================================================
max_per_role = 20000
df = (
    df.groupby("role", group_keys=False)
    .apply(lambda x: x.sample(min(len(x), max_per_role), random_state=42))
    .reset_index(drop=True)
)
print(f"Balanced dataset by role — new size: {len(df)}")

# =====================================================
# DERIVED FEATURES
# =====================================================
print("Generating derived features...")
df["kp_rate"] = df["ch_killParticipation"] / 100
df["damage_share"] = df["ch_teamDamagePercentage"] / 100
df["gold_efficiency"] = df["ch_goldPerMinute"] / df["duration"].clip(lower=1)
df["survivability_ratio"] = df["kills"] / df["deaths"].clip(lower=1)
df["objective_focus"] = (
    df["ch_turretTakedowns"] + df["ch_baronTakedowns"] + df["ch_dragonTakedowns"]
) / 3
df["kpm"] = df["kills"] / df["duration"].clip(lower=1)
df["dpm"] = df["damage"] / df["duration"].clip(lower=1)
df["apm"] = df["assists"] / df["duration"].clip(lower=1)
df["cspm"] = df["cs"] / df["duration"].clip(lower=1)
df["vision_efficiency"] = df["ch_visionScorePerMinute"] / (df["ch_wardTakedowns"] + 1)

# =====================================================
# FEATURE SELECTION
# =====================================================
features = [
    "kills", "deaths", "assists", "damage", "cs", "role",
    "kp_rate", "damage_share", "gold_efficiency",
    "objective_focus", "survivability_ratio", "vision_efficiency",
    "primarystyle_id", "primarystyle_perk1", "primarystyle_perk2", "primarystyle_perk3",
    "substyle_id", "substyle_perk1", "substyle_perk2",
    "ch_kda", "ch_killingSprees", "ch_damagePerMinute",
    "ch_teamDamagePercentage", "ch_killParticipation",
    "ch_turretTakedowns", "ch_baronTakedowns", "ch_dragonTakedowns",
    "ch_enemyJungleMonsterKills",
    "ch_goldPerMinute", "ch_laningPhaseGoldExpAdvantage",
    "ch_maxCsAdvantageOnLaneOpponent",
    "ch_deathsByEnemyChamps", "ch_damageTakenOnTeamPercentage",
    "ch_survivedSingleDigitHpCount",
    "ch_effectiveHealAndShielding", "ch_saveAllyFromDeath",
    "ch_immobilizeAndKillWithAlly", "ch_visionScorePerMinute",
    "kpm", "dpm", "apm", "cspm",
    "_id", "attack", "defense", "magic", "difficulty",
]
features.extend([c for c in df.columns if c.startswith("tag_")])

X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)

# =====================================================
# FEATURE WEIGHTING
# =====================================================
feature_weights = {
    "kpm": 2.0, "dpm": 2.0, "apm": 1.8, "cspm": 1.6,
    "ch_damagePerMinute": 1.5, "ch_killParticipation": 1.4,
    "ch_goldPerMinute": 1.3, "ch_turretTakedowns": 1.3,
    "ch_visionScorePerMinute": 1.2, "objective_focus": 1.2,
    "role": 1.1, "damage_share": 1.2, "kp_rate": 1.1,
    "gold_efficiency": 1.0, "survivability_ratio": 0.9,
    "vision_efficiency": 0.9, "attack": 0.6,
    "defense": 0.6, "magic": 0.6, "difficulty": 0.6,
}
for feat, weight in feature_weights.items():
    if feat in X.columns:
        X[feat] *= weight

# =====================================================
# SCALING
# =====================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Scaled features.")

# =====================================================
# UMAP DIMENSIONALITY REDUCTION
# =====================================================
best_umap_params = {
    "n_neighbors": 40,
    "min_dist": 0.15,
    "n_components": 10,
}
reducer = umap.UMAP(**best_umap_params, metric="euclidean", random_state=42)
X_embed = reducer.fit_transform(X_scaled)
print(f"UMAP completed with shape: {X_embed.shape}")

# =====================================================
# SAVE METADATA
# =====================================================
with open(os.path.join(MODEL_DIR, "umap_metadata.json"), "w") as f:
    json.dump({"best_params": best_umap_params}, f, indent=2)

# =====================================================
# CLUSTERING
# =====================================================
kmeans = MiniBatchKMeans(
    n_clusters=best_k,
    random_state=42,
    batch_size=4096,
    max_iter=200,
    n_init=10
)
df["cluster"] = kmeans.fit_predict(X_embed)

# QUICK SILHOUETTE CHECK
sample_idx = np.random.choice(len(X_embed), min(10000, len(X_embed)), replace=False)
sil_score = silhouette_score(X_embed[sample_idx], kmeans.labels_[sample_idx])
print(f"Quick Silhouette Score: {sil_score:.4f}")

# =====================================================
# SAVE ARTIFACTS
# =====================================================
dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
dump(reducer, os.path.join(MODEL_DIR, "umap_reducer.pkl"))
dump(kmeans, os.path.join(MODEL_DIR, "kmeans_model.pkl"))
print(f"Saved models to {MODEL_DIR}")

# =====================================================
# CLUSTER SUMMARY
# =====================================================
counts = df["cluster"].value_counts().sort_index()
summary_df = pd.DataFrame({
    "cluster": counts.index,
    "count": counts.values,
})
summary_df["percentage"] = summary_df["count"] / len(df) * 100

summary_path = os.path.join(OUT_DIR, "cluster_sizes.csv")
summary_df.to_csv(summary_path, index=False)

print("\nCluster Summary Table:")
print("=" * 50)
total = len(df)
for _, row in summary_df.iterrows():
    c, n, pct = int(row["cluster"]), int(row["count"]), row["percentage"]
    bar = "█" * int(pct / 2)
    print(f"Cluster {c:>2}: {n:>6} samples ({pct:5.2f}%) {bar}")
print("=" * 50)
print(f"Cluster distribution saved to: {summary_path}")

# =====================================================
# SUMMARY
# =====================================================
print("\nTRAINING SUMMARY")
print(f"Used K={best_k}")
print(f"Silhouette Score: {sil_score:.4f}")
print(f"Model saved in: {MODEL_DIR}")
