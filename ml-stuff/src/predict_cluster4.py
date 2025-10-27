# %%
import pandas as pd
import numpy as np
from joblib import load
import os
import json
import umap
from sklearn.preprocessing import StandardScaler

# =====================================================
# CONFIG
# =====================================================
DATA_PATH = "../../data/players/lulululu04#lulu_data.csv"
UMAP_PATH = "../../models/umap_reducer.pkl"
MODEL_PATH = "../../models/kmeans_model.pkl"
SCALER_PATH = "../../models/scaler.pkl"
FEATURES_PATH = "../../models/features.txt"
OUTPUT_PATH = "../../data/player_labeled.csv"

# =====================================================
# ROLE MAP
# =====================================================
ROLE_MAP = {
    "TOP": 1,
    "JUNGLE": 2,
    "MIDDLE": 3,
    "BOTTOM": 4,
    "SUPPORT": 5,
    "UTILITY": 5,
}

# =====================================================
# FEATURE WEIGHTS (same as training)
# =====================================================
FEATURE_WEIGHTS = {
    "kpm": 2.0, "dpm": 2.0, "apm": 1.8, "cspm": 1.6,
    "ch_damagePerMinute": 1.5, "ch_killParticipation": 1.4,
    "ch_goldPerMinute": 1.3, "ch_turretTakedowns": 1.3,
    "ch_visionScorePerMinute": 1.2, "objective_focus": 1.2,
    "role": 1.4, "damage_share": 1.2, "kp_rate": 1.1,
    "gold_efficiency": 1.0, "survivability_ratio": 0.9,
    "vision_efficiency": 0.9, "attack": 0.6,
    "defense": 0.6, "magic": 0.6, "difficulty": 0.6,
}

FEATURE_WEIGHTS = {
    "kpm": 2.0, 
    "apm": 1.8, 
    "cspm": 1.6,
    "ch_killParticipation": 1.4,
    "ch_goldPerMinute": 1.3, 
    "ch_turretTakedowns": 1.3,
    "ch_visionScorePerMinute": 1.2, 
    "objective_focus": 1.2,
    "role": 1.4, 
    "damage_share": 1.2, 
    "kp_rate": 1.1,
    "survivability_ratio": 0.9,
    "vision_efficiency": 0.9, 
    "attack": 0.6,
    "defense": 0.6, 
    "magic": 0.6, 
    "difficulty": 0.6,
}

# =====================================================
# LOAD MODELS
# =====================================================
print("Loading model artifacts from ../../models...")
umap_reducer = load(UMAP_PATH)
kmeans = load(MODEL_PATH)
scaler = load(SCALER_PATH)
print("Models loaded successfully.")

# Load feature list
TRAIN_FEATURES = list(scaler.feature_names_in_)

print(f"Loaded {len(TRAIN_FEATURES)} training features")

# =====================================================
# LOAD DATA
# =====================================================
print(f"Loading data from {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()

print(f"Loaded {len(df)} rows × {len(df.columns)} columns")
# =====================================================
# MATCH LIMITER FOR FAIR COMPARISON
# =====================================================
MATCH_LIMIT = 200 

if MATCH_LIMIT is not None:
    print(f"[Limiter] Limiting to first {MATCH_LIMIT} matches for fair comparison...")
    df = df.tail(MATCH_LIMIT)  
    print(f"[Limiter] Now using {len(df)} matches")
# =====================================================
# PREPROCESSING
# =====================================================
print("\n[Predictor] Reproducing trainer preprocessing...")

# Clean infinities and NaNs
df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

# Exclude item columns (same as trainer)
item_cols = [c for c in df.columns if c.lower().startswith("item")]
if item_cols:
    print(f"Dropping {len(item_cols)} item columns...")
    df = df.drop(columns=item_cols, errors="ignore")

# Tag encoding
if "tags" in df.columns:
    print("Encoding champion tags...")
    tag_dummies = df["tags"].fillna("").str.get_dummies(sep=",").add_prefix("tag_")
    df = pd.concat([df, tag_dummies], axis=1)
    print(f"Added {len(tag_dummies.columns)} tag columns.")

# Role encoding - handle both 'role' and 'role_str' columns
if "role_str" in df.columns:
    print("Found 'role_str' column, converting to numeric 'role'...")
    # Drop rows with missing or empty role_str
    before = len(df)
    df = df[df["role_str"].notna() & (df["role_str"].astype(str).str.strip() != "")]
    print(f"Filtered {before - len(df)} rows with missing role_str")

    # Normalize role strings
    df["role_str"] = df["role_str"].astype(str).str.upper()
    df["role_str"] = df["role_str"].replace({
        "UTILITY": "SUPPORT",
        "MID": "MIDDLE",
        "ADC": "BOTTOM",
        "BOT": "BOTTOM",
        "JGL": "JUNGLE"
    })

    # Keep only valid roles
    valid_roles = set(ROLE_MAP.keys())
    before = len(df)
    df = df[df["role_str"].isin(valid_roles)]
    print(f"Filtered {before - len(df)} rows with invalid role_str values")

    # Map role_str → numeric role
    df["role"] = df["role_str"].map(ROLE_MAP)
elif "role" in df.columns:
    print("Found 'role' column, ensuring it's numeric...")
    # If role is already numeric, keep it
    if df["role"].dtype in [np.int64, np.float64]:
        df["role"] = df["role"].fillna(0).astype(int)
    else:
        # If it's string, convert it
        df["role"] = df["role"].astype(str).str.upper().replace({
            "UTILITY": "SUPPORT",
            "MID": "MIDDLE",
            "ADC": "BOTTOM",
            "BOT": "BOTTOM",
            "JGL": "JUNGLE"
        })
        df["role"] = df["role"].map(ROLE_MAP)
else:
    print("Warning: No role column found, filling with 0")
    df["role"] = 0

# Filter empty perk data (same as trainer)
before = len(df)
if all(col in df.columns for col in ["perk_offense", "perk_flex", "perk_defense"]):
    df = df[~((df["perk_offense"] == 0) & (df["perk_flex"] == 0) & (df["perk_defense"] == 0))]
    print(f"Filtered {before - len(df)} rows with all-zero perks; remaining {len(df)}.")

# =====================================================
# DERIVED FEATURES (EXACTLY matching trainer)
# =====================================================
print("Generating derived features...")

# Basic rate calculations
df["kp_rate"] = df["ch_killParticipation"] / 100
df["damage_share"] = df["ch_teamDamagePercentage"] / 100

# Per-minute metrics
df["kpm"] = df["kills"] / df["duration"].clip(lower=1)
df["apm"] = df["assists"] / df["duration"].clip(lower=1)
df["cspm"] = df["cs"] / df["duration"].clip(lower=1)

# Other derived features
df["survivability_ratio"] = df["kills"] / df["deaths"].clip(lower=1)
df["objective_focus"] = (
    df["ch_turretTakedowns"] + 
    df["ch_baronTakedowns"] + 
    df["ch_dragonTakedowns"]
) / 3
df["vision_efficiency"] = df["ch_visionScorePerMinute"] / (df["ch_wardTakedowns"] + 1)

# NOTE: dpm and gold_efficiency are NOT created because they're commented out in trainer

# =====================================================
# FEATURE ALIGNMENT + VALIDATION
# =====================================================
print("\n[Predictor] Aligning features with training data...")

# Remove duplicate columns
df = df.loc[:, ~df.columns.duplicated(keep='first')]

# Reindex to match training features exactly
X = df.reindex(columns=TRAIN_FEATURES, fill_value=0).copy()

# Validation check
missing_features = set(TRAIN_FEATURES) - set(df.columns)
if missing_features:
    print(f" Warning: Missing features in data (will be filled with 0): {missing_features}")

extra_features = set(df.columns) - set(TRAIN_FEATURES)
if extra_features:
    print(f"Info: Extra features in data (will be ignored): {list(extra_features)[:10]}...")

print(f" Feature matrix shape: {X.shape}")
print(f" Expected features: {len(TRAIN_FEATURES)}")

# =====================================================
# APPLY FEATURE WEIGHTS (same as trainer)
# =====================================================
print("\n[Predictor] Applying feature reweighting...")
weights_applied = 0
for feat, weight in FEATURE_WEIGHTS.items():
    if feat in X.columns:
        X[feat] = X[feat] * weight
        weights_applied += 1

print(f" Applied {weights_applied} feature weights")

# =====================================================
# TRANSFORM & PREDICT
# =====================================================
print("\n[Predictor] Transforming and predicting...")

print(f"Scaler expects: {scaler.feature_names_in_.shape[0]} features")
print(f"Your X has: {X.shape[1]} features")

# Verify exact match
if list(X.columns) != list(scaler.feature_names_in_):
    print("Warning: Feature order mismatch detected!")
    # Force correct order
    X = X[scaler.feature_names_in_]
    print(" Reordered features to match scaler")

# Transform
X_scaled = scaler.transform(X)
X_embed = umap_reducer.transform(X_scaled)
df["cluster"] = kmeans.predict(X_embed)

print(f"✓ Assigned clusters to {len(df)} samples")

# =====================================================
# OUTPUT
# =====================================================
print("\n" + "="*60)
print("CLUSTER DISTRIBUTION (Games per Cluster)")
print("="*60)

cluster_counts = df["cluster"].value_counts().sort_index()
for c, count in cluster_counts.items():
    pct = (count / len(df)) * 100
    bar = "█" * int(pct / 2)
    print(f"  Cluster {c:>2}: {count:>5} games ({pct:5.2f}%) {bar}")

print("="*60)

# Save cluster distribution as CSV
dist_output_path = "../../data/cluster_distribution.csv"
cluster_counts.to_csv(dist_output_path, header=["game_count"])
print(f"\nSaved cluster distribution to {dist_output_path}")

# Save full labeled player data
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved labeled player data to {OUTPUT_PATH}")

print("\nPrediction complete!")