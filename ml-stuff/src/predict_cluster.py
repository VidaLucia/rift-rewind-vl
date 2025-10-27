import pandas as pd
import numpy as np
from joblib import load
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap
from joblib import dump
# CONFIG 
DATA_PATH = "../../data/players/aquatick#001_data.csv"
UMAP_PATH = "../../models/umap_reducer.pkl"
MODEL_PATH = "../../models/kmeans_model.pkl"
OUTPUT_PATH = "../../data/player_labeled.csv"
SCALER_PATH = "../../models/scaler.pkl"

#Role Map
ROLE_MAP = {
    "TOP": 1,
    "JUNGLE": 2,
    "MIDDLE": 3,
    "BOTTOM": 4,
    "SUPPORT": 5,
    "UTILITY": 5,  # Merge UTILITY = SUPPORT
}

def json_safe(obj):
    """Recursively convert NumPy types and objects to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_safe(x) for x in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    else:
        return obj
# LOAD DATA 
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows from {DATA_PATH}")
MATCH_LIMIT = 200 

if MATCH_LIMIT is not None:
    print(f"[Limiter] Limiting to first {MATCH_LIMIT} matches for fair comparison...")
    df = df.tail(MATCH_LIMIT)  
    print(f"[Limiter] Now using {len(df)} matches")

# TEMP: Champion ID Map (TODO: remove when Player Finder fixed) 
champion_id_map = {
    "Aatrox": 266, "Ahri": 103, "Akali": 84, "Akshan": 166, "Alistar": 12,
    "Ambessa": 799, "Amumu": 32, "Anivia": 34, "Annie": 1, "Aphelios": 523,
    "Ashe": 22, "AurelionSol": 136, "Aurora": 893, "Azir": 268, "Bard": 432,
    "Belveth": 200, "Blitzcrank": 53, "Brand": 63, "Braum": 201, "Briar": 233,
    "Caitlyn": 51, "Camille": 164, "Cassiopeia": 69, "Chogath": 31, "Corki": 42,
    "Darius": 122, "Diana": 131, "Draven": 119, "DrMundo": 36, "Ekko": 245,
    "Elise": 60, "Evelynn": 28, "Ezreal": 81, "Fiddlesticks": 9, "Fiora": 114,
    "Fizz": 105, "Galio": 3, "Gangplank": 41, "Garen": 86, "Gnar": 150,
    "Gragas": 79, "Graves": 104, "Gwen": 887, "Hecarim": 120, "Heimerdinger": 74,
    "Hwei": 910, "Illaoi": 420, "Irelia": 39, "Ivern": 427, "Janna": 40,
    "JarvanIV": 59, "Jax": 24, "Jayce": 126, "Jhin": 202, "Jinx": 222,
    "Kaisa": 145, "Kalista": 429, "Karma": 43, "Karthus": 30, "Kassadin": 38,
    "Katarina": 55, "Kayle": 10, "Kayn": 141, "Kennen": 85, "Khazix": 121,
    "Kindred": 203, "Kled": 240, "KogMaw": 96, "KSante": 897, "Leblanc": 7,
    "LeeSin": 64, "Leona": 89, "Lillia": 876, "Lissandra": 127, "Lucian": 236,
    "Lulu": 117, "Lux": 99, "Malphite": 54, "Malzahar": 90, "Maokai": 57,
    "MasterYi": 11, "Mel": 800, "Milio": 902, "MissFortune": 21, "MonkeyKing": 62,
    "Mordekaiser": 82, "Morgana": 25, "Naafiri": 950, "Nami": 267, "Nasus": 75,
    "Nautilus": 111, "Neeko": 518, "Nidalee": 76, "Nilah": 895, "Nocturne": 56,
    "Nunu": 20, "Olaf": 2, "Orianna": 61, "Ornn": 516, "Pantheon": 80,
    "Poppy": 78, "Pyke": 555, "Qiyana": 246, "Quinn": 133, "Rakan": 497,
    "Rammus": 33, "RekSai": 421, "Rell": 526, "Renata": 888, "Renekton": 58,
    "Rengar": 107, "Riven": 92, "Rumble": 68, "Ryze": 13, "Samira": 360,
    "Sejuani": 113, "Senna": 235, "Seraphine": 147, "Sett": 875, "Shaco": 35,
    "Shen": 98, "Shyvana": 102, "Singed": 27, "Sion": 14, "Sivir": 15,
    "Skarner": 72, "Smolder": 901, "Sona": 37, "Soraka": 16, "Swain": 50,
    "Sylas": 517, "Syndra": 134, "TahmKench": 223, "Taliyah": 163, "Talon": 91,
    "Taric": 44, "Teemo": 17, "Thresh": 412, "Tristana": 18, "Trundle": 48,
    "Tryndamere": 23, "TwistedFate": 4, "Twitch": 29, "Udyr": 77, "Urgot": 6,
    "Varus": 110, "Vayne": 67, "Veigar": 45, "Velkoz": 161, "Vex": 711,
    "Vi": 254, "Viego": 234, "Viktor": 112, "Vladimir": 8, "Volibear": 106,
    "Warwick": 19, "Xayah": 498, "Xerath": 101, "XinZhao": 5, "Yasuo": 157,
    "Yone": 777, "Yorick": 83, "Yunara": 804, "Yuumi": 350, "Zac": 154,
    "Zed": 238, "Zeri": 221, "Ziggs": 115, "Zilean": 26, "Zoe": 142, "Zyra": 143
}

df["champion_key"] = (
    df["champion"]
    .str.replace(" ", "")
    .str.replace("'", "")
    .str.replace(".", "")
)
df["_id"] = df["champion_key"].map(champion_id_map).fillna(0).astype(int)
print(f"Added champion ID column (_id) — {df['_id'].nunique()} unique champions")

# LOAD MODEL 
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
print(f"Loading model from {MODEL_PATH}")
kmeans = load(MODEL_PATH)

# FEATURE SET 
features = [
    "kills", "deaths", "assists", "damage", "cs", "role",
    "kp_rate", "damage_share", #"gold_efficiency",
    "objective_focus", "survivability_ratio", "vision_efficiency",
    "primarystyle_id", "primarystyle_perk1", "primarystyle_perk2", "primarystyle_perk3","primarystyle_perk4",
    "substyle_id", "substyle_perk1", "substyle_perk2",
    "ch_kda", "ch_killingSprees", #"ch_damagePerMinute",
    "ch_teamDamagePercentage", #"ch_killParticipation",
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

for feat in features:
    if feat not in df.columns:
        df[feat] = 0
        print(f"Added missing feature '{feat}' (filled with 0)")

available_features = [f for f in features if f in df.columns]
missing = set(features) - set(available_features)
if missing:
    print(f"Missing features required by model: {missing}")
else:
    print("All required features are present.")
if "role" in df.columns:
    df["role"] = (
        df["role"]
        .astype(str)
        .str.upper()
        .map(ROLE_MAP)
        .fillna(0)
        .astype(int)
    )
    print("Converted role column to numeric IDs using ROLE_MAP")
else:
    print("Warning: role column missing; filling with 0")
    df["role"] = 0
X = df[available_features].replace([np.inf, -np.inf], np.nan).fillna(0)
# Feature Group Normalization 
# This prevents damage metrics from overpowering the embedding

#if "_id" in X.columns:
    #print("Downweighting champion ID (_id) to reduce numeric bias")
    #X["_id"] *= 0.001

# Step 2. Log-compress extreme damage / participation metrics 
high_var_cols = ["kp_rate", "dpm", "damage_share"]
for col in high_var_cols:
    if col in X.columns:
        before_max = X[col].max()
        X[col] = np.log1p(X[col])  # compress extremes (e.g. 1200 → ~7)
        X[col] *= 0.5               # scale down slightly
        after_max = X[col].max()
        print(f"Log-compressed '{col}': max {before_max:.1f} -> {after_max:.1f}")
# Downweight the riot champion attributes
feature_weights = {
    "kpm": 2.0, "dpm": 2.0, "apm": 1.8, "cspm": 1.6,
    #"ch_damagePerMinute": 1.5, 
    "ch_killParticipation": 1.4,
    "ch_goldPerMinute": 1.3, "ch_turretTakedowns": 1.3,
    "ch_visionScorePerMinute": 1.2, "objective_focus": 1.2,
    "role": 1.4, "damage_share": 1.2, "kp_rate": 1.1,
    #"gold_efficiency": 1.0, 
    "survivability_ratio": 0.9,
    "vision_efficiency": 0.9, "attack": 0.6,
    "defense": 0.6, "magic": 0.6, "difficulty": 0.6,
}
for feat, weight in feature_weights.items():
    if feat in X.columns:
        X[feat] *= weight


# Step 4. Check pre-scaling variance for diagnostics 
pre_var = pd.Series(np.var(X, axis=0), index=X.columns)
top_pre = pre_var.sort_values(ascending=False).head(10)
print("\nTop 10 feature variances BEFORE scaling:")
print(top_pre)
# Step 5. Automatic variance rebalance for outliers 
print("\nAuto Variance Rebalance ")
var_series = pd.Series(np.var(X, axis=0), index=X.columns)
median_var = var_series.median()
threshold = 5.0  # any feature >5× median variance gets downscaled

rebalance_cols = var_series[var_series > median_var * threshold]
if not rebalance_cols.empty:
    for col, var_val in rebalance_cols.items():
        scale_factor = np.sqrt(var_val / (median_var * threshold))
        X[col] /= scale_factor
        print(f" Downweighted '{col}' by {scale_factor:.2f}× (var {var_val:.1f})")
else:
    print("No extreme variance features found.")
# LOAD SCALER & ALIGN FEATURES 
# Try loading existing scaler
if not os.path.exists(SCALER_PATH):
    print(f"Scaler not found at {SCALER_PATH}. Creating new one")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    dump(scaler, "../../models/scaler_balanced.pkl")
    print("Saved new scaler to ../../models/scaler_balanced.pkl")
else:
    print(f"Loading scaler from {SCALER_PATH}")
    scaler = load(SCALER_PATH)

# Align features if the scaler has known feature names
if hasattr(scaler, "feature_names_in_"):
    train_features = list(scaler.feature_names_in_)
    missing = set(train_features) - set(X.columns)
    extra = set(X.columns) - set(train_features)

    for col in missing:
        X[col] = 0
        print(f" Added missing training feature '{col}'")
    if extra:
        print(f" Dropping {len(extra)} extra columns: {list(extra)[:10]} ")
        X = X.drop(columns=list(extra), errors="ignore")

    X = X[train_features]
    print(f"Aligned feature order with training ({len(train_features)} columns)")

# Transform or retrain scaler 
try:
    X_scaled = scaler.transform(X)
    print("Successfully applied existing scaler.")
except Exception as e:
    print(f"Scaler transform failed ({e}). Refitting a new balanced scaler")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    dump(scaler, "../../models/scaler_balanced.pkl")
    print("Saved rebalanced scaler to ../../models/scaler_balanced.pkl")

# Diagnostics 
local_var = np.var(X_scaled, axis=0).mean()
print(f"Mean variance after scaling: {local_var:.6f}")

scaled_var = pd.Series(np.var(X_scaled, axis=0), index=X.columns)
top_scaled = scaled_var.sort_values(ascending=False).head(10)
print("\nTop 10 features by variance after scaling:")
print(top_scaled)

# If old scaler over-compresses variance (like kp_rate = 3000+), refit a local one
if top_scaled.max() > 10:
    print(" Detected extreme variance post-scaling. Refitting adaptive scaler.")
    adaptive_scaler = StandardScaler()
    X_scaled = adaptive_scaler.fit_transform(X)
    dump(adaptive_scaler, "../../models/scaler_balanced.pkl")
    print("Refit and saved adaptive balanced scaler.")
    scaled_var = pd.Series(np.var(X_scaled, axis=0), index=X.columns)
    print("\n[After adaptive re-scaling]")
    print(scaled_var.sort_values(ascending=False).head(10))
# LOAD UMAP 
print(f"Loading UMAP reducer from {UMAP_PATH}") 
reducer = load(UMAP_PATH) 
X_embed = reducer.transform(X_scaled) 
print(f"UMAP shape: {X_embed.shape}")
# SAFEGUARD: Detect collapsed embedding 
if np.std(X_embed[:, 0]) < 0.01 and np.std(X_embed[:, 1]) < 0.01:
    print("UMAP collapsed — fitting local temporary reducer.")
    local_reducer = umap.UMAP(
        n_neighbors=30, min_dist=0.05, n_components=10, random_state=42
    )
    X_embed = local_reducer.fit_transform(X_scaled)
    print("Local UMAP embedding restored.")

# DIAGNOSTIC: Nearest cluster centroids 
centroids = kmeans.cluster_centers_
dists = np.linalg.norm(X_embed[:, None, :] - centroids[None, :, :], axis=2)
closest = np.argmin(dists, axis=1)
unique, counts = np.unique(closest, return_counts=True)
print("\nNearest centroid counts:")
for c, n in zip(unique, counts):
    pct = n / len(X_embed) * 100
    print(f"  Cluster {c:<2}: {n:>5} samples ({pct:.2f}%)")

# CLUSTER PREDICTION 
df["predicted_cluster"] = kmeans.predict(X_embed)
from sklearn.metrics import pairwise_distances
import json

# SOFT CLUSTER MEMBERSHIP CALCULATION 
print("\n Computing soft cluster memberships (distance-weighted)")

centroids = kmeans.cluster_centers_
dist_matrix = pairwise_distances(X_embed, centroids)

# Inverse distance weighting - closer = higher confidence
inv_dist = 1 / (dist_matrix + 1e-6)
soft_probs = inv_dist / inv_dist.sum(axis=1, keepdims=True)

# Top-3 cluster indices per match (for context analysis)
top3 = np.argsort(-soft_probs, axis=1)[:, :3]
df["top1"] = top3[:, 0]
df["top2"] = top3[:, 1]
df["top3"] = top3[:, 2]

# Average soft membership across all matches 
soft_mean = soft_probs.mean(axis=0)
print("Soft cluster probabilities (mean confidence across matches):")
for i, p in enumerate(soft_mean):
    print(f"  Cluster {i:<2}: {p:.3f}")

# ormalization (to avoid one cluster dominating) 
threshold = 0.20  # adjust for how aggressive balancing should be
mean_prob = soft_mean.mean()
scaling = np.clip(mean_prob / (soft_mean + 1e-9), 0.5, 1.5)
soft_probs_balanced = soft_probs * scaling
soft_probs_balanced /= soft_probs_balanced.sum(axis=1, keepdims=True)

df["soft_cluster"] = np.argmax(soft_probs_balanced, axis=1)

print("\n Balanced soft cluster distribution:")
for c, n in pd.Series(df["soft_cluster"]).value_counts().sort_index().items():
    pct = n / len(df) * 100
    print(f"  Cluster {c:<2}: {n:>5} samples ({pct:.2f}%)")


# df["predicted_cluster"] = df["soft_cluster"]

# SAVE CENTROIDS & PROBABILITIES 
export_dir = "../../data/player_analysis"
os.makedirs(export_dir, exist_ok=True)

centroid_path = os.path.join(export_dir, "cluster_centroids.npy")
probs_path = os.path.join(export_dir, "soft_probabilities.npy")
summary_path = os.path.join(export_dir, "cluster_summary.json")

np.save(centroid_path, centroids)
np.save(probs_path, soft_probs_balanced)

summary = {
    "total_matches": len(df),
    "clusters": {
        str(i): {
            "mean_confidence": float(soft_mean[i]),
            "sample_count": int((df["soft_cluster"] == i).sum())
        } for i in range(len(centroids))
    }
}

with open(summary_path, "w") as f:
    json.dump(json_safe(summary), f, indent=2)

print(f"\n Saved analysis files:")
print(f"   Centroids: {centroid_path}")
print(f"   Soft probabilities: {probs_path}")
print(f"   Cluster summary: {summary_path}")
# CLUSTER SUMMARY 
cluster_counts = df["predicted_cluster"].value_counts().sort_index()
print("\nCluster Distribution")
for c, count in cluster_counts.items():
    pct = count / len(df) * 100
    print(f"  Cluster {c:<2}: {count:>5} samples ({pct:.2f}%)")
from scipy.stats import zscore

from scipy.stats import zscore

print("\nComputing full feature deltas per cluster...")

# Z-score normalize for comparability
feature_df = pd.DataFrame(X_scaled, columns=X.columns)
feature_df["cluster"] = df["soft_cluster"]

cluster_profiles = {}
report_lines = []

global_mean = feature_df.drop(columns=["cluster"]).mean(numeric_only=True)

# =====================================================
# COMPUTE FULL FEATURE DELTAS FOR EVERY CLUSTER
# =====================================================
for c in sorted(feature_df["cluster"].unique()):
    cluster_data = feature_df[feature_df["cluster"] == c]
    cluster_mean = cluster_data.drop(columns=["cluster"]).mean(numeric_only=True)
    delta = cluster_mean - global_mean
    delta_sorted = delta.sort_values(ascending=False)  # high→low deltas

    # Store in JSON
    cluster_profiles[c] = {
        "sample_count": int(len(cluster_data)),
        "feature_deltas": {feat: float(delta[feat]) for feat in delta.index},
    }

    # Append to text report
    report_lines.append(f"\n=== Cluster {c} ===")
    report_lines.append(f"Samples: {len(cluster_data)}")
    report_lines.append("-" * 50)
    report_lines.append(f"{'Feature':<35} {'Δ Mean (cluster - global)':>25}")
    report_lines.append("-" * 50)

    for feat in delta_sorted.index:
        val = delta[feat]
        arrow = "+" if val > 0 else "-"
        report_lines.append(f"{feat:<35} {arrow}{abs(val):>8.4f}")

    report_lines.append("")

# =====================================================
# PLAYSTYLE LABELING (rule-based heuristic)
# =====================================================
print("Deriving playstyle labels for each cluster")

def infer_playstyle(features_dict):
    # use top few high magnitude features for inference
    top_feats = sorted(features_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)[:12]
    fset = " ".join([f for f, _ in top_feats]).lower()

    score = {
        "Aggressive Carry": 0,
        "Vision Support": 0,
        "Macro Objective Player": 0,
        "Defensive / Safe Player": 0,
        "High-Tempo Fighter": 0,
        "Early-Game Dominator": 0,
    }
    if "damage_share" in fset or "dpm" in fset: score["Aggressive Carry"] += 3
    if "kills" in fset or "killing" in fset: score["Aggressive Carry"] += 2
    if "assists" in fset or "vision" in fset: score["Vision Support"] += 3
    if "heal" in fset or "shield" in fset or "save" in fset: score["Vision Support"] += 2
    if "objective" in fset or "baron" in fset or "dragon" in fset: score["Macro Objective Player"] += 3
    if "gold" in fset: score["Macro Objective Player"] += 2
    if "survivability" in fset or "death" in fset or "taken" in fset: score["Defensive / Safe Player"] += 3
    if "kpm" in fset or "apm" in fset: score["High-Tempo Fighter"] += 3
    if "cs" in fset or "laning" in fset: score["Early-Game Dominator"] += 2

    top_two = sorted(score.items(), key=lambda kv: kv[1], reverse=True)[:2]
    if top_two[0][1] == 0:
        return "Generalist"
    if top_two[1][1] >= top_two[0][1] * 0.7:
        return f"{top_two[0][0]} / {top_two[1][0]}"
    return top_two[0][0]

for c in cluster_profiles.keys():
    label = infer_playstyle(cluster_profiles[c]["feature_deltas"])
    cluster_profiles[c]["playstyle_label"] = label
    report_lines.append(f"Playstyle Label (Cluster {c}): {label}\n")

# =====================================================
# SAVE EXTENDED SUMMARY JSON
# =====================================================
summary["clusters_detailed"] = cluster_profiles
summary_path = os.path.join(export_dir, "cluster_summary.json")
with open(summary_path, "w") as f:
    json.dump(json_safe(summary), f, indent=2)
print(f"Added full deltas + playstyle labels to {summary_path}")

# =====================================================
# SAVE HUMAN-READABLE TEXT REPORT
# =====================================================
report_path = os.path.join(export_dir, "clusters_full_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"Saved full cluster report with all features to {report_path}")

# OPTIONAL PREVIEW
for c in sorted(cluster_profiles.keys()):
    label = cluster_profiles[c]["playstyle_label"]
    print(f"Cluster {c:>2}: {label} ({cluster_profiles[c]['sample_count']} samples)")

# Optional: print preview
#for c, info in cluster_profiles.items():
    #print(f"\nCluster {c} ({info['sample_count']} samples)")
    #for feat in info["top_features"]:
        #print(f"   - {feat}")
# VISUALIZATION 
plt.figure(figsize=(6,5))
plt.scatter(X_embed[:, 0], X_embed[:, 1], s=8, c=df["predicted_cluster"], cmap="tab20")
plt.title("UMAP Projection (colored by predicted cluster)")
plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
plt.tight_layout()
plt.show()

# SAVE OUTPUT 
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved clustered player data to {OUTPUT_PATH}")
