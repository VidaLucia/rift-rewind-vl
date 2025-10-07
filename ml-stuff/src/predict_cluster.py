import pandas as pd
import numpy as np
from joblib import load
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import umap

# === CONFIG ===
DATA_PATH = "../../data/players/lulululu04#lulu_data.csv"
UMAP_PATH = "../../models/umap_reducer.pkl"
MODEL_PATH = "../../models/kmeans_model.pkl"
OUTPUT_PATH = "../../data/player_labeled.csv"
SCALER_PATH = "../../models/scaler.pkl"

# === LOAD DATA ===
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows from {DATA_PATH}")

# === TEMP: Champion ID Map (TODO: remove when Player Finder fixed) ===
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

# === LOAD MODEL ===
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
print(f"Loading model from {MODEL_PATH}")
kmeans = load(MODEL_PATH)

# === FEATURE SET ===
features = [
    "kills", "deaths", "assists", "damage", "cs", "role",
    "kp_rate", "damage_share", "gold_efficiency", "objective_focus",
    "survivability_ratio", "vision_efficiency",
    "primarystyle_id", "primarystyle_perk1", "primarystyle_perk2", "primarystyle_perk3",
    "substyle_id", "substyle_perk1", "substyle_perk2",
    "ch_kda", "ch_killingSprees", "ch_damagePerMinute",
    "ch_teamDamagePercentage", "ch_killParticipation",
    "ch_turretTakedowns", "ch_baronTakedowns", "ch_dragonTakedowns",
    "ch_enemyJungleMonsterKills", "ch_goldPerMinute", "ch_laningPhaseGoldExpAdvantage",
    "ch_maxCsAdvantageOnLaneOpponent", "ch_deathsByEnemyChamps", "ch_damageTakenOnTeamPercentage",
    "ch_survivedSingleDigitHpCount", "ch_effectiveHealAndShielding", "ch_saveAllyFromDeath",
    "ch_immobilizeAndKillWithAlly", "ch_visionScorePerMinute",
    "kpm", "dpm", "apm", "cspm", "_id",
    "attack", "defense", "magic", "difficulty",
    "tag_0", "tag_Assassin", "tag_Fighter", "tag_Mage",
    "tag_Marksman", "tag_Support", "tag_Tank",
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

X = df[available_features].replace([np.inf, -np.inf], np.nan).fillna(0)

# === LOAD SCALER & ALIGN FEATURES ===
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")
print(f"Loading scaler from {SCALER_PATH}")
scaler = load(SCALER_PATH)

if hasattr(scaler, "feature_names_in_"):
    train_features = list(scaler.feature_names_in_)
    missing = set(train_features) - set(X.columns)
    extra = set(X.columns) - set(train_features)
    for col in missing:
        X[col] = 0
        print(f" Added missing training feature '{col}'")
    if extra:
        print(f" Dropping {len(extra)} extra columns: {list(extra)[:10]}...")
        X = X.drop(columns=list(extra), errors="ignore")
    X = X[train_features]
    print(f"Aligned feature order with training ({len(train_features)} columns)")

# === SCALING (with adaptive fix) ===
X_scaled = scaler.transform(X)
local_var = np.var(X_scaled, axis=0).mean()
print(f"Data variance (scaled): {local_var:.6f}")
if local_var < 0.05:
    print("Low variance detected — applying local StandardScaler() for better embedding.")
    adaptive_scaler = StandardScaler()
    X_scaled = adaptive_scaler.fit_transform(X_scaled)

# === LOAD UMAP ===
print(f"Loading UMAP reducer from {UMAP_PATH}")
reducer = load(UMAP_PATH)

X_embed = reducer.transform(X_scaled)
print(f"UMAP shape: {X_embed.shape}")

# === SAFEGUARD: Detect collapsed embedding ===
if np.std(X_embed[:, 0]) < 0.01 and np.std(X_embed[:, 1]) < 0.01:
    print("UMAP collapsed — fitting local temporary reducer.")
    local_reducer = umap.UMAP(
        n_neighbors=30, min_dist=0.05, n_components=10, random_state=42
    )
    X_embed = local_reducer.fit_transform(X_scaled)
    print("Local UMAP embedding restored.")

# === DIAGNOSTIC: Nearest cluster centroids ===
centroids = kmeans.cluster_centers_
dists = np.linalg.norm(X_embed[:, None, :] - centroids[None, :, :], axis=2)
closest = np.argmin(dists, axis=1)
unique, counts = np.unique(closest, return_counts=True)
print("\nNearest centroid counts:")
for c, n in zip(unique, counts):
    pct = n / len(X_embed) * 100
    print(f"  Cluster {c:<2}: {n:>5} samples ({pct:.2f}%)")

# === CLUSTER PREDICTION ===
df["predicted_cluster"] = kmeans.predict(X_embed)

# === CLUSTER SUMMARY ===
cluster_counts = df["predicted_cluster"].value_counts().sort_index()
print("\nCluster Distribution")
for c, count in cluster_counts.items():
    pct = count / len(df) * 100
    print(f"  Cluster {c:<2}: {count:>5} samples ({pct:.2f}%)")

# === VISUALIZATION ===
plt.figure(figsize=(6,5))
plt.scatter(X_embed[:, 0], X_embed[:, 1], s=8, c=df["predicted_cluster"], cmap="tab20")
plt.title("UMAP Projection (colored by predicted cluster)")
plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
plt.tight_layout()
plt.show()

# === SAVE OUTPUT ===
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved clustered player data to {OUTPUT_PATH}")
