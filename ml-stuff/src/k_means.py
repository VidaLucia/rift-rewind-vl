import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from pca import visualize_kmeans
from kneed import KneeLocator
import umap.umap_ as umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
DISCOVERY = True
BEST_K = 14  # change after inspecting plot
USE_UMAP = True
USE_TSNE = True
SAVE_PATH = "../../data/labeled_clusters.csv"

#  Load data 
df = pd.read_csv("../../data/normalized_matches.csv")
print(f"Loaded {len(df)} rows")

#  Clean NaNs and invalid values 
df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
print(f"Columns available: {len(df.columns)}")

# Encode champion tags
if "tags" in df.columns:
    print("Encoding champion tags")
    # Split tags like "Fighter,Assassin" into separate flags
    tag_dummies = (
        df["tags"]
        .fillna("")
        .str.get_dummies(sep=",")
        .add_prefix("tag_")
    )
    df = pd.concat([df, tag_dummies], axis=1)
    print(f" Added {len(tag_dummies.columns)} tag columns: {list(tag_dummies.columns)}")

#  Filter out players with no perk data 
before = len(df)
df = df[
    ~((df["perk_offense"] == 0) &
      (df["perk_flex"] == 0) &
      (df["perk_defense"] == 0))
]
# perk_offense,flex and defense are the 3 small runes -> tbh we should just use this to prune and these are not really good features but actual rune data will be good
# primarystyle_id -> is the main rune tree (i.e. precision,domination etc) -> might use this over independent runes
# primarystyle_perk1 -> keystone rune (VERY IMPORTANT probably keep)
# primarystyle_perk2 -> second rune in main tree
# primarystyle_perk3 -> third rune in main tree
# substyle_id -> secondary rune tree
# substyle_perk1 -> first rune in secondary tree
# substyle_perk2 -> second rune in secondary tree
print(f"Filtered {before - len(df)} rows with all-zero perks, remaining: {len(df)}")


#  Define features 
print("Generating derived features")
df["kp_rate"] = df["ch_killParticipation"] / 100
df["damage_share"] = df["ch_teamDamagePercentage"] / 100
df["gold_efficiency"] = df["ch_goldPerMinute"] / df["duration"].clip(lower=1)
df["survivability_ratio"] = df["kills"] / df["deaths"].clip(lower=1)
df["objective_focus"] = (
    df["ch_turretTakedowns"] +
    df["ch_baronTakedowns"] +
    df["ch_dragonTakedowns"]
) / 3
df["vision_efficiency"] = df["ch_visionScorePerMinute"] / (df["ch_wardTakedowns"] + 1)

#  Define Features 
features = [
    # Core performance
    "kills", "deaths", "assists", "gold_earned", "damage", "vision_score",
    "cs", "win", "duration",

    # Derived metrics
    "kp_rate", "damage_share", "gold_efficiency",
    "objective_focus", "survivability_ratio", "vision_efficiency",
    # Runes Main Runes
    "primarystyle_id", "primarystyle_perk1", "primarystyle_perk2", "primarystyle_perk3",
    # Runes Sub Runes
    "substyle_id", "substyle_perk1", "substyle_perk2",
    # Combat / Efficiency
    "ch_kda", "ch_killingSprees", "ch_damagePerMinute", "ch_teamDamagePercentage",
    "ch_killParticipation",

    # Macro / Objectives
    "ch_turretTakedowns", "ch_baronTakedowns", "ch_dragonTakedowns",
    "ch_enemyJungleMonsterKills",

    # Scaling
    "ch_goldPerMinute", "ch_laningPhaseGoldExpAdvantage",
    "ch_maxCsAdvantageOnLaneOpponent",

    # Survivability
    "ch_deathsByEnemyChamps", "ch_damageTakenOnTeamPercentage",
    "ch_survivedSingleDigitHpCount",

    # Utility / Support
    "ch_effectiveHealAndShielding", "ch_saveAllyFromDeath",
    "ch_immobilizeAndKillWithAlly",

    # Champion Info
    "attack", "defense", "magic", "difficulty",
]

# Include tag columns 
tag_cols = [c for c in df.columns if c.startswith("tag_")]
features.extend(tag_cols)

available_features = [f for f in features if f in df.columns]
missing = set(features) - set(available_features)
if missing:
    print(" Missing features:", missing)

X = df[available_features]
print(f"Feature matrix shape: {X.shape}")
if X.empty:
    raise ValueError(" No data left after filtering — check filtering logic!")

#  Weighted Features 
feature_weights = {
    # Emphasize key gameplay features
    "kills": 2.0, "deaths": 2.0, "assists": 1.8,
    "ch_damagePerMinute": 1.5, "ch_killParticipation": 1.4,
    "ch_goldPerMinute": 1.3, "ch_turretTakedowns": 1.4,

    # De-emphasize static attributes
    "attack": 0.6, "defense": 0.6, "magic": 0.6, "difficulty": 0.6,
}
print("Applying weighted features")
for feat, weight in feature_weights.items():
    if feat in X.columns:
        X.loc[:, feat] = X[feat] * weight

#  Scale 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  Dimensionality Reduction 
print("\nRunning PCA (retain 95% variance)")
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA reduced to {X_pca.shape[1]} dimensions")

if USE_UMAP:
    print("Running UMAP for 2D visualization")
    reducer = umap.UMAP(n_neighbors=50, min_dist=0.3, random_state=42)
    X_embed = reducer.fit_transform(X_pca)
elif USE_TSNE:
    print("Running t-SNE for 2D visualization (slow)")
    tsne = TSNE(n_components=2, random_state=42, perplexity=40, n_iter=1000)
    X_embed = tsne.fit_transform(X_pca)
else:
    X_embed = X_pca

#  DISCOVERY PHASE 
K_RANGE = range(4, 41, 2)
if DISCOVERY:
    inertias, silhouettes = [], []
    print("\nRunning KMeans across cluster counts")
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_embed)
        inertia = km.inertia_
        inertias.append(inertia)

        if len(X_embed) > 10000:
            subset_idx = np.random.choice(len(X_embed), 10000, replace=False)
            sil = silhouette_score(X_embed[subset_idx], labels[subset_idx])
        else:
            sil = silhouette_score(X_embed, labels)

        silhouettes.append(sil)
        print(f"K={k:<3} | Inertia={inertia:,.0f} | Silhouette={sil:.4f}")

    #  Plot Elbow & Silhouette 
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax1.plot(K_RANGE, inertias, "o-", color="blue", label="Inertia (Elbow)")
    ax2.plot(K_RANGE, silhouettes, "s--", color="green", label="Silhouette")
    ax1.set_xlabel("Clusters (K)")
    ax1.set_ylabel("Inertia", color="blue")
    ax2.set_ylabel("Silhouette", color="green")
    plt.title("KMeans Elbow + Silhouette Analysis")
    plt.grid(True)
    fig.tight_layout()
    plt.show()

#  FINAL CLUSTERING 
print(f"\nUsing K={BEST_K} for final clustering")
kmeans = KMeans(n_clusters=BEST_K, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_embed)

#  Summarize 
cluster_summary = df.groupby("cluster")[available_features].mean().round(2)
print("\n Cluster Summary ")
print(cluster_summary)

#  Save 
df.to_csv(SAVE_PATH, index=False)
print(f"\nSaved clustered dataset to {SAVE_PATH}")

#  Visualize 
visualize_kmeans(X_embed, df, kmeans, available_features)