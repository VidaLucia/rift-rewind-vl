import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import umap.umap_ as umap
import hdbscan
import seaborn as sns

#  CONFIG 
BEST_K = 14
UMAP_COMPONENTS = 10
MIN_CLUSTER_SIZE = 200
RANDOM_STATE = 42

#  LOAD DATA 
df = pd.read_csv("../../data/normalized_matches.csv")
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

#  TAG ENCODING 
if "tags" in df.columns:
    print("Encoding champion tags...")
    tag_dummies = (
        df["tags"]
        .fillna("")
        .str.get_dummies(sep=",")
        .add_prefix("tag_")
    )
    df = pd.concat([df, tag_dummies], axis=1)
    print(f" Added {len(tag_dummies.columns)} tag columns")

#  FILTER 
before = len(df)
df = df[
    ~((df["perk_offense"] == 0) &
      (df["perk_flex"] == 0) &
      (df["perk_defense"] == 0))
]
print(f"Filtered {before - len(df)} rows (zero runes)")

#  DERIVED FEATURES 
print("Generating derived features...")
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

#  FEATURES 
features = [
    "kills", "deaths", "assists", "gold_earned", "damage", "vision_score",
    "cs", "win", "duration",
    "kp_rate", "damage_share", "gold_efficiency",
    "objective_focus", "survivability_ratio", "vision_efficiency",
    "primarystyle_id", "primarystyle_perk1", "primarystyle_perk2", "primarystyle_perk3",
    "substyle_id", "substyle_perk1", "substyle_perk2",
    "ch_kda", "ch_killingSprees", "ch_damagePerMinute",
    "ch_teamDamagePercentage", "ch_killParticipation",
    "ch_turretTakedowns", "ch_baronTakedowns", "ch_dragonTakedowns",
    "ch_enemyJungleMonsterKills",
    "ch_goldPerMinute", "ch_laningPhaseGoldExpAdvantage", "ch_maxCsAdvantageOnLaneOpponent",
    "ch_deathsByEnemyChamps", "ch_damageTakenOnTeamPercentage",
    "ch_effectiveHealAndShielding", "ch_saveAllyFromDeath", "ch_immobilizeAndKillWithAlly",
    "attack", "defense", "magic", "difficulty"
]

tag_cols = [c for c in df.columns if c.startswith("tag_")]
features.extend(tag_cols)

available_features = [f for f in features if f in df.columns]
X = df[available_features]
print(f"Using {len(available_features)} features")

#  SCALE 
scaler = RobustScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=available_features)

#  CORRELATION PRUNING 
corr_matrix = X_scaled.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.85)]
print(f"Dropping {len(to_drop)} highly correlated features")
X_reduced = X_scaled.drop(columns=to_drop)

#  UMAP REDUCTION 
print("Running UMAP reduction...")
reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, n_components=UMAP_COMPONENTS, random_state=RANDOM_STATE)
X_umap = reducer.fit_transform(X_reduced)
print(f"UMAP reduced to {X_umap.shape[1]}D")

#  CLUSTERING: KMEANS 
print("Running KMeans...")
kmeans = KMeans(n_clusters=BEST_K, random_state=RANDOM_STATE, n_init=20)
labels_kmeans = kmeans.fit_predict(X_umap)
sil_kmeans = silhouette_score(X_umap, labels_kmeans)
print(f"KMeans Silhouette: {sil_kmeans:.4f}")

#  CLUSTERING: HDBSCAN 
print("Running HDBSCAN...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, metric='euclidean')
labels_hdb = clusterer.fit_predict(X_umap)
mask = labels_hdb >= 0
if np.sum(mask) > 1 and len(set(labels_hdb[mask])) > 1:
    sil_hdb = silhouette_score(X_umap[mask], labels_hdb[mask])
else:
    sil_hdb = -1
print(f"HDBSCAN Silhouette: {sil_hdb:.4f}")

#  METRICS 
print(f"Calinski-Harabasz (KMeans): {calinski_harabasz_score(X_umap, labels_kmeans):.2f}")
print(f"Davies-Bouldin (KMeans): {davies_bouldin_score(X_umap, labels_kmeans):.2f}")

#  VISUALIZE 
plt.figure(figsize=(8,6))
plt.scatter(X_umap[:,0], X_umap[:,1], c=labels_kmeans, s=5, cmap='Spectral', alpha=0.7)
plt.title(f"KMeans Clusters via UMAP (Silhouette={sil_kmeans:.3f})")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.colorbar(label="Cluster")
plt.show()

#  CLUSTER SUMMARY 
df["cluster"] = labels_kmeans
cluster_summary = df.groupby("cluster")[available_features].mean().round(2)
print("\n Cluster Summary ")
print(cluster_summary)

df.to_csv("../../data/clustered_umap_hdb_kmeans.csv", index=False)
print(" Saved clustered dataset with both KMeans + HDBSCAN outputs.")
