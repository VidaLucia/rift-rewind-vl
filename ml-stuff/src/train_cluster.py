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
from sklearn.metrics import pairwise_distances
from scipy.stats import zscore


# CONFIG

DATA_PATH = "../../data/normalized_matches.csv"
MODEL_DIR = "../../models"
OUT_DIR = "../../data/player_analysis"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

best_k = 26
DISCOVERY_UMAP = False
DISCOVERY = False


# LOAD DATA

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


# BALANCE DATA BY ROLE

max_per_role = 20000
df = (
    df.groupby("role", group_keys=False)
    .apply(lambda x: x.sample(min(len(x), max_per_role), random_state=42))
    .reset_index(drop=True)
)
print(f"Balanced dataset by role — new size: {len(df)}")


# DERIVED FEATURES

print("Generating derived features...")
df["kp_rate"] = df["ch_killParticipation"] / 100
df["damage_share"] = df["ch_teamDamagePercentage"] / 100
df["gold_efficiency"] = df["ch_goldPerMinute"] / df["duration"].clip(lower=1)
df["survivability_ratio"] = df["kills"] / df["deaths"].clip(lower=1)
df["objective_focus"] = (
    df["ch_turretTakedowns"] + df["ch_baronTakedowns"] + df["ch_dragonTakedowns"]
) / 3
df["kpm"] = df["kills"] / df["duration"].clip(lower=1)
#df["dpm"] = df["damage"] / df["duration"].clip(lower=1)
df["apm"] = df["assists"] / df["duration"].clip(lower=1)
df["cspm"] = df["cs"] / df["duration"].clip(lower=1)
df["vision_efficiency"] = df["ch_visionScorePerMinute"] / (df["ch_wardTakedowns"] + 1)


# FEATURE SELECTION

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
features.extend([c for c in df.columns if c.startswith("tag_")])

X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)


# FEATURE WEIGHTING

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


# SCALING

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Scaled features.")


# UMAP DIMENSIONALITY REDUCTION

best_umap_params = {
    "n_neighbors": 40,
    "min_dist": 0.15,
    "n_components": 10,
}
reducer = umap.UMAP(**best_umap_params, metric="euclidean", random_state=42)
X_embed = reducer.fit_transform(X_scaled)
print(f"UMAP completed with shape: {X_embed.shape}")

# UMAP VISUALIZATION

print("Generating UMAP visualization...")

plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    X_embed[:, 0],
    X_embed[:, 1],
    s=5,
    c=df["role"],  # You can also use df["cluster"] after clustering
    cmap="Spectral",
    alpha=0.6
)
plt.colorbar(scatter, label="Role (pre-clustering)")
plt.title("UMAP Projection of Player Feature Space")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.tight_layout()

umap_vis_path = os.path.join(OUT_DIR, "umap_projection.png")
plt.savefig(umap_vis_path, dpi=300)
plt.close()

print(f"UMAP projection visualization saved to {umap_vis_path}")

# SAVE METADATA

with open(os.path.join(MODEL_DIR, "umap_metadata.json"), "w") as f:
    json.dump({"best_params": best_umap_params}, f, indent=2)


# CLUSTERING

kmeans = MiniBatchKMeans(
    n_clusters=best_k,
    random_state=42,
    batch_size=4096,
    max_iter=200,
    n_init=10
)
df["cluster"] = kmeans.fit_predict(X_embed)
import matplotlib.patheffects as pe


# UMAP CLUSTER VISUALIZATION

print("Generating UMAP cluster visualization...")

plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    X_embed[:, 0],
    X_embed[:, 1],
    s=5,
    c=df["cluster"],
    cmap="tab20",
    alpha=0.7
)
plt.colorbar(scatter, label="Cluster ID")
plt.title(f"UMAP Projection Colored by Cluster (k={best_k})")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.tight_layout()

umap_cluster_vis_path = os.path.join(OUT_DIR, "umap_clusters.png")
plt.savefig(umap_cluster_vis_path, dpi=300)
plt.close()

print(f"Cluster-colored UMAP projection saved to {umap_cluster_vis_path}")

# QUICK SILHOUETTE CHECK
sample_idx = np.random.choice(len(X_embed), min(10000, len(X_embed)), replace=False)
sil_score = silhouette_score(X_embed[sample_idx], kmeans.labels_[sample_idx])
print(f"Quick Silhouette Score: {sil_score:.4f}")


# SAVE ARTIFACTS

dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
dump(reducer, os.path.join(MODEL_DIR, "umap_reducer.pkl"))
dump(kmeans, os.path.join(MODEL_DIR, "kmeans_model.pkl"))
print(f"Saved models to {MODEL_DIR}")


# CLUSTER SUMMARY

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
    bar = "X" * int(pct / 2)
    print(f"Cluster {c:>2}: {n:>6} samples ({pct:5.2f}%) {bar}")
print("=" * 50)
print(f"Cluster distribution saved to: {summary_path}")


# FINAL CLUSTER REPORT GENERATION

print("\nGenerating full cluster analysis report...")
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
X_scaled_df["cluster"] = df["cluster"]

cluster_profiles = {}
report_lines = []

global_mean = X_scaled_df.drop(columns=["cluster"]).mean()

for c in sorted(df["cluster"].unique()):
    cluster_data = X_scaled_df[X_scaled_df["cluster"] == c]
    cluster_mean = cluster_data.drop(columns=["cluster"]).mean()
    delta = cluster_mean - global_mean
    delta_sorted = delta.sort_values(ascending=False)

    cluster_profiles[c] = {
        "sample_count": int(len(cluster_data)),
        "feature_deltas": {feat: float(delta[feat]) for feat in delta.index},
    }

    report_lines.append(f"\n Cluster {c} ")
    report_lines.append(f"Samples: {len(cluster_data)}")
    report_lines.append("-" * 50)
    report_lines.append(f"{'Feature':<35} {'Δ Mean (cluster - global)':>25}")
    report_lines.append("-" * 50)
    for feat, val in delta_sorted.items():
        arrow = "+" if val > 0 else "-"
        report_lines.append(f"{feat:<35} {arrow}{abs(val):>8.4f}")
    report_lines.append("")

# Save human-readable report
report_path = os.path.join(OUT_DIR, "clusters_full_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"Saved detailed cluster report to {report_path}")


# HYBRID ROLE-AWARE CLUSTER ANALYSIS

print("\n[HYBRID] Computing role-aware cluster analysis...")

role_map_rev = {1: "TOP", 2: "JUNGLE", 3: "MIDDLE", 4: "BOTTOM", 5: "SUPPORT"}
role_names = [role_map_rev.get(r, f"ROLE_{r}") for r in sorted(df["role"].unique())]

#  Cluster composition by role 
role_cluster_counts = (
    df.groupby(["role", "cluster"]).size().unstack(fill_value=0)
    .reindex(sorted(df["role"].unique()))
)
role_cluster_pct = role_cluster_counts.div(role_cluster_counts.sum(axis=1), axis=0) * 100

print("\nRole composition across clusters (% within each role):")
for role_id in sorted(df["role"].unique()):
    role_label = role_map_rev.get(role_id, str(role_id))
    row = role_cluster_pct.loc[role_id].sort_values(ascending=False)
    top3 = row.head(3)
    formatted = ", ".join([f"{int(c)} ({pct:.1f}%)" for c, pct in zip(top3.index, top3.values)])
    print(f"  {role_label:<8}: top clusters -> {formatted}")

#  Cluster composition summary (cluster -> role %) 
cluster_role_counts = (
    df.groupby(["cluster", "role"]).size().unstack(fill_value=0)
    .reindex(sorted(df["cluster"].unique()))
)
cluster_role_pct = cluster_role_counts.div(cluster_role_counts.sum(axis=1), axis=0) * 100

print("\nCluster composition by role (% of players in each cluster):")
for cluster_id in sorted(df["cluster"].unique()):
    row = cluster_role_pct.loc[cluster_id].sort_values(ascending=False)
    top_roles = row.head(3)
    formatted = ", ".join(
        [f"{role_map_rev.get(r, r)}: {pct:.1f}%" for r, pct in zip(top_roles.index, top_roles.values)]
    )
    print(f"  Cluster {cluster_id:>2}: {formatted}")

#  Compute per-role feature deltas within clusters 
print("\nComputing per-role feature deltas within each cluster...")
X_scaled_df["role"] = df["role"]

role_cluster_deltas = {}
for role_id, role_df in X_scaled_df.groupby("role"):
    role_label = role_map_rev.get(role_id, f"ROLE_{role_id}")
    global_mean = role_df.drop(columns=["role", "cluster"]).mean()
    role_cluster_deltas[role_label] = {}

    for c in sorted(role_df["cluster"].unique()):
        cdata = role_df[role_df["cluster"] == c]
        cluster_mean = cdata.drop(columns=["role", "cluster"]).mean()
        delta = cluster_mean - global_mean
        top_feats = delta.abs().sort_values(ascending=False).head(10)
        role_cluster_deltas[role_label][int(c)] = {
            "sample_count": int(len(cdata)),
            "top_feature_deltas": {feat: float(delta[feat]) for feat in top_feats.index},
        }

#  Save hybrid summary JSON 
hybrid_summary_path = os.path.join(OUT_DIR, "role_cluster_summary.json")
hybrid_summary = {
    "role_cluster_counts": role_cluster_counts.to_dict(),
    "cluster_role_counts": cluster_role_counts.to_dict(),
    "role_cluster_deltas": role_cluster_deltas,
}
with open(hybrid_summary_path, "w") as f:
    json.dump(hybrid_summary, f, indent=2)

print(f"\n[HYBRID] Saved role-cluster summary to {hybrid_summary_path}")

df.to_csv(os.path.join(OUT_DIR, "labeled_matches.csv"), index=False)
print("Saved full dataset with cluster labels to labeled_matches.csv")


# HYBRID UMAP CLUSTER VISUALIZATION (COLORED + ROLE LABELS)

print("Generating hybrid UMAP visualization (cluster + dominant role labels)...")

#  Compute cluster centroids in UMAP space
centroids = np.vstack([
    X_embed[df["cluster"] == c].mean(axis=0)
    for c in sorted(df["cluster"].unique())
])

#  Determine dominant role per cluster
role_map_rev = {1: "TOP", 2: "JUNGLE", 3: "MIDDLE", 4: "BOTTOM", 5: "SUPPORT"}
dominant_roles = {
    c: role_map_rev.get(cluster_role_counts.loc[c].idxmax(), "?")
    for c in cluster_role_counts.index
}

#  Create scatter colored by cluster ID
plt.figure(figsize=(9, 7), dpi=300)
scatter = plt.scatter(
    X_embed[:, 0],
    X_embed[:, 1],
    s=4,
    c=df["cluster"],
    cmap="tab20",
    alpha=0.8,
    linewidths=0,
)

#  Overlay centroid markers
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    c="black",
    s=45,
    marker="X",
    edgecolors="white",
    linewidths=0.7,
    zorder=5,
)

#  Label each centroid with "Cluster: Role"
for c, centroid in zip(sorted(df["cluster"].unique()), centroids):
    x, y = centroid[0], centroid[1]
    label = f"{c}: {dominant_roles.get(c, '?')}"
    plt.text(
        x, y, label,
        fontsize=6.5,
        color="white",
        ha="center", va="center",
        fontweight="bold",
        path_effects=[pe.withStroke(linewidth=2.2, foreground="black")],
    )

plt.title(f"UMAP Projection Colored by Cluster (k={best_k})\nLabels show Dominant Role per Cluster")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.axis("off")
plt.tight_layout()

hybrid_vis_path = os.path.join(OUT_DIR, "umap_clusters_roles.png")
plt.savefig(hybrid_vis_path, bbox_inches="tight")
plt.close()

print(f"Saved hybrid cluster+role UMAP visualization to {hybrid_vis_path}")
#  Optional heatmap preview 
try:
    plt.figure(figsize=(10, 5))
    plt.imshow(cluster_role_pct, cmap="viridis", aspect="auto")
    plt.colorbar(label="% of cluster made up by each role")
    plt.title("Cluster Composition by Role")
    plt.xlabel("Cluster ID")
    plt.ylabel("Role ID")
    plt.yticks(ticks=range(len(role_names)), labels=role_names)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "role_cluster_heatmap.png"))
    plt.close()
    print(f"[HYBRID] Heatmap saved to role_cluster_heatmap.png")
except Exception as e:
    print(f"[HYBRID] Skipped visualization ({e})")

print("\n Hybrid role-aware clustering analysis complete.")
