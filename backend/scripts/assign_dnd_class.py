# =====================================================
# assign_dnd_class.py
# 1. Compute cluster signatures per role
# 2. Map cluster profiles to D&D classes based on similarity
# 3. Aggregate weighted class distribution by role frequency
# =====================================================

import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))

# --- Paths ---
APP_DATA_DIR = os.path.join(PROJECT_ROOT, "backend", "app", "data")
PLAYER_DIR = os.path.join(PROJECT_ROOT, "backend", "app","data", "players")

# Debug info (optional, helps confirm path resolution)
print(f"[DEBUG] PROJECT_ROOT = {PROJECT_ROOT}")
print(f"[DEBUG] APP_DATA_DIR = {APP_DATA_DIR}")
print(f"[DEBUG] PLAYER_DIR   = {PLAYER_DIR}")
def assign_dnd_classes(player_name: str):
    """
    Given a player's clustered match data, compute D&D class mapping and weighted summary.
    Returns path to the generated summary CSV.
    """
    # -------------------------------------------------
    # PATH SETUP
    # -------------------------------------------------
    player_csv = os.path.join(PLAYER_DIR, f"{player_name}_data_clustered.csv")
    cluster_sig_path = os.path.join(PLAYER_DIR, f"{player_name}_signatures.csv")
    cluster_map_path = os.path.join(PLAYER_DIR, f"{player_name}_cluster_dnd_mapping.csv")

    dnd_json_path = os.path.join(APP_DATA_DIR, "dnd_class.json")
    summary_path = os.path.join(APP_DATA_DIR, "dnd_class_summary_weighted.csv")

    # -------------------------------------------------
    # STEP 1 — Generate Cluster Signatures
    # -------------------------------------------------
    print(f"[1/5] Computing cluster signatures for {player_name}...")

    if not os.path.exists(player_csv):
        raise FileNotFoundError(f"Player data not found at {player_csv}")

    df = pd.read_csv(player_csv)

    # Select only numeric columns (exclude 'cluster' for duplication)
    numeric_cols = df.select_dtypes(include=[float, int]).columns.drop(["cluster"], errors="ignore")

    # Group by role and cluster, compute means
    cluster_profiles = (
        df.groupby(["role_str", "cluster"], as_index=False)[numeric_cols]
          .mean()
    )

    os.makedirs(os.path.dirname(cluster_sig_path), exist_ok=True)
    cluster_profiles.to_csv(cluster_sig_path, index=False)
    print(f"Saved cluster signatures to {cluster_sig_path}")

    # -------------------------------------------------
    # STEP 2 — Load Cluster Signatures & D&D JSON
    # -------------------------------------------------
    print("[2/5] Loading cluster signatures and D&D class profiles...")

    clusters = pd.read_csv(cluster_sig_path)

    # Select numeric stats only
    numeric_cols = clusters.select_dtypes(include=[np.number]).columns
    X = clusters[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Normalize stats
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=numeric_cols)

    # Load D&D JSON profiles
    with open(dnd_json_path, "r") as f:
        dnd_data = json.load(f)

    # Extract numeric feature weights
    class_vectors = {}
    for cname, features in dnd_data.items():
        vec = {feat.lower(): val for feat, val in features.items() if isinstance(val, (int, float))}
        class_vectors[cname] = vec
    # -------------------------------------------------
    # STEP 2.5 — Feature Remapping for JSON Compatibility
    # -------------------------------------------------
    FEATURE_MAP = {
        "kills": "ch_kills",
        "deaths": "ch_deaths",
        "assists": "ch_assists",
        "damage": "ch_totalDamageDealtToChampions",
        "damage_share": "ch_teamDamagePercentage",
        "cs": "ch_totalMinionsKilled",
        "vision": "ch_visionScore",
        "gold": "ch_goldPerMinute",
        "healing": "ch_totalHealsOnTeammates",
        "tankiness": "ch_damageTakenOnTeamPercentage",
        "objectives": "ch_turretTakedowns",
        "survivability": "ch_survivedSingleDigitHpCount"
    }

    # Remap feature names to match clustered CSV columns
    remapped_vectors = {}
    for cname, featmap in class_vectors.items():
        new_vec = {}
        for feat, val in featmap.items():
            key = FEATURE_MAP.get(feat, feat)  # rename if mapping exists
            new_vec[key.lower()] = val
        remapped_vectors[cname] = new_vec

    class_vectors = remapped_vectors

    # Debug check — print overlap rate
    sample_keys = list(X.columns.str.lower())
    for cname, feats in list(class_vectors.items())[:3]:
        overlap = len(set(feats.keys()) & set(sample_keys))
        print(f"[DEBUG] {cname}: {overlap} overlapping features")
    # -------------------------------------------------
    # STEP 3 — Compute Class Similarities
    # -------------------------------------------------
    print("[3/5] Computing similarities...")

    cluster_features = [c.lower() for c in numeric_cols]
    cluster_matrix = X_scaled.values

    class_matrix = []
    class_names = []
    for cname, featmap in class_vectors.items():
        row = [featmap.get(f, 0) for f in cluster_features]
        class_matrix.append(row)
        class_names.append(cname)
    class_matrix = np.array(class_matrix)

    # Compute cosine similarity
    similarities = cosine_similarity(cluster_matrix, class_matrix)
    best_idx = similarities.argmax(axis=1)
    best_scores = similarities.max(axis=1)
    best_names = [class_names[i] for i in best_idx]

    # Combine results
    result = clusters.copy()
    result["DND_Class"] = best_names
    result["Similarity"] = best_scores

    # Top-3 suggestions
    top3 = np.argsort(similarities, axis=1)[:, -3:][:, ::-1]
    result["Top3"] = [
        ", ".join([f"{class_names[i]} ({similarities[r, i]:.2f})" for i in row])
        for r, row in enumerate(top3)
    ]

    # Save mapping
    result.to_csv(cluster_map_path, index=False)
    print(f"Saved cluster -> D&D mapping to {cluster_map_path}")

    # Print summary
    print("\n=== Cluster -> Class Summary ===")
    for _, row in result.iterrows():
        print(f"[{row['role_str']:<8}] Cluster {int(row['cluster']):>2} -> {row['DND_Class']} ({row['Similarity']:.2f})")

    # -------------------------------------------------
    # STEP 4 — Weighted Aggregation by Role Frequency
    # -------------------------------------------------
    print("\n[4/5] Aggregating weighted D&D class likelihoods...")

    # Compute role frequency from player data
    role_freq = (
        df["role_str"]
        .value_counts(normalize=True)
        .rename("role_weight")
        .reset_index()
        .rename(columns={"index": "role_str"})
    )
    print(role_freq)

    # Merge role weights into DnD mapping
    merged = result.merge(role_freq, on="role_str", how="left")
    merged["role_weight"] = merged["role_weight"].fillna(0)
    merged["weighted_score"] = merged["Similarity"] * merged["role_weight"]

    # Aggregate by class
    summary = (
        merged.groupby("DND_Class", as_index=False)
        .agg(
            Count=("DND_Class", "size"),
            Avg_Similarity=("Similarity", "mean"),
            Weighted_Influence=("weighted_score", "sum")
        )
        .sort_values("Weighted_Influence", ascending=False)
    )
    summary["Weighted_Share"] = summary["Weighted_Influence"] / summary["Weighted_Influence"].sum() * 100

    summary.to_csv(summary_path, index=False)
    print(f"Saved weighted D&D summary to {summary_path}")

    # Print summary table
    print("\n=== Weighted D&D Class Distribution (Role × Similarity) ===")
    for _, row in summary.iterrows():
        print(
            f"{row['DND_Class']:<10} | Count: {row['Count']:>2} "
            f"| AvgSim: {row['Avg_Similarity']:.2f} "
            f"| WeightedShare: {row['Weighted_Share']:.1f}%"
        )

    # -------------------------------------------------
    # STEP 5 — Optional Visualization
    # -------------------------------------------------
    #try:
        #import matplotlib.pyplot as plt
        #plt.figure(figsize=(10, 5))
        #plt.bar(summary["DND_Class"], summary["Weighted_Share"], edgecolor="black")
        #plt.title("Weighted D&D Class Likelihoods (by Role Frequency × Similarity)")
        #plt.ylabel("Weighted Percentage Share")
        #plt.xticks(rotation=45, ha="right")
       #plt.tight_layout()
        #plt.show()
    #except ImportError:
        #print("(matplotlib not installed — skipping chart)")
    print("\n[5/5] D&D class assignment completed.")
    print(summary_path)
    
    return summary_path
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        player_name = sys.argv[1]
    else:
        player_name = ""  # default for testing

    print(f"Running D&D class assignment for {player_name}")
    assign_dnd_classes(player_name)
