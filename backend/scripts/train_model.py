
import os
import pandas as pd
import numpy as np
import joblib
import umap.umap_ as umap
import hdbscan
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# ================= CONFIG ===================
DATA_PATH = "../../data/normalized_matches.csv"
OUTPUT_DIR = "../../data/assigned_clusters"
MODEL_DIR = "models"
ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "SUPPORT"]
AGGREGATE_BY_PLAYER = True
MAX_ZSCORE = 4.0
MIN_CLUSTER_SIZE = 40
N_NEIGHBORS = 30
MIN_DIST = 0.1
RANDOM_STATE = 42

# ================= FEATURES ===================
FEATURES = [
    "kills", "deaths", "assists", "damage", "cs", "role",
    "kp_rate", "damage_share",
    "objective_focus", "survivability_ratio", "vision_efficiency",
    "primarystyle_id", "primarystyle_perk1", "primarystyle_perk2",
    "primarystyle_perk3", "primarystyle_perk4",
    "substyle_id", "substyle_perk1", "substyle_perk2",
    "ch_kda", "ch_killingSprees",
    "ch_teamDamagePercentage",
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

# =====================================================
def train_role_cluster(role: str):
    print(f"\n=== TRAINING ROLE: {role} ===")

    # ---------- Load ----------
    df = pd.read_csv(DATA_PATH)
    df = df[df["role"] == role]

    if df.empty:
        print(f"[WARN] No data found for role {role}, skipping.")
        return

    df = df[[c for c in FEATURES if c in df.columns] + ["puuid", "duration"]]
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    df = df[df["duration"] > 600]
    df = df[(df["kills"] + df["deaths"] + df["assists"]) > 3]

    if AGGREGATE_BY_PLAYER:
        agg_cols = [c for c in df.columns if c not in ["puuid", "role", "_id", "duration"]]
        df = df.groupby("puuid")[agg_cols].mean().reset_index()

    print(f"[INFO] Training on {len(df)} samples for role {role}")

    # ---------- Feature Prep ----------
    cat_cols = ["role", "primarystyle_id", "substyle_id"]
    cat_cols = [c for c in cat_cols if c in df.columns]
    num_cols = [c for c in FEATURES if c not in cat_cols and c in df.columns]

    # log transform positive numerics
    pos_cols = [c for c in num_cols if (df[c] >= 0).all()]
    df[pos_cols] = np.log1p(df[pos_cols])

    # fill NaNs
    df[num_cols] = df[num_cols].fillna(0)

    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer([
        ("num", scaler, num_cols),
        ("cat", encoder, cat_cols)
    ])

    X = preprocessor.fit_transform(df)

    scaler.fit(df[num_cols])
    if cat_cols:
        encoder.fit(df[cat_cols])

    # ---------- Outlier Removal ----------
    mask = np.all(np.abs(X) < MAX_ZSCORE, axis=1)
    removed = len(X) - mask.sum()
    if removed > 0:
        print(f"[INFO] Removed {removed} outliers (> {MAX_ZSCORE}σ).")
    X = X[mask]
    df = df.iloc[mask].reset_index(drop=True)

    # ---------- Dimensionality Reduction ----------
    reducer = umap.UMAP(
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        metric="cosine",
        random_state=RANDOM_STATE
    )
    embedding = reducer.fit_transform(X)

    # ---------- Clustering ----------
    clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, metric="euclidean",prediction_data=True)
    labels = clusterer.fit_predict(embedding)
    df["cluster"] = labels

    # GMM soft clustering (optional)
    gmm = GaussianMixture(n_components=8, covariance_type='full', random_state=RANDOM_STATE)
    gmm_labels = gmm.fit_predict(embedding)
    gmm_probs = gmm.predict_proba(embedding)
    df["gmm_cluster"] = gmm_labels
    df["gmm_confidence"] = gmm_probs.max(axis=1)

    # ---------- Save Results ----------
    role_dir = os.path.join(MODEL_DIR, role)
    os.makedirs(role_dir, exist_ok=True)
    out_dir = os.path.join(OUTPUT_DIR, role)
    os.makedirs(out_dir, exist_ok=True)

    df.to_csv(os.path.join(out_dir, f"{role}_clustered.csv"), index=False)

    # Save models
    joblib.dump(scaler, f"{role_dir}/scaler.pkl")
    joblib.dump(encoder, f"{role_dir}/encoder.pkl")
    joblib.dump(reducer, f"{role_dir}/umap_reducer.pkl")
    joblib.dump(clusterer, f"{role_dir}/hdbscan_model.pkl")
    joblib.dump(gmm, f"{role_dir}/gmm_model.pkl")
    joblib.dump(FEATURES, f"{role_dir}/features.pkl")

    # ---------- Visualization ----------
    plt.figure(figsize=(9, 7))
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1],
                    hue=df["cluster"], palette="tab10", s=10, alpha=0.8, legend=False)
    plt.title(f"{role} Clusters (UMAP + HDBSCAN)")
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{role}_cluster_plot.png"), dpi=300)
    plt.close()

    # ---------- Summary ----------
    summary = df.groupby("cluster").mean(numeric_only=True)
    summary.to_csv(os.path.join(out_dir, f"{role}_cluster_signatures.csv"))

    print(f"Finished {role}: {len(summary)} clusters, models saved to {role_dir}")

# =====================================================
def train_all_roles():
    for role in ROLES:
        train_role_cluster(role)
    print("\nAll roles processed successfully.")

# =====================================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    train_all_roles()
