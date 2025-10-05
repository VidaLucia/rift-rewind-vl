import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
DISCOVERY = False
# === Load data ===
df = pd.read_csv("../../data/normalized_matches.csv")
print(f"Loaded {len(df)} rows")

# === Clean NaNs and invalid values ===
df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
print(f"Columns available: {len(df.columns)}")

# === Filter out players with no perk data ===
before = len(df)
df = df[
    ~((df["perk_offense"] == 0) &
      (df["perk_flex"] == 0) &
      (df["perk_defense"] == 0))
]
print(f"Filtered {before - len(df)} rows with all-zero perks, remaining: {len(df)}")

# === Define features ===
features = [
    "kills", "deaths", "assists", "gold_earned", "damage", "vision_score",
    "cs", "win", "duration",
    "ch_damagePerMinute", "ch_killParticipation", "ch_takedowns",
    "ch_deathsByEnemyChamps", "ch_teamDamagePercentage",
    "ch_goldPerMinute", "ch_turretTakedowns"
]

available_features = [f for f in features if f in df.columns]
missing = set(features) - set(available_features)
if missing:
    print(" Missing features:", missing)

X = df[available_features]
print(f"Feature matrix shape: {X.shape}")

if X.empty:
    raise ValueError(" No data left after filtering — check filtering logic!")

# === Scale ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Find optimal K using Elbow + Silhouette ===
K_RANGE = range(4, 41, 2)
inertias, silhouettes = [], []

print("\nRunning KMeans across cluster counts...")
if(DISCOVERY ==True):
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertia = km.inertia_
        inertias.append(inertia)
        
        # sample if data too large for silhouette
        if len(X_scaled) > 10000:
            subset_idx = np.random.choice(len(X_scaled), 10000, replace=False)
            sil = silhouette_score(X_scaled[subset_idx], labels[subset_idx])
        else:
            sil = silhouette_score(X_scaled, labels)
        
        silhouettes.append(sil)
        print(f"K={k:<3} | Inertia={inertia:,.0f} | Silhouette={sil:.4f}")

    # === Plot elbow & silhouette ===
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.plot(K_RANGE, inertias, "o-", color="blue", label="Inertia (Elbow)")
    ax2.plot(K_RANGE, silhouettes, "s--", color="green", label="Silhouette Score")

    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Inertia ", color="blue")
    ax2.set_ylabel("Silhouette ", color="green")
    plt.title("KMeans Elbow + Silhouette Analysis")
    plt.grid(True)
    fig.tight_layout()
    plt.show()


BEST_K = 16  # change after inspecting plot
print(f"\n Using K={BEST_K} for final clustering")

kmeans = KMeans(n_clusters=BEST_K, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

# === Summarize results ===
cluster_summary = df.groupby("cluster")[available_features].mean().round(2)
print("\n=== Cluster Summary ===")
print(cluster_summary)

# === Optionally save labeled data ===
df.to_csv("../../data/labeled_clusters.csv", index=False)
print("\n Saved clustered dataset to ../../data/labeled_clusters.csv")
