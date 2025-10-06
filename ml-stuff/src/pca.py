from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#  VISUALIZATION 
def visualize_kmeans(X_scaled, df, kmeans, features, max_display=15):
    """
    Visualizes KMeans results using PCA (2D and optional 3D).
    Also shows top mean feature values per cluster.
    """
    print("\n Visualizing Clusters ")

    # --- PCA 2D reduction ---
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df["pca1"], df["pca2"] = X_pca[:, 0], X_pca[:, 1]

    plt.figure(figsize=(9, 7))
    sns.scatterplot(
        data=df,
        x="pca1", y="pca2",
        hue="cluster",
        palette="tab10",
        alpha=0.7,
        s=40,
        edgecolor="none"
    )
    plt.title("K-Means Clusters (PCA 2D Projection)")
    plt.xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    plt.ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # --- Optional 3D Plot ---
    try:
        from mpl_toolkits.mplot3d import Axes3D
        pca3 = PCA(n_components=3, random_state=42)
        X_pca3 = pca3.fit_transform(X_scaled)
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(
            X_pca3[:, 0], X_pca3[:, 1], X_pca3[:, 2],
            c=df["cluster"], cmap="tab10", alpha=0.7
        )
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_zlabel("PCA 3")
        ax.set_title("3D Cluster Projection (PCA)")
        plt.colorbar(scatter, ax=ax)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"(3D PCA skipped: {e})")

    # --- Cluster Feature Profiles ---
    cluster_means = df.groupby("cluster")[features].mean()
    top_features = (
        cluster_means.var().sort_values(ascending=False).head(max_display).index
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        cluster_means[top_features],
        cmap="coolwarm",
        annot=False,
        cbar=True,
    )
    plt.title("Top Feature Differences Between Clusters")
    plt.xlabel("Features")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.show()
