#  Headless plotting + stable warnings up front 
import os
import tempfile
import warnings

import matplotlib
matplotlib.use("Agg")  # no Tkinter windows; prevents threading errors

# Make joblib memmap cleanup less noisy and deterministic on Windows
os.environ.setdefault("JOBLIB_TEMP_FOLDER", os.path.join(tempfile.gettempdir(), "joblib_memmap"))
os.makedirs(os.environ["JOBLIB_TEMP_FOLDER"], exist_ok=True)
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

import numpy as np
import pandas as pd
import datetime
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed


warnings.filterwarnings("ignore")


class ClusterDifferentiationAnalyzer:
    """
    ANOVA, Random Forest, pairwise t-tests, effect sizes and SHAP explanations
    aligned to the training feature set.
    """

    def __init__(self, csv_path, cluster_columns="cluster", output_dir=None):
        print(" Cluster Differentiation Analyzer (Final + SHAP edition)")

        self.df = pd.read_csv(csv_path)
        self.cluster_col = cluster_columns

        # Match training feature set
        TRAIN_FEATURES = [
            "kills", "deaths", "assists", "damage", "cs", "role",
            "kp_rate", "damage_share", "gold_efficiency",
            "objective_focus", "survivability_ratio", "vision_efficiency",
            "primarystyle_id", "primarystyle_perk1", "primarystyle_perk2", "primarystyle_perk3",
            "substyle_id", "substyle_perk1", "substyle_perk2",
            "ch_kda", "ch_killingSprees", "ch_damagePerMinute",
            "ch_teamDamagePercentage", "ch_killParticipation",
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
        tag_cols = [c for c in self.df.columns if c.startswith("tag_")]
        TRAIN_FEATURES.extend(tag_cols)

        # Keep numeric + selected features only; drop rows with missing among selected
        numeric_df = self.df.select_dtypes(include=[np.number]).copy()
        available_features = [f for f in TRAIN_FEATURES if f in numeric_df.columns]
        cols = available_features + ([self.cluster_col] if self.cluster_col in numeric_df.columns else [])
        if self.cluster_col not in numeric_df.columns:
            raise ValueError(f"Cluster column '{self.cluster_col}' not found or not numeric in input CSV.")
        self.df = numeric_df[cols].replace([np.inf, -np.inf], np.nan).dropna()

        self.features = available_features
        self.clusters = sorted(self.df[self.cluster_col].unique())

        # Output directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.outdir = output_dir or f"cluster_differentiation_reports/{timestamp}"
        os.makedirs(self.outdir, exist_ok=True)

        print(f"\nLoaded {len(self.df):,} samples | {len(self.clusters)} clusters | {len(self.features)} features")
        print(f"Detected {len(tag_cols)} tag features: {tag_cols}")
        print(f"Output directory: {self.outdir}\n")

        self.results = {}

    #  ANOVA / Kruskal-Wallis 
    def run_anova_kruskal(self, top_n=20, nonparametric=False):
        print(f" Running {'Kruskal-Wallis' if nonparametric else 'ANOVA'} on {len(self.features)} features")

        skipped = []

        def analyze_feature(feature):
            groups = [self.df[self.df[self.cluster_col] == c][feature] for c in self.clusters]
            # Skip constant/invalid
            if any(g.nunique() <= 1 for g in groups) or self.df[feature].nunique() <= 1:
                skipped.append(feature)
                return (feature, np.nan, np.nan, np.nan)
            try:
                if nonparametric:
                    stat, p = stats.kruskal(*groups)
                else:
                    stat, p = stats.f_oneway(*groups)
            except Exception:
                skipped.append(feature)
                return (feature, np.nan, np.nan, np.nan)

            grand_mean = self.df[feature].mean()
            ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
            ss_total = float(((self.df[feature] - grand_mean) ** 2).sum())
            eta_squared = ss_between / ss_total if ss_total > 0 else 0.0
            return (feature, float(stat), float(p), float(eta_squared))

        results = Parallel(n_jobs=-1)(delayed(analyze_feature)(f) for f in self.features)

        df = pd.DataFrame(results, columns=["feature", "statistic", "p_value", "eta_squared"])
        df = df.dropna(subset=["p_value"])
        if len(df) == 0:
            raise RuntimeError("All features were constant across clusters; ANOVA cannot proceed.")
        df["p_value"] = df["p_value"].replace([np.inf, -np.inf], 1.0)
        df["p_adj"] = multipletests(df["p_value"], method="fdr_bh")[1]
        df["significant"] = df["p_adj"] < 0.05
        df = df.sort_values(["p_adj", "statistic"]).reset_index(drop=True)

        print(f"{'Rank':<6}{'Feature':<40}{'Stat':<12}{'p-adj':<12}{'η²':<12}{'Sig'}")
        for i, row in df.head(top_n).iterrows():
            sig = "***" if row["p_adj"] < 1e-3 else "**" if row["p_adj"] < 1e-2 else "*" if row["p_adj"] < 0.05 else ""
            print(f"{i:<6}{row['feature']:<40}{row['statistic']:<12.2f}{row['p_adj']:<12.6f}{row['eta_squared']:<12.4f}{sig}")

        if skipped:
            head = ", ".join(skipped[:10])
            tail = "..." if len(skipped) > 10 else ""
            print(f"\nSkipped {len(skipped)} constant features: {head}{tail}")

        df.to_csv(os.path.join(self.outdir, "anova_feature_differences.csv"), index=False)
        self.results["anova"] = df
        return df

    #  Random Forest + SHAP 
    def random_forest_feature_importance(self, top_n=20, shap_sample=500):
        X = self.df[self.features]
        y = self.df[self.cluster_col]

        rf = RandomForestClassifier(
            n_estimators=300, max_depth=20, random_state=42, n_jobs=-1
        )
        rf.fit(X, y)

        importances_df = (
            pd.DataFrame({"feature": self.features, "importance": rf.feature_importances_})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        cv_scores = cross_val_score(rf, X, y, cv=5, n_jobs=-1)
        print(f"RF Accuracy: {rf.score(X, y)*100:.2f}% | CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

        importances_df.to_csv(os.path.join(self.outdir, "rf_feature_importances.csv"), index=False)
        self.results["rf_model"] = rf
        self.results["random_forest"] = importances_df

        # --------------------------- Fast, multiclass-safe SHAP ------------------------
        print(" Computing SHAP values (fast mode: balanced subsample + approximate=True)")
        # Balanced subsample: up to 50 per cluster (adjust if you want even faster)
        per_cluster = 50
        X_sample = (
            self.df.groupby(self.cluster_col, group_keys=False)
            .apply(lambda x: x.sample(min(len(x), per_cluster), random_state=42))[self.features]
        )
        # Cap absolute sample (in case of many clusters)
        shap_sample = min(shap_sample, len(X_sample))
        X_sample = X_sample.iloc[:shap_sample]
        print(f" SHAP sample size: {len(X_sample)}")

        # Approximate TreeExplainer for speed
        explainer = shap.TreeExplainer(rf, feature_perturbation="interventional", approximate=True)
        shap_values = explainer.shap_values(X_sample)

        # Multiclass → list of arrays; build global mean(|SHAP|) and per-class plots
        if isinstance(shap_values, list):
            shap_values_mean = np.mean(np.abs(np.stack(shap_values, axis=0)), axis=0)  # (n_samples, n_features)
            shap.summary_plot(shap_values_mean, X_sample, feature_names=self.features, show=False)
            plt.title("SHAP Summary (Mean |SHAP| across clusters)")
            plt.tight_layout()
            plt.savefig(os.path.join(self.outdir, "shap_summary_all_clusters.png"))
            plt.close()

            # Per-class
            for i, vals in enumerate(shap_values):
                shap.summary_plot(vals, X_sample, feature_names=self.features, show=False)
                plt.title(f"SHAP Summary - Cluster {self.clusters[i] if i < len(self.clusters) else i}")
                plt.tight_layout()
                plt.savefig(os.path.join(self.outdir, f"shap_cluster_{self.clusters[i] if i < len(self.clusters) else i}.png"))
                plt.close()
        else:
            shap.summary_plot(shap_values, X_sample, feature_names=self.features, show=False)
            plt.title("SHAP Summary (Single Output)")
            plt.tight_layout()
            plt.savefig(os.path.join(self.outdir, "shap_summary_all_clusters.png"))
            plt.close()

        print(" SHAP plots saved")
        return importances_df

    #  Pairwise Cluster Diff 
    def pairwise_cluster_differentiation(self, c1, c2, top_n=15):
        print(f" Pairwise comparison: Cluster {c1} vs Cluster {c2}")
        d1 = self.df[self.df[self.cluster_col] == c1]
        d2 = self.df[self.df[self.cluster_col] == c2]
        diffs = []
        for f in self.features:
            v1, v2 = d1[f].values, d2[f].values
            t, p = stats.ttest_ind(v1, v2, equal_var=False)
            pooled_std = np.sqrt(
                ((len(v1) - 1) * v1.std(ddof=1) ** 2 + (len(v2) - 1) * v2.std(ddof=1) ** 2)
                / max(1, (len(v1) + len(v2) - 2))
            )
            d = (v1.mean() - v2.mean()) / pooled_std if pooled_std > 0 else 0.0
            diffs.append((f, v1.mean(), v2.mean(), d, p))

        df = pd.DataFrame(diffs, columns=["feature", "mean_c1", "mean_c2", "cohens_d", "p_value"])
        df["p_adj"] = multipletests(df["p_value"], method="fdr_bh")[1]
        df["abs_d"] = df["cohens_d"].abs()
        df = df.sort_values("abs_d", ascending=False).reset_index(drop=True)

        print(f"{'Rank':<6}{'Feature':<35}{'Cohen’s d':<12}{'p-adj':<12}{'Sig'}")
        for i, row in df.head(top_n).iterrows():
            sig = "***" if row["p_adj"] < 1e-3 else "**" if row["p_adj"] < 1e-2 else "*" if row["p_adj"] < 0.05 else ""
            print(f"{i:<6}{row['feature']:<35}{row['cohens_d']:<12.4f}{row['p_adj']:<12.6f}{sig}")

        out = os.path.join(self.outdir, f"pairwise_cluster_{c1}_vs_{c2}.csv")
        df.to_csv(out, index=False)
        self.results[f"pairwise_{c1}_vs_{c2}"] = df
        return df

    # Cluster Signatures 
    def cluster_signatures(self, top_n=5):
        print(" Cluster Signatures (Top features per cluster)")
        signatures = {}
        for c in self.clusters:
            c_data = self.df[self.df[self.cluster_col] == c]
            others = self.df[self.df[self.cluster_col] != c]
            effects = []
            for f in self.features:
                v1, v2 = c_data[f].values, others[f].values
                pooled_std = np.sqrt(
                    ((len(v1) - 1) * v1.std(ddof=1) ** 2 + (len(v2) - 1) * v2.std(ddof=1) ** 2)
                    / max(1, (len(v1) + len(v2) - 2))
                )
                d = (v1.mean() - v2.mean()) / pooled_std if pooled_std > 0 else 0.0
                effects.append((f, v1.mean(), v2.mean(), d))
            df = pd.DataFrame(effects, columns=["feature", "cluster_mean", "others_mean", "effect_size"])
            df["abs_effect"] = df["effect_size"].abs()
            df = df.sort_values("abs_effect", ascending=False).reset_index(drop=True)
            df.to_csv(os.path.join(self.outdir, f"cluster_{c}_signature.csv"), index=False)
            signatures[c] = df

            print(f"\nCluster {c} Signature:")
            for _, r in df.head(top_n).iterrows():
                direction = "↑" if r["effect_size"] > 0 else "↓"
                print(f"  {direction} {r['feature']:<35} ({r['effect_size']:+.3f})")

        self.results["signatures"] = signatures
        return signatures

    # LDA 
    def linear_discriminant_analysis(self, n_components=2, use_pca=False):
        print("\n" + "=" * 80)
        print("METHOD 5: Linear Discriminant Analysis (LDA)")
        print("=" * 80)

        X, y = self.df[self.features].values, self.df[self.cluster_col].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if use_pca and X_scaled.shape[1] > 100:
            print("Applying PCA before LDA (retain 95% variance)")
            X_scaled = PCA(n_components=0.95, random_state=42).fit_transform(X_scaled)

        lda = LinearDiscriminantAnalysis(n_components=min(n_components, len(self.clusters) - 1))
        X_lda = lda.fit_transform(X_scaled, y)
        print(f"Explained Variance Ratio: {lda.explained_variance_ratio_}")

        plt.figure(figsize=(8, 6))
        plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap="tab20", s=12, alpha=0.7)
        plt.title("LDA Projection of Clusters")
        plt.xlabel("LD1")
        plt.ylabel("LD2")
        plt.tight_layout()
        plt.savefig(os.path.join(self.outdir, "lda_projection.png"))
        plt.close()

        self.results["lda"] = {"model": lda, "data": X_lda}
        return lda, X_lda

    # Summary + Report 
    def generate_differentiation_report(self):
        print("Generating full report")
        anova = self.run_anova_kruskal()
        rf = self.random_forest_feature_importance()
        _sigs = self.cluster_signatures()
        _lda, _lda_data = self.linear_discriminant_analysis()

        self.plot_top_features(anova, "eta_squared", "Top Features by ANOVA Eta Squared", "top_anova_features.png")
        self.plot_top_features(rf, "importance", "Top Features by Random Forest Importance", "top_rf_features.png")

        summary = [
            f"Samples: {len(self.df)}, Clusters: {len(self.clusters)}, Features: {len(self.features)}",
            "",
            "Top 10 Discriminative Features (η²):"
        ]
        for _, r in anova.head(10).iterrows():
            summary.append(f"  {r['feature']:<35} η²={r['eta_squared']:.3f} (p_adj={r['p_adj']:.5f})")
        with open(os.path.join(self.outdir, "differentiation_summary.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(summary))

        print(f"Report generated in {self.outdir}")

    def plot_top_features(self, df, value_col, title, file_name, top_n=15):
        plt.figure(figsize=(10, 6))
        sns.barplot(y="feature", x=value_col, data=df.head(top_n), palette="viridis")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.outdir, file_name))
        plt.close()


if __name__ == "__main__":
    analyzer = ClusterDifferentiationAnalyzer("labeled_matches.csv", cluster_columns="cluster")
    analyzer.generate_differentiation_report()
