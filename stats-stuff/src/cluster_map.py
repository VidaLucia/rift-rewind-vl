import os
import pandas as pd
import glob

# =====================================================
# CONFIG
# =====================================================
REPORT_DIR = "cluster_differentiation_reports/20251010_093554"
ANOVA_PATH = os.path.join(REPORT_DIR, "anova_feature_differences.csv")
RF_PATH = os.path.join(REPORT_DIR, "rf_feature_importances.csv")
OUTPUT_PATH = os.path.join(REPORT_DIR, "cluster_qualitative_summary.csv")

# =====================================================
# LOAD GLOBAL FEATURE IMPORTANCE SOURCES
# =====================================================
anova = pd.read_csv(ANOVA_PATH)
rf = pd.read_csv(RF_PATH)

# Combine top N features from ANOVA and RF importances
top_n = 20
key_features = set(anova.head(top_n)["feature"]) | set(rf.head(top_n)["feature"])
print(f"Using top {len(key_features)} discriminative features from ANOVA & RF")

# =====================================================
# LOAD EACH CLUSTER SIGNATURE FILE
# =====================================================
signatures = {}
for f in glob.glob(os.path.join(REPORT_DIR, "cluster_*_signature.csv")):
    try:
        cid = int(os.path.basename(f).split("_")[1])
        df = pd.read_csv(f)
        df = df[df["feature"].isin(key_features)]
        if not df.empty:
            df = df.sort_values("abs_effect", ascending=False).head(5)
            signatures[cid] = df
        else:
            print(f"Cluster {cid} has no matching key features")
    except Exception as e:
        print(f"Skipping {f}: {e}")

# =====================================================
# BUILD QUALITATIVE SUMMARY TABLE
# =====================================================
summary_rows = []
for cid, df in sorted(signatures.items()):
    up = df[df["effect_size"] > 0]["feature"].tolist()
    down = df[df["effect_size"] < 0]["feature"].tolist()
    summary_rows.append({
        "cluster": cid,
        "top_up_features": ", ".join(up[:3]) if up else "",
        "top_down_features": ", ".join(down[:3]) if down else "",
        "qualitative_label": ""  # You’ll fill this in manually later
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_PATH, index=False)
print(f"Cluster qualitative summary saved to: {OUTPUT_PATH}\n")

# =====================================================
# DISPLAY PREVIEW
# =====================================================
if not summary.empty:
    print(summary.head(10).to_string(index=False))
else:
    print("No clusters found or no overlapping features.")
