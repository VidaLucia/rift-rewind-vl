# =====================================================
# predict_playstyle.py
# Assign playstyle clusters for new matches or players
# =====================================================
from pathlib import Path
import os
import pandas as pd
import numpy as np
import joblib
import umap
import hdbscan
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.exceptions import NotFittedError
import sys

# ================= CONFIG ===================
BASE_DIR = Path(__file__).resolve().parents[2]  
MODEL_BASE = BASE_DIR / "backend" / "models" 
ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "SUPPORT"]

DATA_DIR = BASE_DIR / "data" / "players"

# If a player name is passed as a CLI argument, use it; else fallback to default
if len(sys.argv) > 1:
    player_name = sys.argv[1]
else:
    player_name = "pyropiller167#na1"  # default for testing

INPUT_PATH = DATA_DIR / f"{player_name}_data.csv"
OUTPUT_PATH = DATA_DIR / f"{player_name}_data_clustered.csv"
DND_SUMMARY_PATH = "../../data/dnd_class_summary_weighted.csv"
CLUSTER_CLASS_PATH = "../../data/cluster_dnd_mapping.csv"

# Role mapping (Riot → model)
ROLE_MAP = {
    "UTILITY": "SUPPORT",
    "MID": "MIDDLE",
    "MIDLANE": "MIDDLE",
    "BOT": "BOTTOM",
    "ADC": "BOTTOM",
    "DUO_CARRY": "BOTTOM",
    "DUO_SUPPORT": "SUPPORT",
    "SOLO": "TOP",
    "NONE": None,
}


# =====================================================
# MODEL LOADING
# =====================================================
def load_role_models(role: str):
    print(MODEL_BASE)
    role_dir = os.path.join(MODEL_BASE, role)
    if not os.path.exists(role_dir):
        raise FileNotFoundError(f"[ERROR] No model directory found for role {role}")

    scaler = joblib.load(f"{role_dir}/scaler.pkl")
    encoder = joblib.load(f"{role_dir}/encoder.pkl")
    reducer = joblib.load(f"{role_dir}/umap_reducer.pkl")
    clusterer = joblib.load(f"{role_dir}/hdbscan_model.pkl")
    gmm = joblib.load(f"{role_dir}/gmm_model.pkl")
    features = joblib.load(f"{role_dir}/features.pkl")

    try:
        _ = scaler.scale_
    except AttributeError:
        raise NotFittedError(f"[ERROR] The scaler for {role} is not fitted.")

    return scaler, encoder, reducer, clusterer, gmm, features


# =====================================================
# PREPROCESSING + PREDICTION
# =====================================================
def preprocess_input(df: pd.DataFrame, features, scaler, encoder):
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "role_str" in df.columns and "role" not in df.columns:
        df["role"] = df["role_str"]

    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    pos_cols = [c for c in num_cols if (df[c] >= 0).all()]
    df[pos_cols] = np.log1p(df[pos_cols])

    if hasattr(scaler, "feature_names_in_"):
        trained_cols = list(scaler.feature_names_in_)
        for col in trained_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[[c for c in df.columns if c in trained_cols or c in encoder.feature_names_in_]]
        num_cols = [c for c in trained_cols if c in df.columns]

    trained_cat_cols = []
    if hasattr(encoder, "feature_names_in_"):
        trained_cat_cols = list(encoder.feature_names_in_)
    cat_cols = [c for c in trained_cat_cols if c in df.columns]

    X_num = scaler.transform(df[num_cols]) if num_cols else np.empty((len(df), 0))
    X_cat = encoder.transform(df[cat_cols]) if cat_cols else np.empty((len(df), 0))
    return np.concatenate([X_num, X_cat], axis=1)


def predict_role_clusters(df: pd.DataFrame):
    results = []
    df["role_str"] = df["role_str"].str.upper().replace(ROLE_MAP)
    df = df[df["role_str"].notna() & (df["role_str"] != "NONE")]

    print("Normalized roles:", df["role_str"].unique())

    for role in ROLES:
        role_df = df[df["role_str"] == role]
        if role_df.empty:
            print(f"[WARN] No data for role {role}, skipping.")
            continue
        try:
            scaler, encoder, reducer, clusterer, gmm, features = load_role_models(role)
        except (FileNotFoundError, NotFittedError) as e:
            print(str(e))
            continue

        X_processed = preprocess_input(role_df, features, scaler, encoder)
        embedding = reducer.transform(X_processed)

        hdb_labels, hdb_strengths = hdbscan.approximate_predict(clusterer, embedding)
        gmm_labels = gmm.predict(embedding)
        gmm_probs = gmm.predict_proba(embedding)

        role_df["cluster"] = hdb_labels
        role_df["hdbscan_confidence"] = hdb_strengths
        role_df["gmm_cluster"] = gmm_labels
        role_df["gmm_confidence"] = gmm_probs.max(axis=1)
        results.append(role_df)

    combined = pd.concat(results, ignore_index=True)
    print("\n================= CLUSTER DISTRIBUTIONS =================")
    for role in combined["role_str"].unique():
        sub = combined[combined["role_str"] == role]
        counts = sub["cluster"].value_counts().sort_index()
        total = counts.sum()
        print(f"\n[{role}]")
        for cid, cnt in counts.items():
            pct = (cnt / total) * 100
            print(f"  Cluster {cid:>2}: {cnt:>5} ({pct:5.1f}%)")
    print("==========================================================\n")
    return combined


# =====================================================
# POSTPROCESSING
# =====================================================
def compute_top_champions(df: pd.DataFrame, player_name: str):
    if "championName" in df.columns and "champion" not in df.columns:
        df["champion"] = df["championName"]

    if "champion" not in df.columns:
        print("[WARN] No 'champion' column found — skipping champion summary.")
        return None

    counts = df["champion"].value_counts(normalize=True) * 100
    champs = counts.index.tolist()
    vals = counts.values.tolist()
    return {
        "player_name": player_name,
        "Top1_Champion": champs[0] if len(champs) > 0 else None,
        "Top1_Percent": round(vals[0], 2) if len(vals) > 0 else 0,
        "Top2_Champion": champs[1] if len(champs) > 1 else None,
        "Top2_Percent": round(vals[1], 2) if len(vals) > 1 else 0,
        "Top3_Champion": champs[2] if len(champs) > 2 else None,
        "Top3_Percent": round(vals[2], 2) if len(vals) > 2 else 0,
    }


def compute_dnd_class(df: pd.DataFrame, player_name: str, mapping_path: str):
    """
    Determine player's dominant D&D class based on their clusters.
    Uses the player-specific cluster→class mapping if available.
    """

    # Prefer player-specific mapping file if it exists
    player_map_path = os.path.join(DATA_DIR, f"{player_name}_cluster_dnd_mapping.csv")
    map_to_use = player_map_path if os.path.exists(player_map_path) else mapping_path
    print(f"[DEBUG] Using D&D mapping file: {map_to_use}")
    if not os.path.exists(map_to_use):
        print(f"[WARN] Missing D&D mapping at {map_to_use}")
        return {"player_name": player_name, "DND_Class": None, "Weighted_Share": 0.0}

    dnd_map = pd.read_csv(map_to_use)

    # Ensure clusters are comparable
    if "cluster" not in dnd_map.columns:
        print(f"[ERROR] Mapping file {map_to_use} missing 'cluster' column.")
        return {"player_name": player_name, "DND_Class": None, "Weighted_Share": 0.0}

    df["cluster"] = df["cluster"].astype(int, errors="ignore")
    dnd_map["cluster"] = dnd_map["cluster"].astype(int, errors="ignore")

    # Compute weighted distribution
    cluster_counts = df["cluster"].value_counts(normalize=True) * 100
    cluster_df = cluster_counts.reset_index()
    cluster_df.columns = ["cluster", "Weighted_Share"]

    merged = cluster_df.merge(dnd_map[["cluster", "DND_Class"]], on="cluster", how="left")

    # Sanity check
    if merged["DND_Class"].isna().all():
        print(f"[ERROR] No cluster matches found in {map_to_use}.")
        print(merged)
        return {"player_name": player_name, "DND_Class": None, "Weighted_Share": 0.0}

    merged = merged.dropna(subset=["DND_Class"])
    top_row = merged.sort_values("Weighted_Share", ascending=False).iloc[0]

    print(f"[INFO] Top D&D class for {player_name}: {top_row['DND_Class']} ({top_row['Weighted_Share']:.1f}%)")

    return {
        "player_name": player_name,
        "DND_Class": str(top_row["DND_Class"]),
        "Weighted_Share": round(top_row["Weighted_Share"], 2),
    }


def run_predict_playstyle(player_name: str):
    input_path = DATA_DIR / f"{player_name}_data.csv"
    output_path = DATA_DIR / f"{player_name}_data_clustered.csv"
    print(f"[1/4] Loading new data for {player_name}...")
    df_new = pd.read_csv(input_path)

    print("[2/4] Predicting clusters...")
    clustered_df = predict_role_clusters(df_new)

    print("[3/4] Computing summaries...")
    champ_summary = compute_top_champions(clustered_df, player_name)
    dnd_summary = compute_dnd_class(clustered_df, player_name, CLUSTER_CLASS_PATH)

    print("[4/4] Saving outputs...")
    clustered_df.to_csv(output_path, index=False)
    print(f"[INFO] Saved detailed match clusters to {output_path}")

    # Append summaries to the clustered file (for readability)
    summary_df = pd.DataFrame([{**(champ_summary or {}), **(dnd_summary or {})}])
    summary_df["__summary__"] = True
    with open(output_path, "a", encoding="utf-8", newline="") as f:
        summary_df.to_csv(f, index=False)

     # ==============================
    # Save player-level summary (FORCE UPDATE)
    # ==============================
    APP_DATA_DIR = BASE_DIR / "backend" / "app" / "data"
    PLAYER_SUMMARY_PATH = APP_DATA_DIR / "player_class_summary.csv"
    os.makedirs(APP_DATA_DIR, exist_ok=True)

    if champ_summary and dnd_summary:
        # Merge champion + DND info
        combined_summary = pd.DataFrame([{**champ_summary, **dnd_summary}])

        # Normalize the player name for consistency
        player_name_norm = player_name.lower().strip()
        combined_summary["player_name"] = player_name_norm
        combined_summary["DND_Class"] = dnd_summary["DND_Class"]
        combined_summary["Weighted_Share"] = dnd_summary["Weighted_Share"]

        if PLAYER_SUMMARY_PATH.exists():
            player_file = pd.read_csv(PLAYER_SUMMARY_PATH)

            # Normalize old file names and force overwrite
            player_file["player_name"] = player_file["player_name"].str.lower().str.strip()

            # Drop existing row for this player
            player_file = player_file[player_file["player_name"] != player_name_norm]

            # Concatenate the new row
            merged = pd.concat([player_file, combined_summary], ignore_index=True)
        else:
            merged = combined_summary

        # Force column order for consistency
        cols = [
            "player_name", "Top1_Champion", "Top1_Percent",
            "Top2_Champion", "Top2_Percent", "Top3_Champion",
            "Top3_Percent", "DND_Class", "Weighted_Share"
        ]
        for c in cols:
            if c not in merged.columns:
                merged[c] = None
        merged = merged[cols]

        # Write to disk
        
        merged.to_csv(PLAYER_SUMMARY_PATH, index=False)
        print(f"[INFO] FORCE-UPDATED {PLAYER_SUMMARY_PATH} for player '{player_name}' → {dnd_summary['DND_Class']}")

    else:
        print("[WARN] Missing champ_summary or dnd_summary — skipping summary write.")

    print("All outputs saved successfully.")
    return {
        "clustered_path": str(output_path),
        "summary_path": str(PLAYER_SUMMARY_PATH),
        "champ_summary": champ_summary,
        "dnd_summary": dnd_summary,
    }
