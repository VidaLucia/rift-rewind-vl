import os
import sys
import pandas as pd
from pathlib import Path

# =====================================================
# PATH FIX — ensures imports always work
# =====================================================
CURRENT_FILE = Path(__file__).resolve()
BACKEND_DIR = CURRENT_FILE.parents[2]   # .../backend
ROOT_DIR = CURRENT_FILE.parents[3]      # .../league-match-grabber
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

print("[DEBUG] Root dir added to sys.path:", ROOT_DIR)

from backend.scripts import predict_playstyle


# =====================================================
# CONFIG
# =====================================================
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data" / "players"))
CLUSTER_CLASS_PATH = Path(os.getenv("CLUSTER_CLASS_PATH", ROOT_DIR / "data" / "cluster_dnd_mapping.csv"))


# =====================================================
# MAIN FUNCTION
# =====================================================
def predict_player(player_name: str):
    """
    Run clustering + D&D class inference for a player's match data.
    Saves:
      - backend/app/data/player_class_summary.csv
    Returns summary dict for API use.
    """

    input_path = DATA_DIR / f"{player_name}_data.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Player data not found at: {input_path}")

    print(f"[DEBUG] Loading match data from {input_path}")
    df = pd.read_csv(input_path)

    # --- Run clustering
    clustered_df = predict_playstyle.predict_role_clusters(df)

    # --- Compute summaries
    champ_summary = predict_playstyle.compute_top_champions(clustered_df, player_name)
    dnd_summary = predict_playstyle.compute_dnd_class(
        clustered_df, player_name, str(CLUSTER_CLASS_PATH)
    )

    # --- Save player_class_summary.csv
    APP_DATA_DIR = ROOT_DIR / "backend" / "app" / "data"
    PLAYER_SUMMARY_PATH = APP_DATA_DIR / "player_class_summary.csv"
    os.makedirs(APP_DATA_DIR, exist_ok=True)

    if champ_summary:
        combined_summary = pd.DataFrame([{**champ_summary, **(dnd_summary or {})}])
        if PLAYER_SUMMARY_PATH.exists():
            prev = pd.read_csv(PLAYER_SUMMARY_PATH)
            prev = prev[prev["player_name"] != player_name]
            combined = pd.concat([prev, combined_summary], ignore_index=True)
        else:
            combined = combined_summary
        combined.to_csv(PLAYER_SUMMARY_PATH, index=False)
        print(f"[INFO] Updated {PLAYER_SUMMARY_PATH} with champion + class info.")
    else:
        print("[WARN] champ_summary is empty; skipping player_class_summary save.")

    # --- Combine output for return
    output = {
        "player_name": player_name,
        "champion_summary": champ_summary,
        "dnd_summary": dnd_summary,
        "clusters_detected": clustered_df["cluster"].nunique(),
        "roles_processed": clustered_df["role_str"].unique().tolist(),
    }

    print(f"[DEBUG] Prediction completed for {player_name}")
    return output
