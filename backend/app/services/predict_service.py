import sys
import os
from pathlib import Path

# =====================================================
# PATH FIX — ensures imports always work regardless of run context
# =====================================================
CURRENT_FILE = Path(__file__).resolve()
BACKEND_DIR = CURRENT_FILE.parents[2]   # .../backend
ROOT_DIR = CURRENT_FILE.parents[3]      # .../league-match-grabber

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

print("[DEBUG] Root dir added to sys.path:", ROOT_DIR)

# Import the new unified function
from backend.scripts.predict_playstyle import run_predict_playstyle


# =====================================================
# MAIN FUNCTION
# =====================================================
def predict_player(player_name: str):
    """
    Run the unified prediction pipeline:
    1. Predict playstyle clusters
    2. Compute inline D&D class mapping
    3. Save all summaries (champion + D&D)
    Returns a combined dictionary for API use.
    """
    try:
        print(f"[INFO] Starting full prediction pipeline for {player_name}")
        result = run_predict_playstyle(player_name)
        print(f"[INFO] Completed prediction pipeline for {player_name}")
        return result

    except FileNotFoundError as e:
        print(f"[ERROR] Missing player data for {player_name}: {e}")
        raise
    except Exception as e:
        print(f"[ERROR] Prediction pipeline failed for {player_name}: {e}")
        raise
