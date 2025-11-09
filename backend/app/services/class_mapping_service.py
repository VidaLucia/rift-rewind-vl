import subprocess
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SCRIPT_PATH = PROJECT_ROOT / "backend" / "scripts" / "assign_dnd_class.py"
SUMMARY_PATH = PROJECT_ROOT / "backend" / "app" / "data" / "dnd_class_summary_weighted.csv"

def run_class_mapping(player_name: str):
    """
    Runs assign_dnd_class.py for a given player by invoking the script.
    Returns result metadata for API or logs.
    """
    try:
        python_exec = sys.executable
        result = subprocess.run(
            [python_exec, str(SCRIPT_PATH), player_name],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(PROJECT_ROOT),  # run from repo root
        )
        #print(SUMMARY_PATH)
        if not SUMMARY_PATH.exists():
            raise FileNotFoundError(f"D&D summary file not generated at {SUMMARY_PATH}")

        return {
            "status": "success",
            "player": player_name,
            "summary_file": str(SUMMARY_PATH),
            "log": result.stdout
        }

    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": e.stderr}
    except Exception as e:
        return {"status": "error", "message": str(e)}
