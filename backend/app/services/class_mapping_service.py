import subprocess
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "backend" / "scripts" / "assign_dnd_class.py"
SUMMARY_PATH = PROJECT_ROOT / "backend" / "app" / "data" / "dnd_class_summary_weighted.csv"

def run_class_mapping(player_name: str):
    """
    Runs assign_dnd_class.py for a given player in a subprocess.
    Always overwrites the weighted D&D summary file.
    """
    try:
        python_exec = sys.executable
        print(f"[DEBUG] Using interpreter: {python_exec}")
        print(f"[DEBUG] Running script: {SCRIPT_PATH}")

        result = subprocess.run(
            [python_exec, str(SCRIPT_PATH), player_name],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(PROJECT_ROOT / "backend"),  # ensures correct path resolution
        )

        # Wait briefly for file flush
        for _ in range(5):
            if SUMMARY_PATH.exists():
                break
            time.sleep(0.2)

        if not SUMMARY_PATH.exists():
            raise FileNotFoundError(f"D&D summary file not generated at {SUMMARY_PATH}")

        print(f"[INFO] Updated D&D summary: {SUMMARY_PATH}")
        return {
            "status": "success",
            "player": player_name,
            "summary_file": str(SUMMARY_PATH),
            "log": result.stdout.strip()
        }

    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Subprocess failed ({e.returncode}): {e.stderr.strip()}"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
