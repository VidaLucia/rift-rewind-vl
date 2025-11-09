import subprocess
import json
import os
import sys
from pathlib import Path

# Base project structure
BASE_DIR = Path(__file__).resolve().parents[2]
SCRIPT_PATH = BASE_DIR / "scripts" / "level_up.py"
SHEET_DIR = BASE_DIR / "data" / "populated"


def run_level_up(player_name: str):
    """
    Runs the level_up.py script for a given player.
    Returns structured results or error messages for FastAPI.
    """
    try:
        python_exec = sys.executable  # Use same Python process that FastAPI runs on
        if not os.path.exists(python_exec):
            raise FileNotFoundError(f"Python interpreter not found: {python_exec}")

        print(f"[DEBUG] Using Python interpreter: {python_exec}")
        print(f"[DEBUG] Running script: {SCRIPT_PATH}")

        # Copy current environment — all venv site-packages are already active
        env = os.environ.copy()
        env["PYTHONPATH"] = str(BASE_DIR)

        # Build subprocess command
        cmd = [python_exec, str(SCRIPT_PATH), player_name]

        # Execute the script synchronously
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=str(BASE_DIR),
            env=env
        )

        print(result.stdout)

        # Validate that the character sheet was updated
        sheet_path = SHEET_DIR / f"{player_name}_sheet.json"
        if not sheet_path.exists():
            raise FileNotFoundError(f"No sheet found for {player_name}")

        with open(sheet_path, "r", encoding="utf-8") as f:
            sheet_data = json.load(f)

        return {
            "status": "success",
            "message": f"{player_name} leveled up successfully.",
            "sheet": sheet_data,
            "log": result.stdout.strip()
        }

    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Subprocess failed with code {e.returncode}",
            "stdout": e.stdout.strip(),
            "stderr": e.stderr.strip(),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
