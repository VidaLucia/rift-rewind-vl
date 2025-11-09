import subprocess
import os
import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]  # .../league-match-grabber
SCRIPT_PATH = BASE_DIR / "backend" / "scripts" / "character_sheet.py"
OUTPUT_DIR = BASE_DIR / "backend" / "app" / "data" / "populated"


def run_sheet_generation(player_name: str):
    """
    Generates a D&D character sheet JSON for the specified player.
    Runs character_sheet.py as a subprocess (inside same .venv),
    then loads and returns the JSON content directly.
    """
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        python_exec = sys.executable
        print(f"[DEBUG] Using interpreter: {python_exec}")
        print(f"[DEBUG] Running script: {SCRIPT_PATH}")

        # Run the character sheet generator script
        result = subprocess.run(
            [python_exec, str(SCRIPT_PATH), player_name],
            text=True,
            check=True,
            cwd=str(BASE_DIR),
            capture_output=True  # so we can read stdout/stderr safely
        )

        print("[DEBUG] Subprocess STDOUT:\n", result.stdout)
        print("[DEBUG] Subprocess STDERR:\n", result.stderr)

        # Construct expected output path
        sheet_path = OUTPUT_DIR / f"{player_name}_sheet.json"
        if not sheet_path.exists():
            raise FileNotFoundError(f"Character sheet not generated: {sheet_path}")

        # Load the generated JSON
        with open(sheet_path, "r", encoding="utf-8") as f:
            sheet_data = json.load(f)

        return {
            "status": "success",
            "player": player_name,
            "sheet_file": str(sheet_path),
            "sheet": sheet_data,
            "log": result.stdout.strip(),
        }

    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Subprocess failed with code {e.returncode}",
            "stderr": (e.stderr or "").strip(),
            "stdout": (e.stdout or "").strip(),
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }
