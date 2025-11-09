import json
import subprocess
import os
from pathlib import Path

# =====================================================
# CONFIG
# =====================================================
BASE_DIR = Path(__file__).resolve().parents[2]  # .../backend
SCRIPT_PATH = BASE_DIR / "scripts" / "character_sheet.py"
DATA_POPULATED = BASE_DIR / "app" / "data" / "populated"  # ✅ correct folder

# Use .venv Python interpreter explicitly
VENV_PYTHON = str(BASE_DIR.parent / ".venv" / "Scripts" / "python.exe")


def generate_character_sheet(player_name: str):
    """
    Executes character_sheet.py within the correct venv and project root.
    Returns the generated sheet JSON (embedded in response).
    """
    try:
        cmd = [VENV_PYTHON, str(SCRIPT_PATH), player_name]
        print(f"[DEBUG] Running sheet generation command:\n  {' '.join(map(str, cmd))}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR.parent),
            check=True
        )

        print("===== SHEET GENERATION STDOUT =====")
        print(result.stdout)
        print("===================================")

        # Locate the most recent *_sheet.json file
        files = sorted(DATA_POPULATED.glob("*_sheet.json"), key=os.path.getmtime, reverse=True)
        if not files:
            raise FileNotFoundError("No sheet was generated in populated/ directory.")
        latest_file = files[0]

        # ✅ Load the JSON contents into memory
        with open(latest_file, "r", encoding="utf-8") as f:
            sheet_data = json.load(f)

        # ✅ Return both file path and parsed JSON
        return {
            "status": "success",
            "player": player_name,
            "sheet_file": str(latest_file),
            "sheet": sheet_data,  # 👈 this is what the frontend expects
            "log": result.stdout,
        }

    except subprocess.CalledProcessError as e:
        print("===== SHEET GENERATION FAILED =====")
        print(e.stderr)
        print("===================================")
        return {
            "status": "error",
            "message": "Script execution failed",
            "stderr": e.stderr,
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
        }
