import subprocess
import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
SCRIPT_PATH = BASE_DIR / "scripts" / "progression.py"

def run_progression(player_name: str):
    try:
        python_exec = sys.executable  # <-- ensures same venv as FastAPI
        env = os.environ.copy()
        env["PYTHONPATH"] = str(BASE_DIR)  # ensure local imports work

        result = subprocess.run(
            [python_exec, str(SCRIPT_PATH), player_name],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(BASE_DIR),
            env=env
        )

        return {"status": "success", "output": result.stdout}
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "message": f"Subprocess failed ({e.returncode})",
            "stderr": e.stderr.strip()
        }
