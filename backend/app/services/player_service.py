import subprocess
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
SCRIPT_PATH = BASE_DIR / "scripts" / "player_finder.py"
PLAYER_DIR = BASE_DIR / "data" / "players"

# Force absolute interpreter path
VENV_PYTHON = str(BASE_DIR.parent / ".venv" / "Scripts" / "python.exe")

def find_and_export_player(region: str, summoner_name: str, tag: str, max_matches: int = 100):
    """
    Run player_finder.py using the venv's Python (so aiohttp exists).
    """
    try:
        if not os.path.exists(VENV_PYTHON):
            raise FileNotFoundError(f"Python interpreter not found at {VENV_PYTHON}")

        cmd = [
            VENV_PYTHON,
            str(SCRIPT_PATH),
            region,
            summoner_name,
            tag,
            str(max_matches)
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=str(BASE_DIR),
        )

        output_file = BASE_DIR.parent/"data"/"players"/ f"{summoner_name}#{tag}_data.csv"
        print(output_file)
        if not output_file.exists():
            raise FileNotFoundError(f"No player data generated for {summoner_name}#{tag}")

        return {
            "status": "success",
            "player": f"{summoner_name}#{tag}",
            "region": region,
            "output_file": str(output_file),
            "message": f"Exported {summoner_name}#{tag} successfully.",
            "log": result.stdout,
        }

    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": e.stderr or str(e)}
    except Exception as e:
        return {"status": "error", "message": str(e)}
