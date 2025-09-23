import json, os

PROGRESS_FILE = "progress.json"

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_progress(update):
    # Load existing progress
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = {}
    else:
        existing = {}

    # Merge update into existing
    for region, tiers in update.items():
        if region not in existing:
            existing[region] = {}
        for tier, divs in tiers.items():
            if tier not in existing[region]:
                existing[region][tier] = []
            for d in divs:
                if d not in existing[region][tier]:
                    existing[region][tier].append(d)

    with open(PROGRESS_FILE, "w") as f:
        json.dump(existing, f, indent=2)

