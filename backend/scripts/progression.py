import os
import json
import random
import pandas as pd
from datetime import datetime
from level_up import level_up_player  
import sys, os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
# =====================================================
# PATH CONFIG (auto-resolves absolute locations)
# =====================================================
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent        # .../backend/scripts
PROJECT_ROOT = SCRIPT_DIR.parent                    # .../backend
DATA_ROOT = PROJECT_ROOT / "app" / "data"           # correct absolute path

# Data subdirectories
DATA_DIR  = DATA_ROOT / "players"                   # .../app/data/players
SHEET_DIR = DATA_ROOT / "populated"                  # .../app/data/populated

# Ensure populated folder exists
SHEET_DIR.mkdir(parents=True, exist_ok=True)

ITEM_POOL = {
    "common": ["Health Potion", "Ration Pack", "Rusted Trinket"],
    "rare": ["Mana Elixir", "Traveler’s Cloak", "Ring of Focus"],
    "epic": ["Blessed Mace", "Shadow Cloak", "Arcane Tome", "Divine Shield"],
    "legendary": ["Heart of Demacia", "Crown of Ionia", "Eclipse Blade"]
}

CLASS_ITEMS = {
    "cleric": ["Holy Symbol", "Healing Robe", "Blessed Mace"],
    "wizard": ["Arcane Tome", "Crystal Wand"],
    "fighter": ["Steel Sword", "Shield of Valor"],
    "rogue": ["Dagger", "Cloak of Shadows"],
    "druid": ["Totem Staff", "Nature Pendant"],
    "ranger": ["Hunting Bow", "Feathered Mantle"],
    "paladin": ["Vow Token", "Sacred Shield"],
    "warlock": ["Cursed Orb", "Eldritch Focus"],
    "bard": ["Lute of Harmony", "Silver Quill"],
    "sorcerer": ["Spell Crystal", "Flameheart Orb"],
    "barbarian": ["War Drum", "Rage Charm"],
    "artificer": ["Runic Gauntlet", "Tinker’s Toolkit"],
}

# =====================================================
# HELPERS
# =====================================================
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def compute_performance_multiplier(df):
    """Compute a performance-based multiplier using KDA and gold/min from match data."""
    for col in ["kills", "assists", "deaths", "ch_goldPerMinute"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0

    avg_kda = ((df["kills"] + df["assists"]) / df["deaths"].clip(lower=1)).mean()
    avg_gold = df["ch_goldPerMinute"].mean()

    mult = 1.0
    if avg_kda > 3:
        mult += 0.15
    elif avg_kda > 2:
        mult += 0.10
    elif avg_kda > 1:
        mult += 0.05

    if avg_gold > 400:
        mult += 0.10
    elif avg_gold > 350:
        mult += 0.05

    print(f"[PERF] Avg KDA: {avg_kda:.2f}, Avg GPM: {avg_gold:.1f}, Multiplier ×{mult:.2f}")
    return round(mult, 2)

# =====================================================
# MAIN PROGRESSION LOOP
# =====================================================
def process_player_progress(player_name):
    sheet_path = os.path.join(SHEET_DIR, f"{player_name}_sheet.json")
    match_path = os.path.join(DATA_DIR, f"{player_name}_data_clustered.csv")

    if not os.path.exists(sheet_path):
        print(f"[ERROR] No character sheet for {player_name}")
        return
    if not os.path.exists(match_path):
        print(f"[ERROR] No match data for {player_name}")
        return

    sheet = load_json(sheet_path)
    df = pd.read_csv(match_path)

    # Ensure progression section exists
    meta = sheet.setdefault("progression", {
        "games_played": 0,
        "pity_counter": 0,
        "experience_points": 0,
        "level": 1
    })

    # Determine new games
    games_already = meta.get("games_played", 0)
    new_df = df.iloc[games_already:]
    new_games = len(new_df)
    if new_games <= 0:
        print(f"[INFO] No new matches for {player_name}")
        return

    # Compute performance bonus
    multiplier = compute_performance_multiplier(new_df)
    print(f"[PERF] {player_name}: {new_games} new games | Multiplier ×{multiplier}")

    # Apply EXP gain
    exp_gain = int(new_games * multiplier)
    meta["experience_points"] += exp_gain
    meta["games_played"] += new_games
    meta["pity_counter"] += new_games

    leveled_up = False
    while meta["experience_points"] >= 50:
        meta["experience_points"] -= 50
        meta["level"] += 1
        leveled_up = True
        grant_items(sheet, guaranteed=True, multiplier=multiplier)
        if level_up_player:
            print(f"[LEVEL SYSTEM] Level up detected — invoking level_up.py for {player_name}...")
            level_up_player(player_name)
        else:
            print(f"[LEVEL SYSTEM] Skipped — could not import level_up_player.")


    if not leveled_up:
        if roll_pity_item(meta["pity_counter"], multiplier):
            grant_items(sheet, guaranteed=False, multiplier=multiplier)
            meta["pity_counter"] = 0
    char_info = sheet.get("character_info", {})
    char_info["level"] = meta["level"]
    char_info["experience_points"] = meta["experience_points"]
    sheet["character_info"] = char_info
    save_json(sheet_path, sheet)
    print(f"[UPDATE] {player_name}: Level {meta['level']} | EXP {meta['experience_points']}/50")


# =====================================================
# ITEM + LEVEL UP HELPERS
# =====================================================
def roll_pity_item(pity_counter, multiplier):
    """Check if pity counter triggers a random item drop."""
    chance = 0.02 * (pity_counter // 10) * multiplier
    roll = random.random()
    print(f"[PITY] Counter={pity_counter}, Roll={roll:.2f}, Chance={chance:.2f}")
    return roll < chance
def grant_items(sheet, guaranteed, multiplier):
    """Distribute new items into the appropriate sheet sections instead of metadata only."""
    cname = sheet["character_info"]["class"].lower()
    rarity_roll = random.random() * multiplier

    if rarity_roll > 1.2:
        rarity = "legendary"
    elif rarity_roll > 0.9:
        rarity = "epic"
    elif rarity_roll > 0.6:
        rarity = "rare"
    else:
        rarity = "common"

    base_pool = ITEM_POOL[rarity]
    num_items = random.randint(2, 4) if guaranteed else 1

    # Determine class-specific item
    class_item = random.choice(CLASS_ITEMS.get(cname, ["Adventurer’s Token"]))
    extra_items = random.sample(base_pool, min(num_items - 1, len(base_pool)))
    new_items = [class_item] + extra_items if guaranteed else extra_items

    # Ensure core sections exist
    eq = sheet.setdefault("equipment", {})
    eq.setdefault("armor", [])
    eq.setdefault("weapons", [])
    eq.setdefault("gear", [])
    eq.setdefault("other_proficiencies_languages", [])

    backstory = sheet.setdefault("backstory", {})
    backstory.setdefault("treasure", [])

    # Categorize and insert items properly
    added_items = []
    for item in new_items:
        # Simple heuristic-based categorization
        item_lower = item.lower()
        if any(k in item_lower for k in ["sword", "mace", "bow", "dagger", "wand", "staff", "weapon", "blade"]):
            target_list = eq["weapons"]
        elif any(k in item_lower for k in ["armor", "robe", "shield", "cloak", "mantle"]):
            target_list = eq["armor"]
        elif rarity in ["epic", "legendary"]:
            target_list = backstory["treasure"]
        else:
            target_list = eq["gear"]

        # Add only if not duplicate
        if item not in target_list:
            target_list.append(item)
            added_items.append(item)

    # Log it in metadata for tracking history (optional)
    sheet.setdefault("metadata", {}).setdefault("item_log", []).append({
        "timestamp": datetime.now().isoformat(),
        "items": added_items,
        "rarity": rarity,
        "multiplier": multiplier,
        "type": "guaranteed" if guaranteed else "pity"
    })

    if added_items:
        print(f"[ITEMS] {'Level-up' if guaranteed else 'Pity'} ({rarity}): {added_items}")
    else:
        print(f"[ITEMS] No new unique items added (duplicates filtered out).")

# =====================================================
# ENTRY POINT
# =====================================================
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python progression.py <player_name>")
        sys.exit(1)

    player_name = sys.argv[1]
    print(f"Processing progression for {player_name}...")
    process_player_progress(player_name)