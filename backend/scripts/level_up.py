import json
import random
import os
from pathlib import Path

# =====================================================
# PATH CONFIG (auto-resolves absolute locations)
# =====================================================
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent        # .../backend/scripts
PROJECT_ROOT = SCRIPT_DIR.parents[1]                # .../backend
DATA_ROOT = PROJECT_ROOT / "backend" / "app" / "data"           # ✅ matches your screenshot

# Template and populated file locations
LEVEL_PATHS_FILE = DATA_ROOT / "class_level.json"
SPELL_LIST_FILE  = DATA_ROOT / "spells.json"
SHEET_DIR        = DATA_ROOT / "populated" 

# =====================================================
# JSON HELPERS
# =====================================================
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


# =====================================================
# ABILITY SCORE IMPROVEMENT LOGIC
# =====================================================
def apply_asi(sheet, level):
    stats = sheet["abilities"]
    base_stats = {k: v for k, v in stats.items() if isinstance(v, int)}

    sorted_stats = sorted(base_stats.items(), key=lambda x: x[1], reverse=True)
    highest = sorted_stats[0][0]
    second_highest = sorted_stats[1][0]

    if level == 4:
        stats[second_highest] += 1
        mod = f"+1 {second_highest}"
    elif level in [8, 12]:
        stats[highest] += 2
        mod = f"+2 {highest}"
    elif level == 16:
        stats[second_highest] += 2
        mod = f"+2 {second_highest}"
    elif level >= 19:
        stats[highest] += 2
        mod = f"+2 {highest}"
    else:
        mod = None

    if mod:
        print(f"[ASI] Level {level}: {mod}")


# =====================================================
# SPELL LEARNING LOGIC (patched for region bias)
# =====================================================
def assign_spells(sheet, level):
    cname = sheet["character_info"]["class"].lower()
    region = sheet["character_info"].get("region", "default").lower()

    all_spells = load_json(SPELL_LIST_FILE)
    class_spells = all_spells.get(cname, {})
    level_spells = class_spells.get(str(level), [])

    if not level_spells:
        print(f"[SPELLS] No spell list for {cname} level {level}")
        return

    # Match region against tags
    regional_spells = [s for s in level_spells if region in s.get("tags", [])]
    pool = regional_spells if regional_spells else level_spells

    # Choose up to 3 unique new spells
    chosen = random.sample(pool, min(3, len(pool)))
    chosen_names = [s["name"] for s in chosen]

    # Determine tier key (keep below level 9)
    tier_key = f"level_{min(level, 9)}"
    tier = sheet["spellcasting"]["spells"].get(tier_key, {})

    for spell_name in chosen_names:
        if spell_name not in tier.get("spells", []):
            tier["spells"].append(spell_name)

    print(f"[SPELLS] {cname.title()} learned ({region.title()}): {', '.join(chosen_names)}")

    # Log learning event
    sheet.setdefault("metadata", {}).setdefault("spell_log", []).append({
        "level": level,
        "region": region,
        "spells_learned": chosen_names
    })


# =====================================================
# SPELL SLOT & CASTING STATS UPDATE
# =====================================================
def update_spellcasting(sheet, level_info):
    spellcasting = sheet["spellcasting"]

    if "cantrips" in level_info:
        spellcasting["cantrips"] = spellcasting.get("cantrips", [])[:level_info["cantrips"]]

    if "spell_slots" in level_info:
        for lvl, slots in level_info["spell_slots"].items():
            lvl_key = f"level_{lvl}"
            if lvl_key in spellcasting["spells"]:
                spellcasting["spells"][lvl_key]["slots_total"] = slots
                # Auto-reset expended slots on level-up
                spellcasting["spells"][lvl_key]["slots_expended"] = 0


# =====================================================
# LEVEL-UP DRIVER
# =====================================================
def level_up_player(player_name):
    path = os.path.join(SHEET_DIR, f"{player_name}_sheet.json")
    if not os.path.exists(path):
        print(f"[ERROR] No sheet found for {player_name}")
        return

    sheet = load_json(path)
    cname = sheet["character_info"]["class"].lower()
    level_paths = load_json(LEVEL_PATHS_FILE)
    class_path = level_paths.get(cname, {})

    current_level = sheet["character_info"].get("level", 1)
    next_level = current_level + 1

    if str(next_level) not in class_path:
        print(f"[INFO] {player_name} already maxed out ({current_level})")
        return

    level_info = class_path[str(next_level)]
    features = level_info.get("features", [])
    if "subclass" in features:
        print("SUBCLASS Feature: not implemented in this version.")
    sheet["character_info"]["level"] = next_level
    print(f"[LEVEL UP] {player_name} advanced to Level {next_level} ({cname.title()})")

    # Proficiency bonus
    pb = level_info.get("proficiency_bonus")
    if pb and "combat" in sheet:
        sheet["combat"]["proficiency_bonus"] = int(pb.replace("+", ""))

    # Features
    sheet.setdefault("features_traits", []).extend(features)

    # Spellcasting classes
    if "spell_slots" in level_info:
        update_spellcasting(sheet, level_info)
        assign_spells(sheet, next_level)

    # Ability Score Improvement check
    if "Ability Score Improvement" in features:
        apply_asi(sheet, next_level)

    save_json(path, sheet)
    print(f"[SAVE] {player_name}_sheet.json updated successfully.")


# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python level_up.py <player_name>")
        sys.exit(1)
    level_up_player(sys.argv[1])
