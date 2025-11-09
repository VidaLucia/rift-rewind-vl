import json
import math
import os
import random
import pandas as pd
from datetime import datetime

# Absolute path of this script (backend/scripts/character_sheet.py)
SCRIPT_DIR = os.path.dirname(__file__)

# Go two levels up to reach the project root (league-match-grabber/)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))

# --- Correct data paths ---
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
BACKEND_DATA_DIR = os.path.join(PROJECT_ROOT, "backend", "app", "data")

TEMPLATE_PATH = os.path.join(DATA_DIR, "template", "dnd_sheet.json")
CHAMPION_REGION_PATH = os.path.join(DATA_DIR, "template", "champion_region.json")
#CLASS_SUMMARY_PATH = os.path.join(DATA_DIR, "template", "dnd_class_summary_weighted.csv")

PLAYER_SUMMARY_PATH = os.path.join(PROJECT_ROOT, "backend", "app", "data", "player_class_summary.csv")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "backend", "app", "data", "populated")

CANTRIP_POOL = {
    "Acid Splash": {"wizard": 3, "sorcerer": 2},
    "Blade Ward": {"wizard": 2, "sorcerer": 1, "warlock": 1},
    "Bone Chill": {"wizard": 3, "warlock": 2},
    "Dancing Lights": {"bard": 2, "wizard": 2, "sorcerer": 1},
    "Eldritch Blast": {"warlock": 10},  # almost guaranteed for warlocks
    "Fire Bolt": {"wizard": 4, "sorcerer": 3},
    "Friends": {"bard": 3, "sorcerer": 1, "warlock": 1},
    "Guidance": {"cleric": 3, "druid": 2},
    "Light": {"cleric": 2, "wizard": 1},
    "Mage Hand": {"wizard": 2, "sorcerer": 2, "bard": 1},
    "Minor Illusion": {"wizard": 3, "bard": 2},
    "Poison Spray": {"druid": 2, "sorcerer": 2, "warlock": 1},
    "Produce Flame": {"druid": 4},
    "Ray of Frost": {"wizard": 3, "sorcerer": 2},
    "Resistance": {"cleric": 3, "druid": 2},
    "Sacred Flame": {"cleric": 5},
    "Selune's Dream": {"cleric": 2, "paladin": 1},
    "Shillelagh": {"druid": 5},
    "Shocking Grasp": {"wizard": 2, "sorcerer": 3},
    "Thaumaturgy": {"cleric": 3, "paladin": 1},
    "Thorn Whip": {"druid": 3},
    "True Strike": {"wizard": 2, "sorcerer": 2},
    "Vicious Mockery": {"bard": 5},
    "Booming Blade": {"wizard": 3, "sorcerer": 2, "artificer": 2},
    "Toll the Dead": {"cleric": 3, "warlock": 2},
    "Bursting Sinew": {"barbarian": 2, "fighter": 1},
}

LEVEL1_SPELL_POOL = {
    "Animal Friendship": {"druid": 3, "ranger": 2, "bard": 1},
    "Armour of Agathys": {"warlock": 5},
    "Arms of Hadar": {"warlock": 4},
    "Bane": {"cleric": 3, "bard": 2},
    "Bless": {"cleric": 5, "paladin": 2},
    "Burning Hands": {"sorcerer": 4, "wizard": 4},
    "Charm Person": {"bard": 4, "sorcerer": 3, "warlock": 2},
    "Chromatic Orb": {"sorcerer": 3, "wizard": 3},
    "Colour Spray": {"sorcerer": 2, "wizard": 2},
    "Command (Halt)": {"cleric": 3, "paladin": 2},
    "Compelled Duel": {"paladin": 3},
    "Create or Destroy Water": {"cleric": 2, "druid": 3},
    "Cure Wounds": {"cleric": 4, "bard": 3, "druid": 3, "paladin": 2, "ranger": 1},
    "Disguise Self": {"bard": 3, "sorcerer": 2, "wizard": 3},
    "Dissonant Whispers": {"bard": 4},
    "Divine Favour": {"paladin": 3},
    "Ensnaring Strike": {"ranger": 3},
    "Entangle": {"druid": 4},
    "Expeditious Retreat": {"sorcerer": 2, "wizard": 2, "artificer": 2},
    "Faerie Fire": {"bard": 2, "druid": 3},
    "False Life": {"sorcerer": 3, "wizard": 3, "artificer": 2},
    "Feather Fall": {"sorcerer": 3, "wizard": 3},
    "Find Familiar": {"wizard": 5},
    "Fog Cloud": {"ranger": 2, "druid": 2, "sorcerer": 1},
    "Goodberry": {"druid": 3, "ranger": 2},
    "Grease": {"wizard": 2, "artificer": 2},
    "Guiding Bolt": {"cleric": 5},
    "Hail of Thorns": {"ranger": 4},
    "Healing Word": {"cleric": 5, "bard": 4, "druid": 4},
    "Hellish Rebuke": {"warlock": 5},
    "Heroism": {"paladin": 3, "bard": 2},
    "Hex": {"warlock": 5},
    "Hunter's Mark": {"ranger": 5},
    "Ice Knife": {"druid": 2, "sorcerer": 2, "wizard": 2},
    "Inflict Wounds": {"cleric": 5},
    "Enhance Leap": {"druid": 2, "ranger": 2},
    "Longstrider": {"druid": 3, "ranger": 3},
    "Mage Armour": {"wizard": 4, "sorcerer": 2, "artificer": 2},
    "Magic Missile": {"wizard": 5, "sorcerer": 4},
    "Protection from Evil and Good": {"cleric": 3, "paladin": 3, "wizard": 2},
    "Ray of Sickness": {"sorcerer": 2, "wizard": 3},
    "Sanctuary": {"cleric": 4},
    "Searing Smite": {"paladin": 3},
    "Shield": {"wizard": 5, "sorcerer": 3},
    "Shield of Faith": {"cleric": 3, "paladin": 3},
    "Sleep": {"sorcerer": 4, "wizard": 4, "bard": 3},
    "Speak with Animals": {"druid": 3, "ranger": 3},
    "Tasha's Hideous Laughter": {"bard": 4, "wizard": 2},
    "Thunderous Smite": {"paladin": 4},
    "Thunderwave": {"wizard": 3, "sorcerer": 3, "bard": 2, "druid": 2},
    "Wrathful Smite": {"paladin": 4},
    "Witch Bolt": {"sorcerer": 3, "warlock": 3, "wizard": 2}
}
LEVEL1_SPELL_COUNTS = {
    "artificer": 2,
    "bard": 4,
    "cleric": 4,
    "druid": 4,
    "paladin": 2,
    "ranger": 2,
    "sorcerer": 4,
    "warlock": 2,
    "wizard": 6
}
LEVEL1_SPELL_SLOTS = {
    "artificer": 2,
    "bard": 2,
    "cleric": 2,
    "druid": 2,
    "paladin": 2,
    "ranger": 2,
    "sorcerer": 2,
    "warlock": 1,
    "wizard": 2
}
# =====================================================
# LOADERS
# =====================================================
def load_template() -> dict:
    with open(TEMPLATE_PATH, "r") as f:
        return json.load(f)

def load_champion_regions() -> dict:
    with open(CHAMPION_REGION_PATH, "r") as f:
        return json.load(f)


# =====================================================
# HELPER FUNCTIONS
# =====================================================
def find_region_for_champion(champion_name: str, region_map: dict) -> str:
    champ = champion_name.lower().strip()
    for region, champs in region_map.items():
        if champ in champs:
            return region
    return "unknown"

def determine_alignment(champion_variety_score: float) -> str:
    moral_axis = random.choice(["Good", "Neutral", "Evil"])
    if champion_variety_score < 0.33:
        lawful_axis = "Lawful"
    elif champion_variety_score < 0.66:
        lawful_axis = "Neutral"
    else:
        lawful_axis = "Chaotic"
    return f"{lawful_axis} {moral_axis}"

def determine_background(class_name: str, region: str) -> str:
    backgrounds = {
        "bandlecity": ["Entertainer", "Folk Hero", "Urchin"],
        "demacia": ["Noble", "Soldier", "Acolyte"],
        "freljord": ["Outlander", "Folk Hero", "Soldier"],
        "ionia": ["Sage", "Monk", "Entertainer"],
        "ixtal": ["Sage", "Outlander", "Acolyte"],
        "noxus": ["Soldier", "Urchin", "Criminal"],
        "piltover": ["Guild Artisan", "Sage", "Charlatan"],
        "shadow_isles": ["Hermit", "Charlatan", "Outlander"],
        "bilgewater": ["Sailor", "Criminal", "Entertainer"],
        "shurima": ["Outlander", "Acolyte", "Sage"],
        "targon": ["Acolyte", "Sage", "Folk Hero"],
        "void": ["Hermit", "Outlander", "Sage"],
        "zaun": ["Criminal", "Urchin", "Artisan"],
    }
    return random.choice(backgrounds.get(region, ["Traveler"]))
def assign_skills_by_class_and_background(class_name: str, background: str) -> dict:
    """Assigns exactly two proficient skills based on class and background."""

    # All possible skills default to False
    skills = {
        "acrobatics": False, "animal_handling": False, "arcana": False, "athletics": False,
        "deception": False, "history": False, "insight": False, "intimidation": False,
        "investigation": False, "medicine": False, "nature": False, "perception": False,
        "performance": False, "persuasion": False, "religion": False, "sleight_of_hand": False,
        "stealth": False, "survival": False
    }

    class_skills = {
        "artificer": ["arcana", "investigation", "history"],
        "barbarian": ["athletics", "survival", "intimidation"],
        "bard": ["performance", "persuasion", "deception"],
        "cleric": ["insight", "religion", "medicine"],
        "druid": ["nature", "animal_handling", "survival"],
        "fighter": ["athletics", "perception", "intimidation"],
        "monk": ["acrobatics", "stealth", "insight"],
        "paladin": ["religion", "persuasion", "athletics"],
        "ranger": ["stealth", "nature", "survival"],
        "rogue": ["stealth", "sleight_of_hand", "deception"],
        "sorcerer": ["arcana", "persuasion", "deception"],
        "warlock": ["arcana", "intimidation", "deception"],
        "wizard": ["arcana", "history", "investigation"]
    }

    background_skills = {
        "noble": ["history", "persuasion"],
        "soldier": ["athletics", "intimidation"],
        "urchin": ["stealth", "sleight_of_hand"],
        "outlander": ["survival", "athletics"],
        "sage": ["arcana", "history"],
        "charlatan": ["deception", "sleight_of_hand"],
        "acolyte": ["religion", "insight"],
        "folk hero": ["animal_handling", "survival"],
        "entertainer": ["performance", "acrobatics"],
        "criminal": ["stealth", "deception"],
        "guild artisan": ["insight", "persuasion"],
        "hermit": ["medicine", "religion"],
        "sailor": ["athletics", "perception"],
        "traveler": ["survival", "perception"],
        "unknown": ["perception", "insight"]
    }

    cname = class_name.lower()
    bname = background.lower()

    class_opts = class_skills.get(cname, [])
    bg_opts = background_skills.get(bname, [])

    # Combine and randomize
    combined = list(set(class_opts + bg_opts))
    if len(combined) <= 2:
        chosen = combined
    else:
        chosen = random.sample(combined, 2)

    for skill in chosen:
        skills[skill] = True

    return skills
def determine_race(region: str) -> str:
    region_races = {
        "bandlecity": ["Halfling", "Gnome"],
        "demacia": ["Human", "Half-Elf"],
        "freljord": ["Human", "Goliath", "Half-Orc"],
        "ionia": ["Elf", "Human"],
        "ixtal": ["Elf", "Human", "Genasi"],
        "noxus": ["Human", "Tiefling"],
        "piltover": ["Gnome", "Human"],
        "shadow_isles": ["Revenant", "Human"],
        "bilgewater": ["Human", "Half-Elf", "Halfling"],
        "shurima": ["Human", "Genasi"],
        "targon": ["Aasimar", "Human"],
        "void": ["Aberration", "Tiefling"],
        "zaun": ["Human", "Gnome", "Half-Orc"],
    }
    return random.choice(region_races.get(region, ["Human"]))
def choose_spells_for_class(class_name: str, n: int) -> list:
    """Randomly choose n Level 1 spells weighted by class affinity."""
    cname = class_name.lower()
    weighted = []
    for spell, weights in LEVEL1_SPELL_POOL.items():
        weight = weights.get(cname, 0)
        if weight > 0:
            weighted.extend([spell] * weight)

    if not weighted:
        return []

    selected = random.sample(weighted, min(n, len(set(weighted))))
    return list(dict.fromkeys(selected))
def assign_speed_by_race(race: str) -> int:
    race = race.lower()
    speed_table = {
        "human": 30, "half-elf": 30, "elf": 35, "halfling": 25, "gnome": 25,
        "goliath": 30, "half-orc": 30, "aasimar": 30, "tiefling": 30,
        "revenant": 25, "aberration": 30, "genasi": 30,
    }
    return speed_table.get(race, 30)

def ability_modifier(score: int) -> int:
    return math.floor((score - 10) / 2)


# =====================================================
# CHARACTER SHEET POPULATION
# =====================================================
def populate_common_fields(sheet: dict, class_name: str, player_name: str, region: str, champ_variety_score: float):
    sheet["character_info"]["character_name"] = f"{player_name}'s {class_name}"
    sheet["character_info"]["class"] = f"{class_name}"
    sheet["character_info"]["player_name"] = player_name
    sheet["character_info"]["alignment"] = determine_alignment(champ_variety_score)
    sheet["character_info"]["background"] = determine_background(class_name, region)
    sheet["character_info"]["race"] = determine_race(region)
    sheet["combat"]["speed"] = assign_speed_by_race(sheet["character_info"]["race"])
    sheet["equipment"]["other_proficiencies_languages"] = [region]
    sheet["appearance"]["description"] = f"A {sheet['character_info']['race']} from {region.capitalize()}."
    sheet["backstory"]["allies_organizations"] = [region.capitalize()]
    sheet["metadata"] = {"created_at": datetime.now().isoformat()}
    sheet["abilities"]["skills"] = assign_skills_by_class_and_background(
    class_name,
    sheet["character_info"]["background"]
)
    sheet["combat"]["armor_class"] = 10 + ability_modifier(sheet["abilities"]["dexterity"])
    sheet["combat"]["initiative"] = ability_modifier(sheet["abilities"]["dexterity"])
    equipment = assign_equipment_by_class_and_background(
        class_name,
        sheet["character_info"]["background"],
        sheet["abilities"]["dexterity"]
    )
    sheet["equipment"]["armor"] = equipment["armor"]
    sheet["equipment"]["weapons"] = equipment["weapons"]
    sheet["equipment"]["gear"] = equipment["gear"]
    sheet["combat"]["armor_class"] = equipment["armor_class"]
    
    return sheet
def choose_cantrips_for_class(class_name: str, n: int) -> list:
    """Choose n cantrips weighted by class affinity."""
    cname = class_name.lower()
    weighted = []
    for spell, weights in CANTRIP_POOL.items():
        weight = weights.get(cname, 0)
        if weight > 0:
            weighted.extend([spell] * weight)

    if not weighted:
        return []

    # Weighted random selection without replacement
    selected = random.sample(weighted, min(n, len(set(weighted))))
    return list(dict.fromkeys(selected))
def assign_equipment_by_class_and_background(class_name: str, background: str,
                                             dex_score: int, con_score: int = 10,
                                             wis_score: int = 10) -> dict:
    """Assign armor, weapon, and gear; compute correct AC including Unarmored Defense."""

    # --- Weapons ---
    weapons = {
        "club": {"name": "Club", "damage": "1d4 Bludgeoning"},
        "dagger": {"name": "Dagger", "damage": "1d4 Piercing"},
        "handaxe": {"name": "Handaxe", "damage": "1d6 Slashing"},
        "spear": {"name": "Spear", "damage": "1d6 Piercing"},
        "longsword": {"name": "Longsword", "damage": "1d8 Slashing"},
        "greataxe": {"name": "Greataxe", "damage": "1d12 Slashing"},
        "rapier": {"name": "Rapier", "damage": "1d8 Piercing"},
        "shortsword": {"name": "Shortsword", "damage": "1d6 Piercing"},
        "longbow": {"name": "Longbow", "damage": "1d8 Piercing"},
        "quarterstaff": {"name": "Quarterstaff", "damage": "1d6 Bludgeoning"},
    }

    # --- Armor ---
    armor = {
        "leather": {"name": "Leather Armor", "base_ac": 11, "dex_cap": None, "type": "Light"},
        "scale_mail": {"name": "Scale Mail", "base_ac": 14, "dex_cap": 2, "type": "Medium"},
        "chain_mail": {"name": "Chain Mail", "base_ac": 16, "dex_cap": 0, "type": "Heavy"},
        "chain_shirt": {"name": "Chain Shirt", "base_ac": 13, "dex_cap": 2, "type": "Medium"},
        "shield": {"name": "Shield", "bonus_ac": 2, "type": "Shield"},
    }

    # --- Class-based loadouts ---
    cname = class_name.lower()
    class_loadouts = {
        "barbarian": {"armor": None, "weapon": weapons["greataxe"]},
        "fighter": {"armor": armor["chain_mail"], "weapon": weapons["longsword"]},
        "paladin": {"armor": armor["chain_mail"], "weapon": weapons["longsword"]},
        "ranger": {"armor": armor["scale_mail"], "weapon": weapons["longbow"]},
        "rogue": {"armor": armor["leather"], "weapon": weapons["rapier"]},
        "bard": {"armor": armor["leather"], "weapon": weapons["shortsword"]},
        "monk": {"armor": None, "weapon": weapons["quarterstaff"]},
        "druid": {"armor": armor["leather"], "weapon": weapons["spear"]},
        "cleric": {"armor": armor["chain_shirt"], "weapon": weapons["spear"]},
        "sorcerer": {"armor": None, "weapon": weapons["dagger"]},
        "warlock": {"armor": armor["leather"], "weapon": weapons["dagger"]},
        "wizard": {"armor": None, "weapon": weapons["quarterstaff"]},
        "artificer": {"armor": armor["scale_mail"], "weapon": weapons["handaxe"]},
    }

    background_gear = {
        "acolyte": ["Holy Symbol", "Prayer Book"],
        "soldier": ["Insignia", "Dice Set"],
        "urchin": ["Small Knife", "Map of City"],
        "folk hero": ["Shovel", "Old Medal"],
        "entertainer": ["Lute", "Costume"],
        "criminal": ["Crowbar", "Dark Cloak"],
        "guild artisan": ["Artisan’s Tools", "Ledger"],
        "noble": ["Fine Clothes", "Signet Ring"],
        "sage": ["Scroll Case", "Ink Pen"],
        "hermit": ["Herbal Kit", "Scroll of Notes"],
        "traveler": ["Bedroll", "Waterskin"],
    }

    loadout = class_loadouts.get(cname, {"armor": armor["leather"], "weapon": weapons["dagger"]})
    equipped_armor = loadout["armor"]
    equipped_weapon = loadout["weapon"]

    # --- Compute ability modifiers ---
    dex_mod = (dex_score - 10) // 2
    con_mod = (con_score - 10) // 2
    wis_mod = (wis_score - 10) // 2

    # --- Compute AC ---
    if cname == "barbarian":
        # Unarmored Defense (10 + Dex + Con)
        ac = 10 + dex_mod + con_mod
    elif cname == "monk":
        # Unarmored Defense (10 + Dex + Wis)
        ac = 10 + dex_mod + wis_mod
    elif equipped_armor:
        base_ac = equipped_armor["base_ac"]
        dex_cap = equipped_armor.get("dex_cap")
        if dex_cap is not None:
            dex_mod = min(dex_mod, dex_cap)
        ac = base_ac + dex_mod
    else:
        ac = 10 + dex_mod  # default unarmored

    # Add shield bonus if applicable
    if cname in ["fighter", "paladin", "cleric"]:
        ac += armor["shield"]["bonus_ac"]

    gear_items = background_gear.get(background.lower(), ["Traveler’s Clothes", "Pouch of Coins"])

    return {
        "armor": [equipped_armor["name"]] if equipped_armor else [],
        "weapons": [equipped_weapon["name"]] if equipped_weapon else [],
        "gear": gear_items,
        "armor_class": ac,
    }
   
def populate_by_class(sheet: dict, class_name: str, player_name: str) -> dict:
    """Populate the JSON differently depending on the class."""
    sheet["character_info"]["character_name"] = f"{player_name}'s {class_name.title()}"
    sheet["character_info"]["player_name"] = player_name
    sheet["metadata"] = {"created_at": datetime.now().isoformat()}

    base_stats = {
        "artificer": {"strength": 8, "dexterity": 14, "constitution": 14, "intelligence": 15, "wisdom": 12, "charisma": 8},
        "barbarian": {"strength": 15, "dexterity": 13, "constitution": 14, "intelligence": 10, "wisdom": 12, "charisma": 8},
        "bard":      {"strength": 8,  "dexterity": 14, "constitution": 12, "intelligence": 13, "wisdom": 10, "charisma": 15},
        "cleric":    {"strength": 14, "dexterity": 8,  "constitution": 13, "intelligence": 10, "wisdom": 15, "charisma": 12},
        "druid":     {"strength": 8,  "dexterity": 12, "constitution": 14, "intelligence": 13, "wisdom": 15, "charisma": 10},
        "fighter":   {"strength": 15, "dexterity": 14, "constitution": 13, "intelligence": 8,  "wisdom": 10, "charisma": 12},
        "monk":      {"strength": 12, "dexterity": 15, "constitution": 13, "intelligence": 10, "wisdom": 14, "charisma": 8},
        "paladin":   {"strength": 15, "dexterity": 10, "constitution": 13, "intelligence": 8,  "wisdom": 12, "charisma": 14},
        "ranger":    {"strength": 12, "dexterity": 15, "constitution": 13, "intelligence": 8,  "wisdom": 14, "charisma": 10},
        "rogue":     {"strength": 12, "dexterity": 15, "constitution": 13, "intelligence": 14, "wisdom": 10, "charisma": 8},
        "sorcerer":  {"strength": 10, "dexterity": 13, "constitution": 14, "intelligence": 8,  "wisdom": 12, "charisma": 15},
        "warlock":   {"strength": 8,  "dexterity": 14, "constitution": 13, "intelligence": 12, "wisdom": 10, "charisma": 15},
        "wizard":    {"strength": 8,  "dexterity": 12, "constitution": 13, "intelligence": 15, "wisdom": 14, "charisma": 10},
    }

    cname = class_name.lower()
    for ability, score in base_stats.get(cname, {}).items():
        sheet["abilities"][ability] = score

    con_mod = ability_modifier(sheet["abilities"]["constitution"])
    if cname == "barbarian":
        base_hp = 12
    elif cname in ["fighter", "paladin", "ranger"]:
        base_hp = 10
    elif cname in ["sorcerer", "wizard"]:
        base_hp = 6
    else:
        base_hp = 8

    sheet["combat"]["hit_points"]["maximum"] = base_hp + con_mod
    sheet["combat"]["hit_points"]["current"] = base_hp + con_mod
    sheet["combat"]["hit_dice"]["total"] = f"1d{base_hp}"
    core_stats = ["strength", "dexterity", "constitution", "intelligence", "wisdom", "charisma"]

    mods = {ability: ability_modifier(sheet["abilities"][ability]) for ability in core_stats}
    top_two = sorted(mods.items(), key=lambda x: x[1], reverse=True)[:2]
    top_abilities = {a for a, _ in top_two}

    saving_throws = {a: (a in top_abilities) for a in core_stats}
    sheet["abilities"]["saving_throws"].update(saving_throws)
    cantrip_count = {
        "artificer": 2,
        "bard": 2,
        "cleric": 3,
        "druid": 2,
        "sorcerer": 4,
        "warlock": 2,
        "wizard": 3
    }.get(class_name.lower(), 0)

    cantrips = choose_cantrips_for_class(class_name, cantrip_count)
    sheet["spellcasting"]["cantrips"] = cantrips
    spell_count = LEVEL1_SPELL_COUNTS.get(class_name.lower(), 0)
    spell_slots = LEVEL1_SPELL_SLOTS.get(class_name.lower(), 0)
    level1_spells = choose_spells_for_class(class_name, spell_count)

    sheet["spellcasting"]["spells"]["level_1"]["slots_total"] = spell_slots
    sheet["spellcasting"]["spells"]["level_1"]["slots_expended"] = 0
    sheet["spellcasting"]["spells"]["level_1"]["spells"] = level1_spells
    spellcasting_abilities = {
        "artificer": "intelligence",
        "bard": "charisma",
        "cleric": "wisdom",
        "druid": "wisdom",
        "paladin": "charisma",
        "ranger": "wisdom",
        "sorcerer": "charisma",
        "warlock": "charisma",
        "wizard": "intelligence"
    }

    spell_ability = spellcasting_abilities.get(class_name.lower())
    if spell_ability:
        ability_mod = ability_modifier(sheet["abilities"][spell_ability])
        prof_bonus = 2  # Level 1 default proficiency
        sheet["spellcasting"]["spellcasting_class"] = class_name.title()
        sheet["spellcasting"]["spellcasting_ability"] = spell_ability.title()
        sheet["spellcasting"]["spell_save_dc"] = 8 + prof_bonus + ability_mod
        sheet["spellcasting"]["spell_attack_bonus"] = prof_bonus + ability_mod
    return sheet


# =====================================================
# MAIN ENTRY POINT
# =====================================================
def generate_character_sheet(player_name: str):
    """
    Generate a D&D-style character sheet for the given player.
    Uses per-player data from data/players/player_class_summary.csv.
    """

    # --- Use the corrected player summary path ---
    region_map = load_champion_regions()
    PLAYER_SUMMARY_PATH = os.path.join(PROJECT_ROOT, "backend", "app", "data", "player_class_summary.csv")

    if not os.path.exists(PLAYER_SUMMARY_PATH):
        raise FileNotFoundError(f"[ERROR] Missing player summary file at {PLAYER_SUMMARY_PATH}")

    df = pd.read_csv(PLAYER_SUMMARY_PATH)
    print(f"[DEBUG] Loaded player summary from {PLAYER_SUMMARY_PATH}")
    print("[DEBUG] Columns:", df.columns.tolist())
    print("[DEBUG] Sample rows:\n", df.head())
    if "player_name" not in df.columns:
        raise ValueError(f"[ERROR] {PLAYER_SUMMARY_PATH} missing 'player_name' column")

    # --- Filter to this player
    player_row = df[df["player_name"] == player_name]
    if player_row.empty:
        raise ValueError(f"[ERROR] Player '{player_name}' not found in {PLAYER_SUMMARY_PATH}")
    print(f"[DEBUG] Player rows found: {len(player_row)}")
    print(player_row)
    # --- Extract key info
    class_name = player_row["DND_Class"].iloc[0]
    top_champion = player_row["Top1_Champion"].iloc[0]
    weighted_share = player_row.get("Weighted_Share", pd.Series([50.0])).iloc[0]
    champ_variety_score = float(weighted_share) / 100

    region = find_region_for_champion(top_champion, region_map)

    print(f"== Generating sheet for {player_name} ==")
    print(f"Class: {class_name}, Region: {region}, Champion: {top_champion}")

    # --- Build sheet
    sheet = load_template()
    sheet = populate_common_fields(sheet, class_name, player_name, region, champ_variety_score)
    sheet = populate_by_class(sheet, class_name, player_name)

    # --- Save output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{player_name}_sheet.json")
    with open(output_path, "w") as f:
        json.dump(sheet, f, indent=4)

    print(f"[INFO] Saved character sheet to {output_path}")
    return output_path


if __name__ == "__main__":
    import sys
    player_name = sys.argv[1] if len(sys.argv) > 1 else "pyropiller167#na1"
    print(f"Generating D&D sheet for {player_name}")
    generate_character_sheet(player_name)
