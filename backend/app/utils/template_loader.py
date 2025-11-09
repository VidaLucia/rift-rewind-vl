import os, json

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "../data")

def load_template(name: str):
    """Load a static JSON template (like dnd_sheet or equipment_template)."""
    path = os.path.join(TEMPLATE_DIR, f"{name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Template not found: {name}.json")
    with open(path, "r") as f:
        return json.load(f)
