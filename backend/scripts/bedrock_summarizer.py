# bedrock_summarizer.py
"""
Generates advanced player summaries using AWS Bedrock Claude Sonnet 4 (2025-05-14)
via an Inference Profile (provisioned throughput).
"""

import os
import json
import boto3
from datetime import datetime

# =====================================================
# CONFIG
# =====================================================
REGION = "us-east-1"
INFERENCE_PROFILE_ARN = (
    "arn:aws:bedrock:us-east-1:742068237220:"
    "inference-profile/global.anthropic.claude-sonnet-4-20250514-v1:0"
)
INPUT_FILE = "player_1_summary.json"
OUTPUT_FILE = "../../data/player_1_summary_bedrock.json"

# =====================================================
# CLIENT
# =====================================================
bedrock = boto3.client("bedrock-runtime", region_name=REGION)

# =====================================================
# PROMPT BUILDER
# =====================================================
def build_prompt(match_data, player_class: str = "Rogue"):
    """
    Builds a detailed, class-based analytical prompt for Claude 4.
    Tone and strategic framing adapt to the player's D&D class.
    """
    class_flavor = {
        "Rogue": "quiet precision, adaptability, and sharp decision-making",
        "Cleric": "balance, support, and timing of intervention",
        "Wizard": "calculated foresight and efficient spell (ability) usage",
        "Fighter": "discipline, aggression control, and mechanical precision",
        "Barbarian": "momentum, resilience, and raw engagement instincts",
        "Druid": "adaptability, flow with natural rhythm (macro play)",
        "Ranger": "positioning, vision control, and target selection",
        "Paladin": "protective instincts and teamfight anchoring",
        "Warlock": "scaling, timing, and pressure through power spikes",
        "Sorcerer": "burst control, risk-reward balancing, and tempo",
        "Bard": "team coordination, map awareness, and tempo orchestration",
        "Monk": "discipline, reactive control, and mechanical precision",
        "Artificer": "ingenuity, optimization, and tactical use of tools or items to gain advantage"

    }

    flavor = class_flavor.get(player_class.title(), class_flavor["Rogue"])

    return f"""
You are a League of Legends analyst and strategist channeling the mindset of a Dungeons & Dragons **{player_class}**.
You speak with analytical clarity, embodying traits of {flavor}.
Avoid roleplay actions or dialogue—respond purely with insightful coaching tone.
Replace any mention of team 200 with Red team and team 100 with Blue team.
Your task: Given structured match data, produce an analytical JSON report that captures both statistical and strategic insights.
Focus on *why* and *how* turning points occurred, not just when.

### Analysis Goals

1. **Lane/Jungle Phase Summary**
   - Examine early CS, gold, and positional control.
   - Identify key power spikes, gank timings, and laning mistakes.
   - Mention whether the player played aggressively or passively and how that affected outcomes.

2. **Momentum Analysis**
   - Detect major gold-swing events and their causes.
   - Comment on recoveries, collapses, or momentum stalls.
   - Highlight if team fights, deaths, or macro errors triggered reversals.

3. **Objective Impact**
   - Correlate specific objectives (Dragon, Baron, Rift Herald, Towers) with gold/momentum changes.
   - Identify which objectives were contested, traded, or thrown away, and who benefited.

4. **Player Style Evaluation**
   - Use kills, deaths, assists, lane data, and objective participation and relate it to their class: {player_class}.
   - Compare early vs. late game decision-making.
   - Be concise and grounded in gameplay terms.
   - 

5. **Improvement Advice**
   - Give three concrete, role-specific suggestions focusing on tactics, map control, and scaling decisions.
   - Avoid vague advice (“play safer”); give situational recommendations (“freeze near tower when behind”, “rotate to mid at level 9 if ahead”).

Return a STRICT JSON object in this format (and nothing else):

{{
  "lane_phase_summary": "",
  "momentum_summary": "",
  "objective_summary": "",
  "player_style": "",
  "advice": ["", "", ""]
}}

### Player Class Context
The player embodies the mindset and playstyle traits of a **{player_class}**, applying them to their in-game decision-making.

### Match Data
{json.dumps(match_data, indent=2)}
"""

# =====================================================
# CALL BEDROCK THROUGH INFERENCE PROFILE
# =====================================================
def call_bedrock(prompt: str) -> str:
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "temperature": 0.3,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ],
    }

    response = bedrock.invoke_model(
    modelId=INFERENCE_PROFILE_ARN,
    body=json.dumps(payload),
    accept="application/json",
    contentType="application/json",
)

    result = json.loads(response["body"].read())
    try:
        return result["content"][0]["text"]
    except (KeyError, IndexError):
        raise RuntimeError(f"Unexpected model response: {json.dumps(result, indent=2)}")

# =====================================================
# MAIN PIPELINE
# =====================================================
def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(INPUT_FILE)

    print(f"[1/3] Loading {INPUT_FILE} ...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("[2/3] Invoking Claude 4 via Inference Profile ...")
    prompt = build_prompt(data, player_class=data.get("class", "Rogue"))
    llm_output = call_bedrock(prompt)

    try:
        structured = json.loads(llm_output)
    except json.JSONDecodeError:
        structured = {"raw_output": llm_output}
    structured["timestamp"] = datetime.now().isoformat()

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(structured, f, indent=2, ensure_ascii=False)

    print(f"[3/3]  Saved → {OUTPUT_FILE}")
def summarize_summary_file(input_path: str, output_path: str = None, player_class: str = None) -> dict:
    """
    Runs the Bedrock summarizer on the given summary JSON and returns the structured result.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    resolved_class = player_class or data.get("class", "Rogue")

    prompt = build_prompt(data, player_class=resolved_class)
    llm_output = call_bedrock(prompt)

    try:
        structured = json.loads(llm_output)
    except json.JSONDecodeError:
        structured = {"raw_output": llm_output}

    structured["timestamp"] = datetime.now().isoformat()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(structured, f, indent=2, ensure_ascii=False)

    return structured
# =====================================================
# ENTRY
# =====================================================
if __name__ == "__main__":
    main()
