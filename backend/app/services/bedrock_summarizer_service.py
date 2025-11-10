import os
from backend.scripts.bedrock_summarizer import summarize_summary_file

def run_bedrock_summarizer_service(region: str, match_id: str, puuid: str, player_class: str = None):
    """
    Locate the player's timeline summary file and generate a Bedrock-based advanced summary.
    Optionally accepts a predicted D&D player class (e.g., 'Artificer', 'Rogue', etc.).
    """
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data/timeline"))
    input_file = os.path.join(data_dir, f"{match_id}_player_{puuid}_summary.json")
    output_file = os.path.join(data_dir, f"{match_id}_player_{puuid}_summary_bedrock.json")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Player summary not found: {input_file}")

    structured_output = summarize_summary_file(
        input_path=input_file,
        output_path=output_file,
        player_class=player_class 
    )
    return {
        "status": "success",
        "region": region,
        "match_id": match_id,
        "puuid": puuid,
        "player_class": player_class,  # include in response for logging
        "summary": structured_output,
        "output_path": output_file
    }