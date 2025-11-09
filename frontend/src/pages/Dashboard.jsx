import { useState } from "react";
import {
  getHealth,
  predictPlayerClass,
  generateCharacterSheet,
  updateProgress,
  levelUpPlayer,
  findPlayer,
  runTimelinePipeline,
  summarizeWithBedrock,
  getPlayerHistory,
  getCharacterSheet
} from "../api/backend";
import MatchCard from "../components/MatchCard";

export default function Dashboard() {
  const [playerName, setPlayerName] = useState("umbreon#glow");
  const [region, setRegion] = useState("na1");
  const [matchId, setMatchId] = useState("");
  const [puuid, setPuuid] = useState("");
  const [output, setOutput] = useState("");
  const [matches, setMatches] = useState([]);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [characterSheet, setCharacterSheet] = useState(null);
  const [progressData, setProgressData] = useState(null); // ✅ new
  const [expandedMatch, setExpandedMatch] = useState(null);
  const [matchSummaries, setMatchSummaries] = useState({});
  const [loadingMatch, setLoadingMatch] = useState(null);
  const [showCharacterSheet, setShowCharacterSheet] = useState(false);

  async function handleExpand(match) {
    const matchId = match.match_id;
    if (expandedMatch === matchId) {
      setExpandedMatch(null);
      return;
    }

    try {
      setLoadingMatch(matchId);
      setExpandedMatch(matchId);

      if (matchSummaries[matchId]) return;

      const timelineRes = await runTimelinePipeline(region, matchId, match.puuid || "");
      const bedrockRes = await summarizeWithBedrock(region, matchId, match.puuid || "");

      setMatchSummaries((prev) => ({
        ...prev,
        [matchId]: {
          timeline: timelineRes.data,
          bedrock: bedrockRes.data,
        },
      }));
    } catch (err) {
      console.error("Error loading match details:", err);
      setMatchSummaries((prev) => ({
        ...prev,
        [match.match_id]: { error: err.message },
      }));
    } finally {
      setLoadingMatch(null);
    }
  }

  async function handleAction(action) {
    try {
      setOutput("Running...");
      let res;
      switch (action) {
        case "health":
          res = await getHealth();
          break;

        case "predict":
          res = await predictPlayerClass(playerName);
          const output = res.data.prediction_output;
          const sheet = res.data.character_sheet?.sheet;
          if (sheet) setCharacterSheet(sheet);
          const logText = res.data?.dnd_class_mapping?.log || "";
          const classLines = logText.split("\n").filter((line) => line.match(/WeightedShare/));

          const classEntries = [];
          const regex =
            /(\w+)\s+\|\s+Count:\s+\d+\s+\|\s+AvgSim:\s+[0-9.]+\s+\|\s+WeightedShare:\s+([0-9.]+)%/;
          for (const line of classLines) {
            const match = line.match(regex);
            if (match) classEntries.push({ name: match[1], weight: parseFloat(match[2]) });
          }
          classEntries.sort((a, b) => b.weight - a.weight);
          const topClasses = classEntries.slice(0, 3);
          setPrediction({
            ...output,
            weightedClasses: topClasses,
          });
          setOutput(JSON.stringify(res.data, null, 2));
          break;

        case "progress":
          res = await updateProgress(playerName);
          console.log("[DEBUG] Progress response:", res.data);
          setProgressData(res.data); 
          setOutput(JSON.stringify(res.data, null, 2));
          try {
            const sheetRes = await generateCharacterSheet(playerName);
            const newSheet = sheetRes.data.sheet || sheetRes.data.character_sheet?.sheet;
            if (newSheet) {
              console.log("[DEBUG] Refreshed D&D sheet after progression");
              setCharacterSheet(newSheet);
            } else {
              console.warn("[WARN] No sheet found in response:", sheetRes.data);
            }
          } catch (refreshErr) {
            console.warn("[WARN] Could not refresh D&D sheet:", refreshErr);
          }
          break;

        case "viewSheet":
          setOutput("Loading D&D sheet...");
          try {
            const sheetRes = await getCharacterSheet(playerName);
            const sheetData = sheetRes.data?.sheet;
            if (sheetData) {
              console.log("[DEBUG] Loaded D&D sheet:", sheetData);
              setCharacterSheet(sheetData);
              setShowCharacterSheet(true); // ✅ make it visible
              setOutput("✅ Loaded D&D character sheet successfully.");
            } else {
              setOutput("⚠️ No saved character sheet found for this player.");
              setShowCharacterSheet(false);
            }
          } catch (err) {
            console.error("[ERROR] Failed to load sheet:", err);
            setShowCharacterSheet(false);
            setOutput(`❌ Failed to load D&D sheet: ${err.response?.data?.detail || err.message}`);
          }
          break;
        case "level":
          res = await levelUpPlayer(playerName);
          break;

        case "find": {
          const [summonerName, tag] = playerName.split("#");
          res = await findPlayer(region, summonerName, tag);
          const possiblePuuid = res?.data?.puuid || res?.data?.player_puuid;
          if (possiblePuuid) setPuuid(possiblePuuid);
          await fetchHistory();
          break;
        }

        case "timeline":
          res = await runTimelinePipeline(region, matchId, puuid);
          break;

        case "bedrock":
          res = await summarizeWithBedrock(region, matchId, puuid);
          break;

        default:
          throw new Error("Unknown action");
      }
    } catch (err) {
      setOutput(` ${err.response?.data?.detail || err.message}`);
    }
  }

  async function fetchHistory() {
    try {
      setLoadingHistory(true);
      const res = await getPlayerHistory(playerName);
      setMatches(res.data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoadingHistory(false);
    }
  }

  return (
    <div className="p-8 space-y-6">
      <h1 className="text-3xl font-bold text-blue-400">League Dashboard</h1>

      {/* === Top Controls === */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* LEFT: Player Controls */}
        <div className="card space-y-3">
          <h2 className="text-xl font-semibold text-gray-200">Player</h2>

          <div className="flex gap-2">
            <select
              value={region}
              onChange={(e) => setRegion(e.target.value)}
              className="bg-slate-700 text-white p-2 rounded w-28 border border-slate-600 focus:ring-2 focus:ring-blue-500"
            >
              <option value="na1">NA</option>
              <option value="euw1">EUW</option>
              <option value="kr">KR</option>
            </select>

            <input
              type="text"
              value={playerName}
              onChange={(e) => setPlayerName(e.target.value)}
              placeholder="e.g. umbreon#glow"
              className="flex-1 p-2 bg-slate-700 text-white rounded border border-slate-600 focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div className="grid grid-cols-2 gap-2">
            <button onClick={() => handleAction("predict")} className="btn-primary col-span-2">
              Predict
            </button>

            <button
              onClick={() => handleAction("progress")}
              className="col-span-2 px-4 py-2 rounded-lg font-semibold bg-green-600 hover:bg-green-700 text-white transition-all"
            >
              Progress Player
            </button>
            <button
              onClick={() => handleAction("viewSheet")}
              className="col-span-2 px-4 py-2 rounded-lg font-semibold bg-indigo-600 hover:bg-indigo-700 text-white transition-all"
            >
              View D&D Sheet
            </button>
            <button onClick={() => handleAction("find")} className="btn-primary col-span-2">
              Find Player
            </button>
          </div>
        </div>

        {/* RIGHT: Predict Info */}
        <div className="card space-y-2 p-4">
          <h2 className="text-xl font-semibold text-blue-300">Predict Info</h2>
          {prediction ? (
            <div className="space-y-2">
              <p className="text-lg text-gray-200">
                <span className="font-semibold">Class:</span>{" "}
                <span className="text-blue-400">{prediction.dnd_summary.DND_Class}</span>{" "}
                ({prediction.dnd_summary.Weighted_Share}%)
              </p>

              <p className="text-gray-400 text-sm">
                <strong>Top Champions:</strong><br />
                <p>{prediction.champion_summary.Top1_Champion} ({prediction.champion_summary.Top1_Percent}%) •{" "}</p>
                <p>{prediction.champion_summary.Top2_Champion} ({prediction.champion_summary.Top2_Percent}%) •{" "}</p>
                <p>{prediction.champion_summary.Top3_Champion} ({prediction.champion_summary.Top3_Percent}%)</p>
              </p>

              <p className="text-gray-400 text-sm">
                <strong>Main Role:</strong> {prediction.roles_processed[0]}
              </p>
            </div>
          ) : (
            <p className="text-gray-500 italic">No prediction yet. Click “Predict”.</p>
          )}
        </div>
      </div>

      {/* === Progress Summary === */}
      {progressData && (
        <div className="card space-y-3 mt-6 border-l-4 border-green-500">
          <h2 className="text-xl font-semibold text-green-400 mb-2">Progress Summary</h2>

          {(() => {
            const log = progressData.output || "";
            const kdaMatch = log.match(/KDA:\s*([\d.]+)/);
            const gpmMatch = log.match(/GPM:\s*([\d.]+)/);
            const multiplierMatch = log.match(/Multiplier\s*x([\d.]+)/);
            const levelMatch = log.match(/Level\s+(\d+)/);
            const xpMatch = log.match(/XP\s+(\d+\/\d+)/);
            const commonItemsMatch = log.match(/\[ITEMS\]\s+Level-up \(common\):\s*\[([^\]]+)\]/);
            const rareItemsMatch = log.match(/\[ITEMS\]\s+Level-up \(rare\):\s*\[([^\]]+)\]/);
            const newItems = [];

            if (commonItemsMatch) newItems.push(...commonItemsMatch[1].split(",").map(s => s.trim().replace(/['"]/g, "")));
            if (rareItemsMatch) newItems.push(...rareItemsMatch[1].split(",").map(s => s.trim().replace(/['"]/g, "")));

            return (
              <div className="bg-slate-800/60 p-4 rounded text-gray-200 space-y-2">
                <p><strong>Status:</strong> {progressData.status}</p>
                {kdaMatch && <p><strong>Avg KDA:</strong> {kdaMatch[1]}</p>}
                {gpmMatch && <p><strong>Avg GPM:</strong> {gpmMatch[1]}</p>}
                {multiplierMatch && <p><strong>Multiplier:</strong> ×{multiplierMatch[1]}</p>}
                {levelMatch && <p><strong>🏅Levels Gained:</strong> {levelMatch[1]}</p>}
                {xpMatch && <p><strong>XP Progress:</strong> {xpMatch[1]}</p>}
                {newItems.length > 0 && (
                  <div>
                    <strong> Items Earned:</strong>
                    <ul className="list-disc list-inside text-sm text-gray-300">
                      {newItems.map((it, i) => (
                        <li key={i}>{it}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            );
          })()}

          <details className="bg-slate-900/40 rounded p-3 text-gray-300">
            <summary className="cursor-pointer text-sm font-semibold text-green-400">
              View Raw JSON
            </summary>
            <pre className="text-xs whitespace-pre-wrap mt-2 bg-slate-950/40 p-3 rounded">
              {JSON.stringify(progressData, null, 2)}
            </pre>
          </details>
        </div>
      )}
      {/* === Output === */}
      <div className="card">
        <h2 className="text-xl font-semibold text-gray-200 mb-2">Output</h2>
        <pre className="text-sm text-gray-300 whitespace-pre-wrap overflow-auto max-h-[400px] bg-slate-900/50 p-4 rounded">
          {output || "No output yet."}
        </pre>
      </div>

      {/* === Character Sheet === */}
      {showCharacterSheet && characterSheet && (
        <div className="card space-y-3 mt-6">
          <h2 className="text-xl font-semibold text-blue-300 mb-2">D&D Character Sheet</h2>
          <div className="bg-slate-800/60 p-4 rounded text-gray-200">
            <p><strong>Name:</strong> {characterSheet.character_info.character_name}</p>
            <p><strong>Class:</strong> {characterSheet.character_info.class}</p>
            <p><strong>Level:</strong> {characterSheet.character_info.level}</p>
            <p><strong>Race:</strong> {characterSheet.character_info.race}</p>
            <p><strong>Alignment:</strong> {characterSheet.character_info.alignment}</p>
            <p><strong>Background:</strong> {characterSheet.character_info.background}</p>
          </div>

          <details className="bg-slate-900/40 rounded p-3 text-gray-300">
            <summary className="cursor-pointer text-sm font-semibold text-blue-400">
              View Full JSON
            </summary>
            <pre className="text-xs whitespace-pre-wrap mt-2 bg-slate-950/40 p-3 rounded">
              {JSON.stringify(characterSheet, null, 2)}
            </pre>
          </details>

          <button
            onClick={() => setShowCharacterSheet(false)}
            className="mt-2 px-4 py-2 bg-slate-700 text-gray-200 rounded hover:bg-slate-600"
          >
            Close Sheet
          </button>
        </div>
      )}

      {/* === Match History === */}
      <div className="card space-y-3 mt-6">
        <h2 className="text-xl font-semibold text-blue-300 mb-2">Match History</h2>
        {loadingHistory ? (
          <p className="text-gray-400">Loading match history...</p>
        ) : matches.length > 0 ? (
          <div className="space-y-2">
            {matches.map((m, i) => (
              <MatchCard
                key={i}
                match={m}
                region={region}
                puuid={puuid}
                playerClass={prediction?.dnd_summary?.DND_Class || "Rogue"}
                isExpanded={expandedMatch === m.match_id}
                isLoading={loadingMatch === m.match_id}
                summary={matchSummaries[m.match_id]}
                onExpand={handleExpand}
              />
            ))}
          </div>
        ) : (
          <p className="text-gray-500">No matches yet. Click “Find Player” to load history.</p>
        )}
      </div>
    </div>
  );
}
