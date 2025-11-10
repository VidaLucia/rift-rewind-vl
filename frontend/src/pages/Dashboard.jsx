import { useState } from "react";
import { motion } from "framer-motion";
import { Sparkles, Sword, ScrollText, History, LineChart } from "lucide-react";

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
  const [progressData, setProgressData] = useState(null);
  const [expandedMatch, setExpandedMatch] = useState(null);
  const [matchSummaries, setMatchSummaries] = useState({});
  const [loadingMatch, setLoadingMatch] = useState(null);
  const [showCharacterSheet, setShowCharacterSheet] = useState(false);

  // ======================================================
  // Expand Match Details (Timeline + Bedrock Summary)
  // ======================================================
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

  // ======================================================
  // Handle Button Actions
  // ======================================================
  async function handleAction(action) {
    try {
      setOutput("Running...");
      let res;

      switch (action) {
        case "predict":
          res = await predictPlayerClass(playerName);
          const output = res.data.prediction_output;
          const sheet = res.data.character_sheet?.sheet;
          if (sheet) setCharacterSheet(sheet);
          setPrediction(output);
          setOutput(JSON.stringify(res.data, null, 2));
          break;

        case "progress":
          res = await updateProgress(playerName);
          setProgressData(res.data);
          setOutput(JSON.stringify(res.data, null, 2));
          try {
            const sheetRes = await generateCharacterSheet(playerName);
            const newSheet = sheetRes.data.sheet || sheetRes.data.character_sheet?.sheet;
            if (newSheet) setCharacterSheet(newSheet);
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
              setCharacterSheet(sheetData);
              setShowCharacterSheet(true);
              setOutput("✅ Loaded D&D character sheet successfully.");
            } else {
              setShowCharacterSheet(false);
              setOutput("⚠️ No saved character sheet found for this player.");
            }
          } catch (err) {
            setShowCharacterSheet(false);
            setOutput(`❌ Failed to load D&D sheet: ${err.response?.data?.detail || err.message}`);
          }
          break;

        case "find": {
          const [summonerName, tag] = playerName.split("#");
          res = await findPlayer(region, summonerName, tag);
          const possiblePuuid = res?.data?.puuid || res?.data?.player_puuid;
          if (possiblePuuid) setPuuid(possiblePuuid);
          await fetchHistory();
          break;
        }

        default:
          throw new Error("Unknown action");
      }
    } catch (err) {
      setOutput(`${err.response?.data?.detail || err.message}`);
    }
  }

  // ======================================================
  // Fetch Player Match History
  // ======================================================
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

  // ======================================================
  // UI Layout
  // ======================================================
  return (
    <div className="min-h-screen bg-gradient-to-br from-[#0f172a] via-[#1e1b4b] to-[#4c1d95] text-gray-100 p-8 relative overflow-hidden">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_30%_20%,rgba(56,189,248,0.25),transparent_60%),radial-gradient(circle_at_70%_80%,rgba(147,51,234,0.3),transparent_60%)] pointer-events-none"></div>

      {/* Header */}
      <header className="relative text-center mb-12 z-10">
        <motion.h1
          className="text-5xl font-extrabold bg-gradient-to-r from-fuchsia-400 via-blue-400 to-cyan-400 bg-clip-text text-transparent drop-shadow-[0_0_20px_rgba(56,189,248,0.4)]"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
        >
          What DND Class Are You?
        </motion.h1>
        <p className="text-gray-400 mt-2 text-sm tracking-wide">  
          See what class you are in D&D
        </p>
      </header>

      {/* === Control Panel + Prediction === */}
      <motion.section
        className="max-w-6xl mx-auto grid md:grid-cols-2 gap-6 z-10"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        {/* === Player Setup === */}
        <div className="p-6 rounded-2xl bg-white/10 backdrop-blur-xl border border-white/20 shadow-lg hover:shadow-blue-500/20 transition-all">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Sword className="w-5 h-5 text-blue-400" /> Player Setup
          </h2>

          <div className="flex gap-2 mb-4">
            <select
              value={region}
              onChange={(e) => setRegion(e.target.value)}
              className="bg-white/10 border border-white/20 rounded-xl p-2 w-28 focus:ring-2 focus:ring-blue-400"
            >
              <option value="na1">NA</option>
              <option value="euw1">EUW</option>
              <option value="kr">KR</option>
            </select>

            <input
              value={playerName}
              onChange={(e) => setPlayerName(e.target.value)}
              className="flex-1 bg-white/10 border border-white/20 rounded-xl p-2 placeholder-gray-400 focus:ring-2 focus:ring-fuchsia-400"
              placeholder="e.g. umbreon#glow"
            />
          </div>

          {/* Buttons */}
          <div className="grid grid-cols-2 gap-3">
            <button
              onClick={() => handleAction("predict")}
              className="col-span-2 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-xl py-2 font-semibold hover:opacity-90 shadow-[0_0_10px_#60a5fa]"
            >
              <Sparkles className="inline w-4 h-4 mr-1" /> Predict
            </button>
            <button
              onClick={() => handleAction("progress")}
              className="col-span-2 bg-gradient-to-r from-green-500 to-emerald-600 rounded-xl py-2 font-semibold hover:opacity-90 shadow-[0_0_10px_#34d399]"
            >
              <LineChart className="inline w-4 h-4 mr-1" /> Progress Player
            </button>
            <button
              onClick={() => handleAction("viewSheet")}
              className="col-span-2 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-xl py-2 font-semibold hover:opacity-90 shadow-[0_0_10px_#8b5cf6]"
            >
              <ScrollText className="inline w-4 h-4 mr-1" /> View D&D Sheet
            </button>
            <button
              onClick={() => handleAction("find")}
              className="col-span-2 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-xl py-2 font-semibold hover:opacity-90 shadow-[0_0_10px_#06b6d4]"
            >
              <History className="inline w-4 h-4 mr-1" /> Find Player
            </button>
          </div>
        </div>

        {/* === Prediction Info === */}
        <motion.div
          className="p-6 rounded-2xl bg-gradient-to-br from-purple-900/40 via-indigo-900/30 to-blue-900/20 
                    backdrop-blur-xl border border-purple-400/20 shadow-lg hover:shadow-purple-500/20 
                    transition-all relative overflow-hidden"
          whileHover={{ scale: 1.02 }}
        >
          {/* Background glow */}
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(168,85,247,0.25),transparent_60%)] pointer-events-none" />

          <h2 className="text-2xl font-bold mb-4 text-transparent bg-clip-text 
                        bg-gradient-to-r from-purple-300 via-fuchsia-300 to-blue-300 flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-fuchsia-400" />
            Prediction Info
          </h2>

          {prediction ? (
            <div className="space-y-6 relative z-10">
              {/* === Main Class === */}
              <div className="text-center py-4 rounded-xl bg-white/5 border border-white/10 shadow-inner">
                <p className="text-sm uppercase tracking-wide text-gray-400 mb-1">Class</p>
                <p className="text-3xl font-extrabold text-fuchsia-300 drop-shadow-[0_0_8px_rgba(217,70,239,0.5)]">
                  {prediction.dnd_summary?.DND_Class || "Unknown"}
                </p>
                <p className="text-sm text-gray-400 mt-1">
                  <strong>Main Role:</strong>{" "}
                  <span className="text-blue-300 font-semibold">
                    {prediction.main_role || "N/A"}
                  </span>
                </p>
              </div>

              {/* === Top 3 D&D Class Weights === */}
              {prediction.dnd_summary?.Top3_Classes?.length > 0 && (
                <div>
                  <h3 className="text-sm font-semibold text-fuchsia-300 uppercase tracking-wide mb-3">
                    Class Weight Breakdown
                  </h3>
                  <div className="space-y-3">
                    {prediction.dnd_summary.Top3_Classes.map((cls, idx) => (
                      <div key={idx}>
                        <div className="flex justify-between text-sm mb-1">
                          <span className="text-gray-200 font-medium">{cls.DND_Class}</span>
                          <span className="text-fuchsia-400 font-semibold">{cls.Weighted_Share}%</span>
                        </div>
                        <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden">
                          <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${cls.Weighted_Share}%` }}
                            transition={{ duration: 0.8, delay: idx * 0.1 }}
                            className={`h-full rounded-full ${
                              idx === 0
                                ? "bg-gradient-to-r from-fuchsia-500 to-pink-400"
                                : idx === 1
                                ? "bg-gradient-to-r from-indigo-400 to-blue-400"
                                : "bg-gradient-to-r from-purple-400 to-fuchsia-300"
                            }`}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* === Top Champions === */}
              {prediction.champ_summary && (
                <div>
                  <h3 className="text-sm font-semibold text-blue-300 uppercase tracking-wide mb-3">
                    Top Champions
                  </h3>
                  <div className="grid grid-cols-3 gap-3 text-center">
                    {[1, 2, 3].map((i) => {
                      const champ = prediction.champ_summary[`Top${i}_Champion`];
                      const percent = prediction.champ_summary[`Top${i}_Percent`];
                      return (
                        <div
                          key={i}
                          className="bg-white/5 p-2 rounded-lg border border-white/10 hover:bg-white/10 transition-all"
                        >
                          <p className="text-sm text-gray-200 font-semibold">{champ}</p>
                          <p className="text-xs text-blue-400">{percent}%</p>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <p className="text-gray-500 italic">No prediction yet. Click “Predict”.</p>
          )}
        </motion.div>
      </motion.section>

      {/* === Character Sheet Viewer === */}
      {showCharacterSheet && characterSheet && (
        <motion.section
          className="max-w-6xl mx-auto mt-10 p-6 bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl shadow-xl"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <h2 className="text-xl font-semibold text-blue-300 mb-3">
            D&D Character Sheet
          </h2>

          <div className="bg-slate-800/60 p-4 rounded text-gray-200 space-y-1">
            <p><strong>Name:</strong> {characterSheet.character_info.character_name}</p>
            <p><strong>Class:</strong> {characterSheet.character_info.class}</p>
            <p><strong>Level:</strong> {characterSheet.character_info.level}</p>
            <p><strong>Race:</strong> {characterSheet.character_info.race}</p>
            <p><strong>Alignment:</strong> {characterSheet.character_info.alignment}</p>
            <p><strong>Background:</strong> {characterSheet.character_info.background}</p>
          </div>

          <details className="bg-slate-900/40 rounded p-3 text-gray-300 mt-3">
            <summary className="cursor-pointer text-sm font-semibold text-blue-400">
              View Full JSON
            </summary>
            <pre className="text-xs whitespace-pre-wrap mt-2 bg-slate-950/40 p-3 rounded">
              {JSON.stringify(characterSheet, null, 2)}
            </pre>
          </details>

          <button
            onClick={() => setShowCharacterSheet(false)}
            className="mt-4 px-4 py-2 bg-slate-700 text-gray-200 rounded hover:bg-slate-600"
          >
            Close Sheet
          </button>
        </motion.section>
      )}

      {/* === Match History Section === */}
      <motion.section
        className="max-w-6xl mx-auto mt-10 p-6 bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl shadow-xl"
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h2 className="text-xl font-semibold text-blue-300 mb-2">
          Match History
        </h2>
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
          <p className="text-gray-500">
            No matches yet. Click “Find Player” to load history.
          </p>
        )}
      </motion.section>

      {/* === Console Output === */}
      <motion.section
        className="max-w-6xl mx-auto mt-10 p-6 bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl shadow-xl"
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h2 className="text-xl font-semibold text-blue-300 mb-2">
          Console Output
        </h2>
        <pre className="bg-black/40 text-gray-200 text-sm rounded-xl p-4 max-h-[300px] overflow-y-auto">
          {output || "No output yet."}
        </pre>
      </motion.section>
    </div>
  );
}
