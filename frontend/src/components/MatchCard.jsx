import React, { useState } from "react";
import { runTimelinePipeline, summarizeWithBedrock } from "../api/backend";

export default function MatchCard({ match, region, puuid, playerClass }) {   // ✅ added playerClass prop
  const [expanded, setExpanded] = useState(false);
  const [loading, setLoading] = useState(false);
  const [timelineSummary, setTimelineSummary] = useState(null);
  const [bedrockSummary, setBedrockSummary] = useState(null);
  const [error, setError] = useState(null);

  const {
    champion, role_str, win, kills, deaths, assists,
    ch_kda, cs, vision_score, ch_killParticipation,
    ch_teamDamagePercentage, ch_goldPerMinute,
    duration, date
  } = match;

  const resultColor = win
    ? "border-blue-500/70 bg-blue-900/20"
    : "border-red-500/70 bg-red-900/20";

  const kdaColor =
    ch_kda > 5 ? "text-green-400" :
    ch_kda > 3 ? "text-blue-300" :
    ch_kda > 2 ? "text-yellow-300" :
    "text-red-300";

  async function handleExpand() {
    if (expanded) {
      setExpanded(false);
      return;
    }
    setExpanded(true);
    if (timelineSummary || bedrockSummary) return; // already loaded

    try {
      setLoading(true);
      console.log(`[DEBUG] Running timeline + Bedrock with class: ${playerClass}`); 

      const timelineRes = await runTimelinePipeline(region, match.match_id, puuid);
      const bedrockRes = await summarizeWithBedrock(region, match.match_id, puuid, playerClass || "Rogue");  

      setTimelineSummary(timelineRes.data);
      setBedrockSummary(bedrockRes.data);
      setError(null);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  }
  
  return (
    <div className={`p-4 rounded-lg border ${resultColor}`}>
      {/* MAIN CARD HEADER */}
      <div className="flex items-center justify-between cursor-pointer" onClick={handleExpand}>
        <div className="flex items-center gap-3">
          <div>
            <p className="font-semibold text-white">{champion}</p>
            <p className="text-xs text-gray-400">
              {role_str} • {new Date(date).toLocaleDateString()}
            </p>
          </div>
        </div>

        <div className="text-center">
          <p className={`text-lg font-bold ${kdaColor}`}>
            {kills}/{deaths}/{assists}
          </p>
          <p className="text-xs text-gray-400">KDA {ch_kda?.toFixed(2) ?? "–"}</p>
        </div>

        <div className="text-right text-sm text-gray-400 space-y-1">
          <p>CS: {cs} • Vision: {vision_score}</p>
          <p>
            KP: {(ch_killParticipation * 100).toFixed(0)}% • Dmg:{" "}
            {(ch_teamDamagePercentage * 100).toFixed(0)}%
          </p>
          <p>
            Gold/min: {Math.round(ch_goldPerMinute)} • {Math.round(duration / 60)}m
          </p>
        </div>

        <span
          className={`px-2 py-1 rounded text-xs font-semibold ${
            win ? "bg-blue-600/40 text-blue-200" : "bg-red-600/40 text-red-200"
          }`}
        >
          {win ? "Victory" : "Defeat"}
        </span>
      </div>

      {/* EXPANDED SECTION */}
      {expanded && (
        <div className="mt-3 bg-slate-800/40 border-t border-slate-700 pt-3">
          {loading ? (
            <p className="text-gray-400 italic">Loading timeline & summary...</p>
          ) : error ? (
            <p className="text-red-400 italic">❌ {error}</p>
          ) : (
            <>
              {timelineSummary && (
                <div className="mb-3">
                  <h3 className="text-blue-400 font-semibold text-sm mb-1">
                    Timeline Insights
                  </h3>
                  <div className="text-gray-300 text-sm whitespace-pre-wrap bg-slate-900/40 p-3 rounded leading-relaxed">
                    {(() => {
                      try {
                        const t = timelineSummary.result || timelineSummary;

                        // --- Handle solo kills & teamfights safely ---
                        const soloKills = Array.isArray(t.solo_kills)
                          ? t.solo_kills.length
                          : t.soloKills ?? 0;

                        const teamfights = Array.isArray(t.teamfights)
                          ? t.teamfights.length
                          : t.teamfights_count ?? 0;

                        // --- Lane advantage snapshot (minute 10) ---
                        const lane = t.lane_advantage || t.lane || {};
                        const goldDiff10 = Number(
                          lane.gold_diff_10 ?? lane.goldDiff10 ?? lane.goldAt10 ?? 0
                        );
                        const csDiff10 = Number(
                          lane.cs_diff_10 ?? lane.csDiff10 ?? lane.csAt10 ?? 0
                        );
                        const xpDiff10 = Number(
                          lane.xp_diff_10 ?? lane.xpDiff10 ?? lane.xpAt10 ?? 0
                        );

                        // --- Objective counts ---
                        const objs = Array.isArray(t.objectives) ? t.objectives : [];
                        const dragons = objs.filter(
                          (o) => /DRAGON/i.test(o.monsterType || o.type || "")
                        ).length;
                        const heralds = objs.filter(
                          (o) => /RIFT.?HERALD/i.test(o.monsterType || o.type || "")
                        ).length;
                        const barons = objs.filter(
                          (o) => /BARON/i.test(o.monsterType || o.type || "")
                        ).length;
                        const towers = objs.filter(
                          (o) => /TOWER/i.test(o.monsterType || o.type || "")
                        ).length;

                        // --- Momentum / comebacks ---
                        const momentum = t.momentum || {};
                        const laneSwings = Array.isArray(momentum.lane_swings)
                          ? momentum.lane_swings.length
                          : 0;
                        const teamSwings = Array.isArray(momentum.team_swings)
                          ? momentum.team_swings.length
                          : 0;
                        const hadComeback = teamSwings > 0 || laneSwings > 0;

                        return (
                          <>
                            {soloKills} solo kill{soloKills !== 1 ? "s" : ""} and {teamfights} teamfight
                            {teamfights !== 1 ? "s" : ""} detected.{"\n"}
                            Advantage at 10 min: {goldDiff10 >= 0 ? "+" : ""}
                            {goldDiff10} gold, {csDiff10 >= 0 ? "+" : ""}
                            {csDiff10} CS, {xpDiff10 >= 0 ? "+" : ""}
                            {xpDiff10} XP.{"\n"}
                            Objectives secured: {dragons} dragon{dragons !== 1 ? "s" : ""}, {heralds} herald
                            {heralds !== 1 ? "s" : ""}, {barons} baron{barons !== 1 ? "s" : ""}, {towers} tower
                            {towers !== 1 ? "s" : ""}.{"\n"}
                            {hadComeback
                              ? `${teamSwings + laneSwings} major momentum swing${teamSwings + laneSwings !== 1 ? "s" : ""}.`
                              : "Stable game with no major comebacks."}
                          </>
                        );
                      } catch (err) {
                        console.error("Timeline summary parse error:", err);
                        return "Could not parse timeline summary.";
                      }
                    })()}
                  </div>
                </div>
              )}

              {bedrockSummary && (
                <div className="mt-4">
                  <h3 className="text-purple-400 font-semibold text-sm mb-2">
                    Bedrock Analysis ({bedrockSummary.player_class || "Unspecified Class"})
                  </h3>

                  {(() => {
                    const s = bedrockSummary.summary || bedrockSummary; // handle both formats

                    return (
                      <div className="text-gray-200 text-sm bg-slate-900/50 rounded p-3 space-y-3 leading-relaxed">
                        {/* Lane Phase */}
                        {s.lane_phase_summary && (
                          <section>
                            <h4 className="text-blue-300 font-semibold text-xs uppercase tracking-wide mb-1">
                              Lane Phase Summary
                            </h4>
                            <p className="text-gray-300">{s.lane_phase_summary}</p>
                          </section>
                        )}

                        {/* Momentum */}
                        {s.momentum_summary && (
                          <section>
                            <h4 className="text-blue-300 font-semibold text-xs uppercase tracking-wide mb-1">
                              Momentum Analysis
                            </h4>
                            <p className="text-gray-300">{s.momentum_summary}</p>
                          </section>
                        )}

                        {/* Objectives */}
                        {s.objective_summary && (
                          <section>
                            <h4 className="text-blue-300 font-semibold text-xs uppercase tracking-wide mb-1">
                              Objective Impact
                            </h4>
                            <p className="text-gray-300">{s.objective_summary}</p>
                          </section>
                        )}

                        {/* Player Style */}
                        {s.player_style && (
                          <section>
                            <h4 className="text-blue-300 font-semibold text-xs uppercase tracking-wide mb-1">
                              Player Style
                            </h4>
                            <p className="text-gray-300">{s.player_style}</p>
                          </section>
                        )}

                        {/* Advice */}
                        {Array.isArray(s.advice) && s.advice.length > 0 && (
                          <section>
                            <h4 className="text-blue-300 font-semibold text-xs uppercase tracking-wide mb-1">
                              Improvement Advice
                            </h4>
                            <ul className="list-disc pl-6 text-gray-300 space-y-1">
                              {s.advice.map((tip, i) => (
                                <li key={i}>{tip}</li>
                              ))}
                            </ul>
                          </section>
                        )}

                        {/* Timestamp */}
                        {s.timestamp && (
                          <p className="text-xs text-gray-500 text-right mt-2">
                            Generated: {new Date(s.timestamp).toLocaleString()}
                          </p>
                        )}
                      </div>
                    );
                  })()}
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
