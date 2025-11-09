import axios from "axios";

const API = axios.create({
  baseURL: "http://127.0.0.1:8001", // or your AWS API Gateway URL when deployed
  timeout: 120000,
});

// ----------- HEALTH -----------
export const getHealth = () => API.get("/health/");

// ----------- PREDICTION -----------
export const predictPlayerClass = (playerName) =>
  API.post(`/predict/player/${encodeURIComponent(playerName)}`);

// ----------- CHARACTER SHEETS -----------
export const generateCharacterSheet = (playerName) =>
  API.post(`/sheet/generate`, { player_name: playerName });

// ----------- PROGRESSION -----------
export const updateProgress = (playerName) =>
  API.post(`/progression/progress/${encodeURIComponent(playerName)}`);

// ----------- LEVEL UP -----------
export const levelUpPlayer = (playerName) =>
  API.post(`/levelup/levelup/${encodeURIComponent(playerName)}`);

// ----------- PLAYER -----------
export const findPlayer = (region, summonerName, tag, maxMatches = 50) =>
  API.post(`/player/find`, {
    region: region,
    summoner_name: summonerName,
    tag: tag,
    max_matches: maxMatches,
  });
// ----------- TIMELINE -----------
export const runTimelinePipeline = (region, matchId, puuid) =>
  API.post(`/timeline/pipeline/${region}/${matchId}/${puuid}`);

// ----------- BEDROCK SUMMARY -----------
export const summarizeWithBedrock = (region, matchId, puuid, playerClass) =>
  API.post(`/summary/timeline/bedrock/${region}/${matchId}/${puuid}`, null, {
    params: { player_class: playerClass },
  });

export const getPlayerHistory = (playerName) =>
  API.get(`/player/history/${encodeURIComponent(playerName)}`);

export async function getCharacterSheet(playerName) {
  return API.get(`/sheet/character-sheet/${encodeURIComponent(playerName)}`);
}
export default API;
