import React, { useEffect, useState } from "react";

export default function DnDCharacterVisualizer({ player }) {
  const [sheet, setSheet] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Fetch the full prediction (which already includes the character JSON)
  const fetchCharacterData = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`http://127.0.0.1:8000/predict/${encodeURIComponent(player)}`);
      const data = await res.json();

      // Try to load JSON from the returned sheet file if it exists
      if (data.character_sheet?.sheet_file) {
        const sheetRes = await fetch(`http://127.0.0.1:8000/static/${data.character_sheet.sheet_file.split("data\\")[1]}`);
        const sheetJson = await sheetRes.json();
        setSheet(sheetJson);
      } else {
        setSheet(data.character_sheet?.sheet || null);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (player) fetchCharacterData();
  }, [player]);

  if (loading) return <div className="text-gray-400">Generating character sheet...</div>;
  if (error) return <div className="text-red-400">Error: {error}</div>;
  if (!sheet) return <div className="text-gray-400">No data yet.</div>;

  const info = sheet.character_info || {};
  const abilities = sheet.abilities || {};
  const combat = sheet.combat || {};
  const equipment = sheet.equipment || {};

  return (
    <div className="bg-zinc-900 text-white p-6 rounded-2xl shadow-lg w-full max-w-5xl mx-auto mt-10">
      <h1 className="text-3xl font-bold mb-2">{info.character_name}</h1>
      <p className="text-zinc-400 text-sm mb-6">
        {info.class} • {info.race} • {info.background} • {info.alignment}
      </p>

      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-2 mb-8">
        {Object.entries(abilities)
          .filter(([_, val]) => typeof val === "number")
          .map(([key, val]) => (
            <div key={key} className="bg-zinc-800 rounded-md text-center py-2">
              <p className="text-sm uppercase text-zinc-400">{key}</p>
              <p className="text-2xl font-semibold">{val}</p>
            </div>
          ))}
      </div>

      <h3 className="text-xl font-semibold mb-3">Combat Stats</h3>
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="bg-zinc-800 p-3 rounded-md text-center">
          <p className="text-sm text-zinc-400">AC</p>
          <p className="text-2xl font-bold">{combat.armor_class}</p>
        </div>
        <div className="bg-zinc-800 p-3 rounded-md text-center">
          <p className="text-sm text-zinc-400">HP</p>
          <p className="text-2xl font-bold">{combat.hit_points?.maximum}</p>
        </div>
        <div className="bg-zinc-800 p-3 rounded-md text-center">
          <p className="text-sm text-zinc-400">Speed</p>
          <p className="text-2xl font-bold">{combat.speed}</p>
        </div>
      </div>

      <h3 className="text-xl font-semibold mb-2">Equipment</h3>
      <ul className="list-disc ml-6 text-sm">
        {(equipment.weapons || []).map((w, i) => <li key={i}>{w}</li>)}
        {(equipment.armor || []).map((a, i) => <li key={i}>{a}</li>)}
        {(equipment.gear || []).map((g, i) => <li key={i}>{g}</li>)}
      </ul>
    </div>
  );
}
