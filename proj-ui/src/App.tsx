import { useEffect, useState } from "react";
import { initState, getSummary, listPlayers, makePick, getRecs,
  type RecommendationOut , type InitStateRequest,
  undo, autopick, 
  type PlayerOut} from "./lib/api";
import { createPortal } from "react-dom";

export default function App() {
  const [sessionId, setSessionId] = useState<string | null>(null);

  return sessionId ? (
    <DraftScreen sessionId={sessionId} />
  ) : (
    <InitForm onCreated={setSessionId} />
  );
}


function InitForm({ onCreated }: { onCreated: (id: string) => void }) {
  const [leagueSize, setLeagueSize] = useState(10);
  const [userTeamIndex, setUserTeamIndex] = useState(0);
  const [benchSpots, setBenchSpots] = useState(7);
  const [roster, setRoster] = useState<Record<string, number>>({
    QB: 1,
    RB: 2,
    WR: 2,
    TE: 1,
    FLEX: 1,
    DST: 1,
    K: 1,
  });
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function handleInit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setErr(null);
    try {
      const body: InitStateRequest = {
        league_size: leagueSize,
        user_team_index: userTeamIndex,
        bench_spots: benchSpots,
        starter_requirements: roster,
      };
      const res = await initState(body);
      onCreated(res.session_id);
    } catch (e: any) {
      setErr(e.message ?? "Failed to init");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-xl mx-auto p-6 space-y-4">
      <h1 className="text-2xl font-semibold">Start a Draft</h1>
      <form onSubmit={handleInit} className="space-y-3">
        <div className="grid grid-cols-3 gap-3">
          <label className="col-span-2">League size</label>
          <input
            type="number"
            className="border rounded p-2"
            value={leagueSize}
            onChange={(e) => setLeagueSize(+e.target.value)}
          />
          <label className="col-span-2">Your pick index (0-based)</label>
          <input
            type="number"
            className="border rounded p-2"
            value={userTeamIndex}
            onChange={(e) => setUserTeamIndex(+e.target.value)}
          />
          <label className="col-span-2">Bench spots</label>
          <input
            type="number"
            className="border rounded p-2"
            value={benchSpots}
            onChange={(e) => setBenchSpots(+e.target.value)}
          />
        </div>

        <div className="border rounded p-3 bg-white">
          <div className="font-medium mb-2">Starter requirements</div>
          <div className="grid grid-cols-3 gap-2">
            {Object.entries(roster).map(([pos, val]) => (
              <div key={pos} className="flex items-center gap-2">
                <span className="w-10">{pos}</span>
                <input
                  type="number"
                  className="border rounded p-1 w-16"
                  value={val}
                  onChange={(e) => {
                    const v = Math.max(0, +e.target.value);
                    setRoster((r) => ({ ...r, [pos]: v }));
                  }}
                />
              </div>
            ))}
          </div>
        </div>

        <button
          disabled={loading}
          className="bg-blue-600 text-white rounded px-4 py-2 disabled:opacity-50"
        >
          {loading ? "Creating…" : "Create session"}
        </button>
        {err && <div className="text-red-600">{err}</div>}
      </form>
    </div>
  );
}

function DraftScreen({ sessionId }: { sessionId: string }) {
  const [summary, setSummary] = useState<
    Awaited<ReturnType<typeof getSummary>> | null
  >(null);
  const [players, setPlayers] = useState<
    Awaited<ReturnType<typeof listPlayers>>
  >([]);
  const [recs, setRecs] = useState<RecommendationOut[]>([]);
  const [loading, setLoading] = useState(false);
  const [posFilter, setPosFilter] = useState<string>("");
  const [query, setQuery] = useState("");
  const [limit, setLimit] = useState(20);

  async function undoPick() {
    if (!summary) return;
    setLoading(true);
    try {
      await undo(sessionId);
      setRecs([]);
      await refresh();
    } catch (e: any) {
      alert(e.message ?? "Undo failed");
    } finally {
      setLoading(false);
    }
  }
  
  async function doAutopick() {
    if (!summary) return;
    setLoading(true);
    try {
      await autopick(sessionId, 5);
      setRecs([]);
      await refresh();
    } catch (e: any) {
      alert(e.message ?? "Autopick failed");
    } finally {
      setLoading(false);
    }
  }

  async function refresh() {
    const s = await getSummary(sessionId);
    setSummary(s);
    const p = await listPlayers(sessionId, { position: posFilter || undefined, q: query || undefined, limit });
    setPlayers(p);
  }

  useEffect(() => {
    let t = setTimeout(() => { refresh(); }, 250); // 250ms debounce on query/limit
    return () => clearTimeout(t);
    // include all inputs that affect the list
  }, [sessionId, posFilter, query, limit]);

  async function pick(name: string) {
    if (!summary) return;
    setLoading(true);
    try {
      await makePick(sessionId, summary.on_the_clock_team, name);
      setRecs([]);
      await refresh();
    } catch (e: any) {
      alert(e.message ?? "Pick failed");
    } finally {
      setLoading(false);
    }
  }

  async function fetchRecs() {
    setLoading(true);
    try {
      const r = await getRecs(sessionId, { num_simulations: 20 });
      setRecs(r);
    } catch (e: any) {
      alert(e.message ?? "Recs failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      {/* main content (status + players) */}
      <div className="relative max-w-5xl mx-auto p-6 min-h-screen pr-[380px]">
        <div className="grid gap-6 lg:grid-cols-2">
          {/* left column: status */}
          <div className="p-4 rounded-2xl shadow bg-white">
            <div className="text-sm text-slate-500">Session</div>
            <div className="font-mono text-xs break-all">{sessionId}</div>

            <div className="mt-3">
              <div className="text-sm text-slate-500">On the clock</div>
              <div className="text-xl font-semibold">
                {summary ? `Team ${summary.on_the_clock_team}` : "…"}
              </div>
            </div>

            <div className="mt-3">
              <div className="text-sm text-slate-500 mb-1">Your remaining starters</div>
              <div className="flex flex-wrap gap-2">
                {summary && Object.entries(summary.slots_remaining).map(([pos, n]) => (
                  <span key={pos} className="px-2 py-1 rounded bg-slate-100">{`${pos}:${n}`}</span>
                ))}
              </div>
            </div>

            {/* Controls */}
            <div className="mt-4 flex flex-wrap gap-2">
              <button onClick={refresh} disabled={loading}
                className="bg-slate-800 text-white rounded px-3 py-2 disabled:opacity-50">
                Refresh
              </button>
              <button onClick={undoPick} disabled={loading}
                className="bg-orange-600 text-white rounded px-3 py-2 disabled:opacity-50">
                Undo
              </button>
              <button onClick={doAutopick} disabled={loading}
                className="bg-sky-600 text-white rounded px-3 py-2 disabled:opacity-50">
                Auto-pick
              </button>
            </div>
          </div>

          {/* My Team panel */}
          <div className="p-4 rounded-2xl shadow bg-white">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold">My Team</h2>
              <div className="text-xs text-slate-500">
                {summary ? `${Object.values(summary.my_roster ?? {}).reduce((a, b) => a + b.length, 0)} players` : "…"}
              </div>
            </div>

            <div className="mt-3 space-y-3">
              {summary && (() => {
                const order = ["QB","RB","WR","TE","FLEX","DST","K","BENCH"];
                const roster = summary.my_roster || {};
                return order
                  .filter(pos => roster[pos]?.length)
                  .map((pos: string) => (
                    <div key={pos}>
                      <div className="text-sm font-medium mb-1">{pos}</div>
                      <ul className="space-y-1">
                        {roster[pos].map((p: PlayerOut, i: number) => (
                          <li key={`${pos}-${i}`} className="text-sm flex justify-between">
                            <span className="truncate">{p.name}</span>
                            <span className="text-slate-500">{p.team} • {p.position}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  ));
              })()}
              {/* If totally empty */}
              {summary && Object.values(summary.my_roster || {}).every(arr => (arr as PlayerOut[]).length === 0) && (
                <div className="text-sm text-slate-500">No players yet.</div>
              )}
            </div>
          </div>
  
          {/* middle column: available players */}
          <div className="col-span-1 space-y-3">
            <div className="p-4 rounded-2xl shadow bg-white">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold">Available</h2>
                <select
                  className="border rounded px-2 py-1"
                  value={posFilter}
                  onChange={(e) => setPosFilter(e.target.value)}
                >
                  <option value="">All</option>
                  <option>RB</option><option>WR</option><option>QB</option>
                  <option>TE</option><option>DST</option><option>K</option>
                </select>
                <div className="flex items-center gap-2 mt-2">
                  <input
                    className="border rounded px-2 py-1 flex-1"
                    placeholder="Search players…"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                  />
                  <input
                    className="border rounded px-2 py-1 w-24"
                    type="number"
                    min={5}
                    max={100}
                    value={limit}
                    onChange={(e) => setLimit(+e.target.value)}
                    title="Max results"
                  />
                  <button
                    className="border rounded px-2 py-1"
                    onClick={() => { setQuery(""); }}
                    type="button"
                    title="Clear search"
                  >
                    Clear
                  </button>
                </div>
              </div>
              <ul className="divide-y mt-2">
                {players.map((p) => (
                  <li key={`${p.name}-${p.team}-${p.position}`} className="py-2 flex items-center justify-between">
                    <div>
                      <div className="font-medium">{p.name}</div>
                      <div className="text-xs text-slate-500">
                        {p.team} • {p.position} • ADP {p.adp.toFixed(1)} • FPTS {p.fpts.toFixed(1)}
                      </div>
                    </div>
                    <button
                      disabled={loading}
                      onClick={() => pick(p.name)}
                      className="bg-blue-600 text-white rounded px-3 py-1 disabled:opacity-50"
                    >
                      Draft
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>{/* end inner grid */}
      </div>{/* end outer container */}
  
      {createPortal(
        <div
          id="recs-fixed-shell"
          style={{
            position: "fixed",
            top: 0,
            right: 0,
            height: "100vh",
            width: 360,
            zIndex: 10000,
            pointerEvents: "none", // clicks pass through outer shell
          }}
        >
          <div style={{ height: "100%", padding: 16 }}>
            {/* Card: internally scrollable */}
            <div
              style={{
                height: "100%",
                background: "purple",
                borderRadius: 16,
                boxShadow: "0 10px 30px rgba(0,0,0,0.15)",
                display: "flex",
                flexDirection: "column",
                overflow: "hidden",
                pointerEvents: "auto", // re-enable clicks inside the card
              }}
            >
              <div
                style={{
                  padding: 16,
                  borderBottom: "1px solid rgba(0,0,0,0.08)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  flexShrink: 0,
                }}
              >
                <h2 className="text-lg font-semibold">Recommendations</h2>
                <button
                  disabled={loading}
                  onClick={fetchRecs}
                  className="bg-emerald-600 text-white rounded px-3 py-1 disabled:opacity-50"
                >
                  {loading ? "Simulating…" : "Run"}
                </button>
              </div>

              {/* Only this area scrolls */}
              <div style={{ flex: 1, overflow: "auto", padding: 16 }}>
                <ul className="divide-y">
                  {recs.map((r) => (
                    <li
                      key={`${r.player.name}-${r.player.team}-${r.player.position}`}
                      className="py-2"
                    >
                      <div className="font-medium">
                        {r.player.name}{" "}
                        <span className="text-slate-500 text-sm">
                          ({r.player.position})
                        </span>
                      </div>
                      <div className="text-xs text-slate-500">
                        EV {r.ev_unconditional.toFixed(1)} • P(avail){" "}
                        {(r.p_available * 100).toFixed(0)}% • VOR {r.vor.toFixed(1)} • ADP{" "}
                        {r.adp.toFixed(1)}
                      </div>
                      {!!r.rationale?.length && (
                        <ul className="list-disc ml-5 text-xs mt-1 text-slate-600">
                          {r.rationale.slice(0, 2).map((line, i) => (
                            <li key={i}>{line}</li>
                          ))}
                        </ul>
                      )}
                    </li>
                  ))}
                  {!recs.length && (
                    <li className="py-2 text-sm text-slate-500">No results yet.</li>
                  )}
                </ul>
              </div>
            </div>
          </div>
        </div>,
        document.body
      )}
    </>
  );
}
