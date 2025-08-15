import { API_BASE } from "../config";

async function http<T>(path: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText} – ${text}`);
  }
  return res.json();
}

export type InitStateRequest = {
  league_size: number;
  user_team_index: number;
  bench_spots: number;
  starter_requirements: Record<string, number>;
};
export type InitStateResponse = {
  session_id: string;
  current_pick_index: number;
  draft_order: number[];
};

export async function initState(body: InitStateRequest) {
  return http<InitStateResponse>("/state/init", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export type StateSummary = {
    session_id: string;
    current_pick_index: number;
    on_the_clock_team: number;
    slots_remaining: Record<string, number>;
    my_roster?: Record<string, PlayerOut[]>;
  };
  
export async function getSummary(sessionId: string) {
const url = `/state/summary?session_id=${encodeURIComponent(sessionId)}`;
return http<StateSummary>(url);
}

export type PlayerOut = {
name: string;
team: string;
position: string;
fpts: number;
adp: number;
};

export async function listPlayers(sessionId: string, opts?: { position?: string; q?: string; limit?: number; }) {
const params = new URLSearchParams({ session_id: sessionId });
if (opts?.position) params.set("position", opts.position);
if (opts?.q) params.set("q", opts.q);
if (opts?.limit) params.set("limit", String(opts.limit));
return http<PlayerOut[]>(`/players?${params.toString()}`);
}

export async function makePick(sessionId: string, teamIndex: number, playerName: string) {
return http<StateSummary>("/state/pick", {
    method: "POST",
    body: JSON.stringify({ session_id: sessionId, team_index: teamIndex, player_name: playerName }),
});
}

export type RecommendationOut = {
player: PlayerOut;
ev_if_available: number;
p_available: number;
ev_unconditional: number;
vor: number;
adp: number;
adp_delta: number;
fills_slot: string;
rationale: string[];
};

export async function getRecs(sessionId: string, body?: Partial<{
candidate_pool_size: number;
num_simulations: number;
use_vor: boolean;
weight_adp: number;
candidate_adp_margin: number;
candidate_pool_balanced: boolean;
candidate_pool_dynamic: boolean;
opponent_top_n: number;
user_strategy_weight_adp: number;
user_strategy_top_n: number;
allow_flex_early: boolean;
flex_threshold: number;
allow_bench_early: boolean;
bench_threshold: number;
}>) {
const payload = {
    session_id: sessionId,
    candidate_pool_size: 10,
    num_simulations: 50,    // start small; can expose a slider later
    use_vor: true,
    weight_adp: -1.0,
    candidate_adp_margin: 3,
    candidate_pool_dynamic: true,
    opponent_top_n: 5,
    user_strategy_weight_adp: -0.5,
    user_strategy_top_n: 5,
    allow_flex_early: true,
    flex_threshold: 0.7,
    allow_bench_early: false,
    bench_threshold: 1.0,
    ...body,
};
return http<RecommendationOut[]>("/recs", {
    method: "POST",
    body: JSON.stringify(payload),
});
}


export function autopick(sessionId: string, topN = 5) {
    return http<StateSummary>(
      `/state/autopick?session_id=${encodeURIComponent(sessionId)}&top_n=${topN}`,
      { method: "POST" }
    );
  }

export function undo(sessionId: string) {
    return http<StateSummary>(`/state/undo?session_id=${encodeURIComponent(sessionId)}`, {
      method: "POST",
    });
  }

export async function undoPick(sessionId: string) {
const res = await fetch(`${import.meta.env.VITE_API_URL}/state/undo?session_id=${encodeURIComponent(sessionId)}`, {
    method: "POST",
});
if (!res.ok) throw new Error(await res.text());
return res.json(); // returns StateSummary
}

export async function autoPick(sessionId: string, topN = 5) {
const res = await fetch(`${import.meta.env.VITE_API_URL}/state/autopick?session_id=${encodeURIComponent(sessionId)}&top_n=${topN}`, {
    method: "POST",
});
if (!res.ok) throw new Error(await res.text());
return res.json(); // returns StateSummary
}
