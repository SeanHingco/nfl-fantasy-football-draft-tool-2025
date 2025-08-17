"""
draft_simulator.py
===================

Helper functions for running draft simulations.  This module contains
utilities to copy the draft state without mutating the original, select
players for opponents based on ADP and positional need, and compute
candidate player pools.  The full Monte Carlo simulation logic that
combines these helpers will be built on top of these functions.
"""

from __future__ import annotations

import copy
import random
from typing import Callable, Dict, List, Optional, Sequence

from draft_models import Player, TeamRoster, Recommendation
from draft_state import DraftState
from collections import defaultdict
from typing import Optional, Dict, List, Sequence, Tuple, Any
import os, math
from concurrent.futures import ProcessPoolExecutor, as_completed

SIM_WORKERS = int(os.getenv("SIM_WORKERS", str(os.cpu_count() or 1)))

def _simulate_candidate_once(
    draft_state,
    candidate,
    user_team_idx: int,
    starter_requirements: Dict[str, int],
    replacement_levels,
    user_strategy_weight_adp: float,
    user_strategy_top_n: int,
    opponent_top_n: int,
    allow_flex_early: bool,
    flex_threshold: float,
    allow_bench_early: bool,
    bench_threshold: float,
):
    """
    One simulation for a single candidate.
    Returns (success: bool, value_if_success: float).
    """
    import random

    # utils available in your file:
    # copy_draft_state, compute_flex_decision, compute_bench_decision,
    # score_players, select_player_for_team, evaluate_team_value

    sim_state = copy_draft_state(draft_state)
    forced_candidate = False

    def same(a, b):
        return (a.name, a.team, a.position) == (b.name, b.team, b.position)

    while not sim_state.is_draft_over():
        team_idx = sim_state.get_current_team_index()

        if team_idx == user_team_idx:
            if not forced_candidate:
                cand_sim = next((p for p in sim_state.available_players if same(p, candidate)), None)
                if cand_sim is not None and sim_state.teams[user_team_idx].can_draft(cand_sim):
                    sim_state.make_pick(user_team_idx, cand_sim)
                    forced_candidate = True
                else:
                    sim_state.advance_pick()
                    break  # candidate unavailable -> unsuccessful run
            else:
                eligible = [p for p in sim_state.available_players
                            if sim_state.teams[user_team_idx].can_draft(p)]
                if eligible:
                    # Decide gates
                    flex_now, is_flex_only = compute_flex_decision(
                        sim_state.teams[user_team_idx],
                        eligible,
                        starter_requirements,
                        replacement_levels,
                        allow_flex_early=allow_flex_early,
                        flex_threshold=flex_threshold,
                    )
                    bench_now, is_bench_only = compute_bench_decision(
                        sim_state.teams[user_team_idx],
                        eligible,
                        starter_requirements,
                        replacement_levels,
                        allow_bench_early=allow_bench_early,
                        bench_threshold=bench_threshold,
                    )

                    # Base score (ADP blend)
                    scores = score_players(eligible, weight_adp=user_strategy_weight_adp)

                    # Apply nudges/penalties
                    for p in eligible:
                        if is_flex_only.get(p, False):
                            scores[p] += 0.25 if flex_now else -1e6
                        if is_bench_only.get(p, False):
                            scores[p] += 0.25 if bench_now else -1e6

                    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                    top_pool = [p for p, _ in ranked[:user_strategy_top_n]
                                if sim_state.teams[user_team_idx].can_draft(p)]
                    if top_pool:
                        sim_state.make_pick(user_team_idx, random.choice(top_pool))
            sim_state.advance_pick()
        else:
            opp_player = select_player_for_team(sim_state, team_idx, top_n=opponent_top_n)
            if opp_player:
                sim_state.make_pick(team_idx, opp_player)
            sim_state.advance_pick()

    if not forced_candidate:
        return False, 0.0

    value = evaluate_team_value(sim_state.teams[user_team_idx], starter_requirements)
    return True, float(value)


def evaluate_candidate_batch(
    draft_state,
    candidate,
    *,
    num_simulations: int,
    user_team_idx: int,
    starter_requirements: Dict[str, int],
    replacement_levels,
    user_strategy_weight_adp: float,
    user_strategy_top_n: int,
    opponent_top_n: int,
    allow_flex_early: bool,
    flex_threshold: float,
    allow_bench_early: bool,
    bench_threshold: float,
) -> Dict[str, Any]:
    """
    Runs `num_simulations` for one candidate and returns mergeable aggregates.
    """
    total_value = 0.0
    successful_runs = 0

    for _ in range(num_simulations):
        ok, val = _simulate_candidate_once(
            draft_state, candidate, user_team_idx, starter_requirements, replacement_levels,
            user_strategy_weight_adp, user_strategy_top_n, opponent_top_n,
            allow_flex_early, flex_threshold, allow_bench_early, bench_threshold,
        )
        if ok:
            total_value += val
            successful_runs += 1

    return {
        "total_value": total_value,
        "successful_runs": successful_runs,
        "num_simulations": num_simulations,
    }


def evaluate_team_value(team: TeamRoster, starter_requirements: Dict[str, int]) -> float:
    """Estimate the fantasy value of a team's roster.

    This function sums the projected points of the players who would start
    each week according to the roster requirements.  For positions with
    multiple starters (e.g. two RBs), the highest-scoring players at that
    position are used.  The FLEX slot is filled with the highest-scoring
    remaining RB/WR/TE not already counted.  Bench players beyond the
    required slots are ignored in this evaluation.

    Parameters
    ----------
    team : TeamRoster
        The team whose roster to evaluate.
    starter_requirements : Dict[str, int]
        Mapping of base positions to required starter counts.  Must
        include "FLEX" to specify the number of flex slots.

    Returns
    -------
    float
        The sum of projected points for the starting lineup.
    """
    total_points = 0.0
    # Keep track of players already counted to avoid double-counting
    used_players: set[Player] = set()
    # Evaluate base positions (excluding FLEX)
    for pos, count in starter_requirements.items():
        if pos == "FLEX" or count <= 0:
            continue
        players_at_pos = team.roster.get(pos, [])
        # Sort descending by projected points
        players_sorted = sorted(players_at_pos, key=lambda p: p.fpts, reverse=True)
        for player in players_sorted[:count]:
            total_points += player.fpts
            used_players.add(player)
    # Handle FLEX positions
    flex_count = starter_requirements.get("FLEX", 0)
    if flex_count > 0:
        # Gather all eligible players (RB/WR/TE) not already used
        flex_candidates: List[Player] = []
        for p in team.roster.get("RB", []):
            if p not in used_players:
                flex_candidates.append(p)
        for p in team.roster.get("WR", []):
            if p not in used_players:
                flex_candidates.append(p)
        for p in team.roster.get("TE", []):
            if p not in used_players:
                flex_candidates.append(p)
        # Sort candidates by points and pick top flex_count
        flex_candidates_sorted = sorted(flex_candidates, key=lambda p: p.fpts, reverse=True)
        for player in flex_candidates_sorted[:flex_count]:
            total_points += player.fpts
            used_players.add(player)
    return total_points

def compute_replacement_levels(
    players: Sequence[Player],
    starter_requirements: Dict[str, int],
    n_teams: int
) -> Dict[str, float]:
    """
    For each position, compute a replacement-level projection by taking
    the projection of the last starter you'd expect across the league.

    Example (10 teams): QB -> 10th best, RB -> 20th best, WR -> 20th best, TE -> 10th, DST -> 10th, K -> 10th.
    """
    by_pos: Dict[str, List[Player]] = defaultdict(list)
    for p in players:
        by_pos[p.position.upper()].append(p)

    # Sort each position by projected points (desc)
    for pos in by_pos:
        by_pos[pos].sort(key=lambda p: p.fpts, reverse=True)

    repl: Dict[str, float] = {}
    for pos, starters_per_team in starter_requirements.items():
        pos_up = pos.upper()
        pool = by_pos.get(pos_up, [])
        # how many starters total in the league at this position
        total_starters = starters_per_team * n_teams
        # edge cases: if pool too small, use last player's fpts or 0
        if not pool:
            repl[pos_up] = 0.0
        else:
            idx = min(max(total_starters - 1, 0), len(pool) - 1)
            repl[pos_up] = float(pool[idx].fpts)
    return repl


def score_players_vor(
    players: Sequence[Player],
    replacement_levels: Dict[str, float],
    *,
    weight_adp: float = -1.0
) -> Dict[Player, float]:
    """
    Score = VOR + (weight_adp * ADP)
    where VOR = player.fpts - replacement_levels[player.position]
    """
    scores: Dict[Player, float] = {}
    for p in players:
        pos = p.position.upper()
        repl = replacement_levels.get(pos, 0.0)
        vor = p.fpts - repl
        scores[p] = vor + weight_adp * p.adp
    return scores

def player_vor(p: Player, replacement_levels: Dict[str, float]) -> float:
    """Value over replacement for a single player, using provided replacement levels."""
    return p.fpts - replacement_levels.get(p.position.upper(), 0.0)

def compute_flex_decision(
    team: TeamRoster,
    eligible: List[Player],
    starter_requirements: Dict[str, int],
    replacement_levels: Dict[str, float],
    allow_flex_early: bool,
    flex_threshold: float,
) -> Tuple[bool, Dict[Player, bool]]:
    """
    Decide whether taking a FLEX-only player is justified now.
    Returns (flex_now, is_flex_only_map).

    - 'flex_now' is True if the best FLEX-only VOR beats the best base-slot VOR by >= threshold.
    - 'is_flex_only_map[p]' is True if p can't fill a base slot but could fill FLEX.
    """
    # Remaining starter slots right now
    slots = team.slots_remaining

    # Partition eligible players into base-slot fillers vs flex-only (RB/WR/TE with no base slot left)
    is_flex_only: Dict[Player, bool] = {}
    base_candidates: List[Player] = []
    flex_only_candidates: List[Player] = []

    for p in eligible:
        pos = p.position.upper()
        can_base = slots.get(pos, 0) > 0
        can_flex = (pos in {"RB", "WR", "TE"}) and (slots.get("FLEX", 0) > 0)
        flex_only_flag = (not can_base) and can_flex
        is_flex_only[p] = flex_only_flag
        if flex_only_flag:
            flex_only_candidates.append(p)
        elif can_base:
            base_candidates.append(p)
        # if neither base nor flex fits, we ignore here (bench logic handled elsewhere)

    # If FLEX early is disallowed, we short-circuit decision
    if not allow_flex_early:
        return (False, is_flex_only)

    # Compute best VOR for base and for flex-only
    best_base_vor = max((player_vor(p, replacement_levels) for p in base_candidates), default=float("-inf"))
    best_flex_vor = max((player_vor(p, replacement_levels) for p in flex_only_candidates), default=float("-inf"))

    # If there are no base candidates, FLEX is obviously allowed
    if best_base_vor == float("-inf"):
        return (True, is_flex_only)

    # Gate: only allow FLEX if it's genuinely better by threshold
    flex_now = best_flex_vor >= (best_base_vor + flex_threshold)
    return (flex_now, is_flex_only)

def compute_bench_decision(
    team: TeamRoster,
    eligible: List[Player],
    starter_requirements: Dict[str, int],
    replacement_levels: Dict[str, float],
    allow_bench_early: bool,
    bench_threshold: float,
) -> Tuple[bool, Dict[Player, bool]]:
    """
    Returns (bench_now, is_bench_only_map).
    'bench_now' True means it's justified to draft bench-only now.
    A player is 'bench-only' if they cannot fill any base slot nor FLEX.
    """
    slots = team.slots_remaining

    def can_base(p: Player) -> bool:
        return slots.get(p.position.upper(), 0) > 0

    def can_flex(p: Player) -> bool:
        pos = p.position.upper()
        return (pos in {"RB", "WR", "TE"}) and (slots.get("FLEX", 0) > 0)

    is_bench_only: Dict[Player, bool] = {}
    base_or_flex: List[Player] = []
    bench_only: List[Player] = []

    for p in eligible:
        cb = can_base(p)
        cf = can_flex(p)
        bench_flag = not (cb or cf)
        is_bench_only[p] = bench_flag
        if bench_flag:
            bench_only.append(p)
        else:
            base_or_flex.append(p)

    if not allow_bench_early:
        return (False, is_bench_only)

    best_starter_vor = max((player_vor(p, replacement_levels) for p in base_or_flex), default=float("-inf"))
    best_bench_vor = max((player_vor(p, replacement_levels) for p in bench_only), default=float("-inf"))

    # If no starter-eligible candidates exist, allow bench
    if best_starter_vor == float("-inf"):
        return (True, is_bench_only)

    bench_now = best_bench_vor >= (best_starter_vor + bench_threshold)
    return (bench_now, is_bench_only)

def current_round(draft_state: DraftState) -> int:
    return (draft_state.current_pick_index // draft_state.n_teams) + 1

def _flex_share(slots_remaining: Dict[str,int]) -> Dict[str,float]:
    """Distribute FLEX capacity across RB/WR/TE equally (soft heuristic)."""
    flex = float(slots_remaining.get("FLEX", 0))
    if flex <= 0:
        return {"RB":0.0,"WR":0.0,"TE":0.0}
    # equal split; you could make this smarter later (e.g., weight by available talent)
    share = flex / 3.0
    return {"RB":share, "WR":share, "TE":share}

def compute_position_need_weights(
    team: TeamRoster,
    *,
    allow_bench_early: bool,
    bench_threshold: float,  # not used here, but keep signature parallel if you later blend bench demand
) -> Dict[str, float]:
    """
    Return non-negative weights per position proportional to how much we still 'need' them:
      - base need = remaining starter slots for that position
      - plus a share of FLEX capacity for RB/WR/TE
      - bench demand is 0 unless you decide to include it later (we keep it simple now)
    Positions with no base slots left (and not bench-eligible per your can_draft rules) get weight 0.
    """
    slots = team.slots_remaining  # e.g., {'QB':1,'RB':2,'WR':2,'TE':1,'FLEX':1,'DST':1,'K':1}
    # base positional needs
    need = {pos: max(0.0, float(cnt)) for pos, cnt in slots.items() if pos != "FLEX"}

    # add FLEX share to RB/WR/TE
    flex = _flex_share(slots)
    for pos in ("RB","WR","TE"):
        need[pos] = need.get(pos, 0.0) + flex[pos]

    # If a position has 0 base slots and (by your rules) typically no bench, keep it at 0.
    # Your TeamRoster.can_draft already prevents benching K/DST; so when K/DST slots hit 0, they drop out naturally.

    # Never return negatives
    for k in list(need.keys()):
        if need[k] < 0:
            need[k] = 0.0

    return need  # e.g., {'QB':0,'RB':2.33,'WR':2.33,'TE':1.33,'DST':0,'K':0}
    

def compute_caps_from_weights(
    pool_size: int,
    weights: Dict[str, float],
    hard_min: Dict[str, int] | None = None,
    hard_max: Dict[str, int] | None = None,
) -> Dict[str, int]:
    """
    Turn weights into integer caps that sum to pool_size.
    - Apply hard mins first (useful if you want to always show at least 1 TE when TE still needed).
    - Then distribute the remaining slots by proportional weights with round-robin for rounding.
    - Apply hard max (optional) at the end; any overflow is redistributed.
    """
    hard_min = hard_min or {}
    hard_max = hard_max or {}

    # Start with mins
    caps = {pos: int(hard_min.get(pos, 0)) for pos in weights}
    used = sum(caps.values())
    remaining = max(0, pool_size - used)

    # Normalize weights (ignore positions already satisfied by hard min)
    # If all weights are zero, fall back to uniform over positions present.
    residual_weights = {pos: max(0.0, weights.get(pos, 0.0)) for pos in weights}
    total_w = sum(residual_weights.values())
    if total_w == 0:
        # uniform among keys
        for pos in residual_weights:
            residual_weights[pos] = 1.0
        total_w = float(len(residual_weights))

    # fractional desired adds
    desired = {pos: (remaining * (w / total_w)) for pos, w in residual_weights.items()}
    # greedy rounding by largest fractional part
    floor = {pos: int(desired[pos]) for pos in desired}
    frac = sorted(((desired[p] - floor[p], p) for p in desired), reverse=True)

    caps.update({pos: caps.get(pos, 0) + floor[pos] for pos in floor})
    allocated = sum(floor.values())
    left = remaining - allocated

    i = 0
    while left > 0 and i < len(frac):
        _, pos = frac[i]
        caps[pos] = caps.get(pos, 0) + 1
        left -= 1
        i += 1

    # apply hard max if provided, redistribute overflow
    if hard_max:
        overflow = 0
        for pos in list(caps.keys()):
            mx = hard_max.get(pos, None)
            if mx is not None and caps[pos] > mx:
                overflow += caps[pos] - mx
                caps[pos] = mx
        if overflow > 0:
            # redistribute overflow to others without max or not at max
            candidates = [p for p in caps if (hard_max.get(p, 10**9) > caps[p])]
            j = 0
            while overflow > 0 and candidates:
                p = candidates[j % len(candidates)]
                caps[p] += 1
                overflow -= 1
                j += 1

    # Final sanity: ensure sum == pool_size
    total_caps = sum(caps.values())
    if total_caps < pool_size:
        # assign extras to the highest-weight positions
        order = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        k = 0
        while total_caps < pool_size and order:
            pos = order[k % len(order)][0]
            caps[pos] = caps.get(pos, 0) + 1
            total_caps += 1
            k += 1
    elif total_caps > pool_size:
        # trim from lowest-weight positions
        order = sorted(weights.items(), key=lambda x: x[1])
        k = 0
        while total_caps > pool_size and order:
            pos = order[k % len(order)][0]
            if caps.get(pos, 0) > 0:
                caps[pos] -= 1
                total_caps -= 1
            k += 1

    return caps

def recommend_players(
    draft_state: DraftState,
    *,
    starter_requirements: Dict[str, int],
    candidate_pool_size: int = 15,
    num_simulations: int = 50,
    weight_adp: float = -0.1,
    opponent_top_n: int = 5,
    user_strategy_weight_adp: float = -0.1,
    user_strategy_top_n: int = 5,
    candidate_adp_margin: int = 0,
    use_vor: bool = True,
    candidate_pool_balanced: bool = False,
    candidate_pool_dynamic: bool = False,
    per_position_caps: Optional[Dict[str, int]] = None,
    allow_flex_early: bool = True,
    flex_threshold: float = 0.5,   # how much better FLEX must be vs best base (VOR points)
    allow_bench_early: bool = False,
    bench_threshold: float = 1.0,
) -> List[Recommendation]:
    """Recommend players for the user's next pick using Monte Carlo simulation.

    This function evaluates a pool of candidate players for the user's next
    draft pick by simulating the remainder of the draft many times.
    For each candidate, it forces the user to draft that player at their
    next pick in each simulation, fills in the rest of the draft using the
    provided opponent strategy and a simple user strategy for subsequent
    picks, and computes the average value of the user's final roster.

    Parameters
    ----------
    draft_state : DraftState
        The current draft state.  This object will **not** be modified.
    starter_requirements : Dict[str, int]
        Positional starter requirements (including FLEX) used to evaluate
        final rosters.
    candidate_pool_size : int, optional
        Number of top-scoring players to consider as potential picks.  The
        scoring is determined by :func:`score_players` using
        ``weight_adp``.  Defaults to 15.
    num_simulations : int, optional
        Number of simulations to run for each candidate.  More
        simulations yield more accurate estimates but increase runtime.
        Defaults to 100.
    weight_adp : float, optional
        Weight applied to ADP when ranking candidates for the pool.  A
        negative value favors earlier ADP.  Defaults to -0.1.
    opponent_top_n : int, optional
        How many top ADP players to consider when opponents draft.
        Defaults to 5.
    user_strategy_weight_adp : float, optional
        ADP weight for the simple user strategy used when drafting in
        simulated later rounds.  Defaults to -0.1.
    user_strategy_top_n : int, optional
        How many of the highest-scoring players to consider when the user
        drafts in simulated later rounds.  Defaults to 5.

    Returns
    -------
    List[Recommendation]
        A list of recommendations sorted by expected value
        in descending order.  The expected value is the average projected
        points of the user's final roster across all simulations where the
        player was drafted.
    """
    import random
    import time
    # Identify the index of the user's next pick
    user_team_idx = draft_state.user_team_index
    # Find next pick index for user in the original state
    user_picks = [i for i in draft_state.get_pick_number_for_team(user_team_idx) if i >= draft_state.current_pick_index]
    if not user_picks:
        return []
    next_user_pick_index = user_picks[0]

    # --- Determine approximate overall pick number (1-based) of your next pick
    next_overall_pick = next_user_pick_index + 1

    # --- Optional ADP proximity filter
    candidates_base = list(draft_state.available_players)
    if candidate_adp_margin > 0:
        adp_floor = max(1, next_overall_pick - candidate_adp_margin)
        candidates_base = [p for p in candidates_base if p.adp >= adp_floor]

    # --- Choose scoring method
    if use_vor:
        # compute replacement levels using current available pool
        n_teams = len(draft_state.teams)
        replacement_levels = compute_replacement_levels(
            candidates_base, starter_requirements, n_teams
        )
        player_scores = score_players_vor(candidates_base, replacement_levels, weight_adp=weight_adp)
    else:
        # fallback to simple projections + ADP blend
        player_scores = score_players(candidates_base, weight_adp=weight_adp)

    # Build candidate pool based on scoring
    # Score all available players and sort descending by score
    # --- Build candidate pool (dynamic-by-need, or balanced, or simple top-N)
    if candidate_pool_dynamic:
        # 1) compute weights from CURRENT roster need (user team)
        user_team = draft_state.teams[user_team_idx]
        need_weights = compute_position_need_weights(
            user_team,
            allow_bench_early=False,      # keep bench out of pool budgeting; we still show upside via scoring
            bench_threshold=1.0,
        )
        # Optional: tiny floor so TE isn't invisible when TE need is low-but-not-zero:
        hard_min = {}
        for pos, w in need_weights.items():
            if w > 0 and pos in ("TE", "QB", "DST", "K"):
                hard_min[pos] = 1  # ensure at least one slot IF there is any remaining need

        # 2) turn weights into integer caps that sum to candidate_pool_size
        caps = compute_caps_from_weights(candidate_pool_size, need_weights, hard_min=hard_min)

        # 3) pick top-by-score per position up to cap; reallocate leftovers if a position dries up
        by_pos = defaultdict(list)
        for p, sc in sorted(player_scores.items(), key=lambda x: x[1], reverse=True):
            by_pos[p.position.upper()].append(p)

        pool = []
        remaining_caps = caps.copy()

        # first pass: take up to cap for each pos
        for pos, cap in remaining_caps.items():
            take = by_pos.get(pos, [])[:cap]
            pool.extend(take)
            remaining_caps[pos] = max(0, cap - len(take))

        # If we didn't fill the pool (e.g., a pos lacked enough candidates), fill from global remainder
        if len(pool) < candidate_pool_size:
            already = set(pool)
            remainder = [p for p, _ in sorted(player_scores.items(), key=lambda x: x[1], reverse=True) if p not in already]
            need_more = candidate_pool_size - len(pool)
            pool.extend(remainder[:need_more])

        candidate_players = pool

    elif candidate_pool_balanced:
        default_caps = {"RB": 5, "WR": 5, "QB": 2, "TE": 2, "DST": 1, "K": 1}
        base_caps = per_position_caps or default_caps
        caps = base_caps  # fixed caps
        sorted_all = sorted(player_scores.items(), key=lambda x: x[1], reverse=True)
        pool, taken_per_pos = [], defaultdict(int)
        for p, _ in sorted_all:
            pos = p.position.upper()
            if taken_per_pos[pos] < caps.get(pos, 0):
                pool.append(p)
                taken_per_pos[pos] += 1
                if len(pool) >= candidate_pool_size:
                    break
        candidate_players = pool

    else:
        sorted_candidates = sorted(player_scores.items(), key=lambda x: x[1], reverse=True)
        candidate_players = [p for (p, _) in sorted_candidates[:candidate_pool_size]]

    results: List[tuple[Player, float]] = []

    # For each candidate, run simulations
    start = time.time()
    # for candidate in candidate_players:
    #     total_value = 0.0
    #     successful_runs = 0
    #     for _ in range(num_simulations):
    #         # Copy state for simulation
    #         sim_state = copy_draft_state(draft_state)
    #         # Simulate draft from current pick to end
    #         forced_candidate = False
    #         def same(a,b): return (a.name,a.team,a.position)==(b.name,b.team,b.position)

    #         while not sim_state.is_draft_over():
    #             team_idx = sim_state.get_current_team_index()

    #             if team_idx == user_team_idx:
    #                 if not forced_candidate:
    #                     # find this candidate's object in the sim copy
    #                     cand_sim = next((p for p in sim_state.available_players if same(p, candidate)), None)
    #                     if cand_sim is not None and sim_state.teams[user_team_idx].can_draft(cand_sim):
    #                         sim_state.make_pick(user_team_idx, cand_sim)
    #                         forced_candidate = True
    #                     else:
    #                         # candidate unavailable in this run -> bail early (no success counted)
    #                         sim_state.advance_pick()
    #                         break
    #                 else:
    #                     # later user picks with FLEX + BENCH gates
    #                     eligible = [p for p in sim_state.available_players
    #                                 if sim_state.teams[user_team_idx].can_draft(p)]
    #                     if eligible:
    #                         # Decide gates
    #                         flex_now, is_flex_only = compute_flex_decision(
    #                             sim_state.teams[user_team_idx],
    #                             eligible,
    #                             starter_requirements,
    #                             replacement_levels,
    #                             allow_flex_early=allow_flex_early,
    #                             flex_threshold=flex_threshold,
    #                         )
    #                         bench_now, is_bench_only = compute_bench_decision(
    #                             sim_state.teams[user_team_idx],
    #                             eligible,
    #                             starter_requirements,
    #                             replacement_levels,
    #                             allow_bench_early=allow_bench_early,
    #                             bench_threshold=bench_threshold,
    #                         )

    #                         # Base score (ADP blend)
    #                         scores = score_players(eligible, weight_adp=user_strategy_weight_adp)

    #                         # Apply nudges/penalties
    #                         for p in eligible:
    #                             # FLEX-only
    #                             if is_flex_only.get(p, False):
    #                                 scores[p] += 0.25 if flex_now else -1e6  # tune the penalty if you want softer behavior
    #                             # BENCH-only
    #                             if is_bench_only.get(p, False):
    #                                 scores[p] += 0.25 if bench_now else -1e6

    #                         ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    #                         top_pool = [p for p, _ in ranked[:user_strategy_top_n] if sim_state.teams[user_team_idx].can_draft(p)]
    #                         if top_pool:
    #                             sim_state.make_pick(user_team_idx, random.choice(top_pool))
    #                 sim_state.advance_pick()
    #             else:
    #                 opp_player = select_player_for_team(sim_state, team_idx, top_n=opponent_top_n)
    #                 if opp_player:
    #                     sim_state.make_pick(team_idx, opp_player)
    #                 sim_state.advance_pick()
    #         # After full draft (or break), evaluate if candidate was drafted
    #         if forced_candidate:
    #             value = evaluate_team_value(sim_state.teams[user_team_idx], starter_requirements)
    #             total_value += value
    #             successful_runs += 1
    #     # Calculate expected value across simulations (averaging over all runs)
    #     avg_value = (total_value / successful_runs) if successful_runs > 0 else 0.0
    #     p_available = successful_runs / num_simulations if num_simulations > 0 else 0.0
    #     ev_if_available = avg_value
    #     ev_unconditional = total_value / num_simulations if num_simulations > 0 else 0.0
    #     vor_now = player_vor(candidate, replacement_levels)
    #     adp = float(candidate.adp)
    #     adp_delta = (next_overall_pick - adp)

    #     slots_now = draft_state.teams[user_team_idx].slots_remaining
    #     pos = candidate.position.upper()
    #     if slots_now.get(pos, 0) > 0:
    #         fills_slot = f"{pos} starter"
    #     elif pos in {"RB","WR","TE"} and slots_now.get("FLEX", 0) > 0:
    #         fills_slot = "FLEX"
    #     else:
    #         fills_slot = "Bench"

    #     rationale = []
    #     rationale.append(f"+{vor_now:.1f} VOR vs {pos} replacement")
    #     rationale.append(f"{p_available*100:.0f}% chance available")
    #     if adp_delta >= 1:
    #         rationale.append(f"ADP value (+{adp_delta:.0f} picks)")
    #     elif adp_delta <= -1:
    #         rationale.append(f"Reach ({-adp_delta:.0f} picks)")
    #     rationale.append(f"Fills {fills_slot}")

    #     results.append(Recommendation(
    #         player=candidate,
    #         ev_if_available=ev_if_available,
    #         p_available=p_available,
    #         ev_unconditional=ev_unconditional,
    #         vor=vor_now,
    #         adp=adp,
    #         adp_delta=adp_delta,
    #         fills_slot=fills_slot,
    #         rationale=rationale
    #     ))
    
    #  prepare common arguments for parallel processing
    common_kwargs = dict(
        user_team_idx=user_team_idx,
        starter_requirements=starter_requirements,
        replacement_levels=replacement_levels,
        user_strategy_weight_adp=user_strategy_weight_adp,
        user_strategy_top_n=user_strategy_top_n,
        opponent_top_n=opponent_top_n,
        allow_flex_early=allow_flex_early,
        flex_threshold=flex_threshold,
        allow_bench_early=allow_bench_early,
        bench_threshold=bench_threshold,
    )
    results: List[Recommendation] = []

    with ProcessPoolExecutor(max_workers=SIM_WORKERS) as pool:
        fut_map = {
            pool.submit(
                evaluate_candidate_batch,
                draft_state,     # must be picklable; OK if your DraftState is pure-Python
                candidate,
                num_simulations=num_simulations,
                **common_kwargs,
            ): candidate
            for candidate in candidate_players
        }

        for fut in as_completed(fut_map):
            candidate = fut_map[fut]
            part = fut.result()  # {"total_value": ..., "successful_runs": ..., "num_simulations": ...}

            total_value = part["total_value"]
            successful_runs = part["successful_runs"]

            # compute per-candidate metrics (same math as before)
            avg_value = (total_value / successful_runs) if successful_runs > 0 else 0.0
            p_available = (successful_runs / num_simulations) if num_simulations > 0 else 0.0
            ev_if_available = avg_value
            ev_unconditional = (total_value / num_simulations) if num_simulations > 0 else 0.0
            vor_now = player_vor(candidate, replacement_levels)
            adp = float(candidate.adp)
            adp_delta = (next_overall_pick - adp)

            slots_now = draft_state.teams[user_team_idx].slots_remaining
            pos = candidate.position.upper()
            if slots_now.get(pos, 0) > 0:
                fills_slot = f"{pos} starter"
            elif pos in {"RB","WR","TE"} and slots_now.get("FLEX", 0) > 0:
                fills_slot = "FLEX"
            else:
                fills_slot = "Bench"

            rationale = []
            rationale.append(f"+{vor_now:.1f} VOR vs {pos} replacement")
            rationale.append(f"{p_available*100:.0f}% chance available")
            if adp_delta >= 1:
                rationale.append(f"ADP value (+{adp_delta:.0f} picks)")
            elif adp_delta <= -1:
                rationale.append(f"Reach ({-adp_delta:.0f} picks)")
            rationale.append(f"Fills {fills_slot}")

            results.append(Recommendation(
                player=candidate,
                ev_if_available=ev_if_available,
                p_available=p_available,
                ev_unconditional=ev_unconditional,
                vor=vor_now,
                adp=adp,
                adp_delta=adp_delta,
                fills_slot=fills_slot,
                rationale=rationale
            ))
    print("Simulation time:", time.time() - start)
    # Sort results by expected value descending
    results.sort(key=lambda r: (r.ev_unconditional, r.ev_if_available, r.vor), reverse=True)
    return results


def copy_draft_state(state: DraftState) -> DraftState:
    """Return a deep copy of a ``DraftState`` instance.

    ``copy.deepcopy`` is sufficient here because the ``DraftState`` object
    contains only built-in types, dataclasses, and lists.  The copy is
    independent of the original; mutations to the copy will not affect
    the original state.

    Parameters
    ----------
    state : DraftState
        The draft state to clone.

    Returns
    -------
    DraftState
        A deep copy of the provided state.
    """
    return copy.deepcopy(state)


def select_player_for_team(state: DraftState, team_index: int, *, top_n: int = 5) -> Optional[Player]:
    """Choose an available player for the specified team based on ADP.

    The selection process sorts the available players by ADP (ascending),
    filters out any players that the team cannot draft (using
    ``TeamRoster.can_draft``), and then selects one of the top ``top_n``
    candidates at random.  This introduces a small amount of randomness
    into the simulations so that opponents do not always pick the exact
    same player in every run.

    Parameters
    ----------
    state : DraftState
        The current draft state.
    team_index : int
        Index of the team making the pick.
    top_n : int, optional
        Number of highest-ranked (by ADP) candidates to consider for
        random selection.  Defaults to 5.

    Returns
    -------
    Optional[Player]
        The selected player, or ``None`` if no eligible players remain.
    """
    # Get players sorted by ADP (ascending); if ADP ties, prefer higher projected points
    players_sorted: List[Player] = sorted(
        state.available_players,
        key=lambda p: (p.adp, -p.fpts)
    )
    # Filter to players the team can draft
    candidates: List[Player] = [p for p in players_sorted if state.teams[team_index].can_draft(p)]
    if not candidates:
        return None
    # Determine the pool size for randomness
    pool_size = min(top_n, len(candidates))
    # Randomly choose from the top pool_size players
    return random.choice(candidates[:pool_size])


def score_players(players: Sequence[Player], *, weight_adp: float = -0.1) -> Dict[Player, float]:
    """Assign a simple score to players based on their projections and ADP.

    The score is computed as ``fpts + (weight_adp * adp)``.  A negative
    ``weight_adp`` value increases the score of players with earlier ADP
    (lower numerical value).  Adjust the weight to tune the balance
    between projected points and draft capital.

    Parameters
    ----------
    players : Sequence[Player]
        Players to score.
    weight_adp : float, optional
        Weight applied to ADP.  Default is -0.1, meaning each position of
        ADP reduces the score by 0.1.

    Returns
    -------
    Dict[Player, float]
        Mapping from player to their computed score.
    """
    return {p: p.fpts + weight_adp * p.adp for p in players}
