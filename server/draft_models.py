"""
draft_models.py
===============

Definitions of core domain objects used by the draft simulator.  These
classes encapsulate player data and team roster state, providing a
structured and type-safe way to represent participants in the draft.

The design uses lightweight dataclasses for immutable value objects
(`Player`) and regular classes for mutable state (`TeamRoster`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Recommendation:
    player: Player
    ev_if_available: float        # avg team value in runs where you actually drafted this player
    p_available: float            # successful_runs / num_simulations
    ev_unconditional: float       # risk-adjusted = ev_if_available * p_available (same as total_value / num_sims)
    vor: float                    # current VOR snapshot at evaluation time
    adp: float
    adp_delta: float              # (next_overall_pick - adp) -> how many picks of “value” you’re getting
    fills_slot: str               # "QB starter" / "WR starter" / "FLEX" / "Bench"
    rationale: List[str]          # short bullet strings for UI

@dataclass(frozen=True)
class Player:
    """Represents a fantasy football player.

    Attributes
    ----------
    name : str
        The player's full name (e.g., "Ja'Marr Chase").
    team : str
        The NFL team abbreviation (e.g., "CIN").
    position : str
        The player's base position ("QB", "RB", "WR", "TE", "DST", "K").
    fpts : float
        Projected fantasy points.  Used as the value metric in the simulation.
    adp : float
        Average draft position.  Lower numbers indicate higher priority.
    """

    name: str
    team: str
    position: str
    fpts: float
    adp: float


class TeamRoster:
    """Represents a drafting team's roster and positional needs.

    The roster is initialised with required starter slots for each position
    (e.g., 1 QB, 2 RB, etc.) and a number of bench spots.  Players can
    be drafted into either the starting position (if a slot remains) or
    onto the bench.  The class provides methods to check whether a
    particular player can be drafted and to update the roster state when
    a player is drafted.

    Parameters
    ----------
    team_id : int
        Identifier for the team (e.g., index in the draft order).
    starter_requirements : Dict[str, int]
        Mapping of position codes to the number of required starter slots
        for that position.  For example, ``{"QB": 1, "RB": 2, "WR": 2, "TE": 1,
        "FLEX": 1, "DST": 1, "K": 1}``.
    bench_spots : int
        Number of bench slots available.  Bench slots can be filled with
        any offensive position (QB, RB, WR, TE) unless league rules dictate
        otherwise.  Defenses and kickers are typically not benched.
    """

    def __init__(self, team_id: int, starter_requirements: Dict[str, int], bench_spots: int) -> None:
        self.team_id = team_id
        # Track how many starter slots remain for each position
        self._slots_remaining: Dict[str, int] = starter_requirements.copy()
        # Number of bench spots left to fill
        self._bench_remaining: int = bench_spots
        # Roster lists by position; bench uses key "BENCH"
        self.roster: Dict[str, List[Player]] = {pos: [] for pos in starter_requirements}
        self.roster["BENCH"] = []

    def can_draft(self, player: Player) -> bool:
        """Return True if this team can draft the given player.

        The logic follows standard roster rules:

        - If the player’s position still has starter slots available, the
          team can draft the player into that position.
        - Otherwise, if there are bench spots remaining and the player's
          position is eligible for bench (typically QB, RB, WR, TE), the team
          can draft the player onto the bench.
        - Teams do not typically bench DST or K unless bench rules allow
          it; those positions are ignored for bench drafting by default.

        Parameters
        ----------
        player : Player
            The player to evaluate.

        Returns
        -------
        bool
            True if the team can add this player, False otherwise.
        """
        pos = player.position.upper()
        # Starter slot for the player's base position is available
        if self._slots_remaining.get(pos, 0) > 0:
            return True
        # FLEX slot can be filled by RB, WR or TE
        if pos in {"RB", "WR", "TE"} and self._slots_remaining.get("FLEX", 0) > 0:
            return True
        # Otherwise, check bench eligibility (skill positions only)
        if self._bench_remaining > 0 and pos in {"QB", "RB", "WR", "TE"}:
            return True
        return False

    def draft_player(self, player: Player) -> bool:
        """Attempt to add a player to the roster.

        Places the player into a starting slot if available for their
        position; otherwise places them on the bench (if eligible).  Returns
        True if the player was successfully added, False if the roster is
        full for that position and no bench slots remain.
        """
        if not self.can_draft(player):
            return False
        pos = player.position.upper()
        # If there is a starter slot for the base position, fill it
        if self._slots_remaining.get(pos, 0) > 0:
            self.roster.setdefault(pos, []).append(player)
            self._slots_remaining[pos] -= 1
            return True
        # Otherwise, if flex is available and player is RB/WR/TE, fill flex
        if pos in {"RB", "WR", "TE"} and self._slots_remaining.get("FLEX", 0) > 0:
            self.roster.setdefault("FLEX", []).append(player)
            self._slots_remaining["FLEX"] -= 1
            return True
        # Otherwise, fill bench if eligible (skill positions only)
        if self._bench_remaining > 0 and pos in {"QB", "RB", "WR", "TE"}:
            self.roster["BENCH"].append(player)
            self._bench_remaining -= 1
            return True
        return False
    
    def undraft_player(self, player: Player) -> bool:
        pos = player.position.upper()

        if player in self.roster.get(pos, []):
            self.roster[pos].remove(player)
            self._slots_remaining[pos] += 1
            return True

        if player in self.roster.get("FLEX", []):
            self.roster["FLEX"].remove(player)
            self._slots_remaining["FLEX"] += 1
            return True

        if player in self.roster.get("BENCH", []):
            self.roster["BENCH"].remove(player)
            self._bench_remaining += 1
            return True

        return False

    @property
    def slots_remaining(self) -> Dict[str, int]:
        """Return a copy of the remaining starter slots."""
        return self._slots_remaining.copy()

    @property
    def bench_remaining(self) -> int:
        """Return the number of bench spots left."""
        return self._bench_remaining
