"""
draft_state.py
==============

This module defines the ``DraftState`` class, which encapsulates the
current state of a fantasy football draft.  It keeps track of the
available players, team rosters, draft order, and pick progression.  A
``DraftState`` instance can be updated as real picks occur and later
passed into a Monte Carlo simulator to generate recommendations based on
what remains on the board.

The class does **not** implement the Monte Carlo logic itself—that
functionality will live in a separate simulator.  Instead, it focuses on
maintaining and updating the state consistently as picks are made.

Usage
-----

```python
from data_loader import load_players_as_objects
from draft_models import TeamRoster
from draft_state import DraftState
from draft_utils import calculate_rounds

# Load players and initialise the draft state
players = load_players_as_objects(data_dir)
starter_req = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "DST": 1, "K": 1}
bench = 7
n_teams = 10
user_team = 3  # The user drafts from team index 3

draft_state = DraftState(players, starter_req, bench, n_teams, user_team)

# Make a pick (e.g., remove a drafted player)
draft_state.make_pick(team_index=1, player=some_player)

# Advance to the next pick
draft_state.advance_pick()
```
"""

from __future__ import annotations

from typing import List, Optional

from draft_models import Player, TeamRoster
from draft_utils import calculate_rounds, generate_snake_order


class DraftState:
    """Maintain the state of an ongoing fantasy football draft.

    Parameters
    ----------
    players : List[Player]
        The full list of available players prior to the draft starting.  This
        list will be copied internally and mutated as players are drafted.
    starter_requirements : dict
        Mapping of position codes to the number of required starters for that
        position.  For example, ``{"QB": 1, "RB": 2, "WR": 2, "TE": 1,
        "FLEX": 1, "DST": 1, "K": 1}``.
    bench_spots : int
        The number of bench spots each team has.
    n_teams : int
        Total number of teams in the league.
    user_team_index : int
        The index of the user's team (0-based).  Used later when computing
        recommendations.
    """

    def __init__(
        self,
        players: List[Player],
        starter_requirements: dict[str, int],
        bench_spots: int,
        n_teams: int,
        user_team_index: int,
    ) -> None:
        # Copy the players list so that modifications don't affect the caller
        self._original_players: List[Player] = list(players)
        self.available_players: List[Player] = list(players)
        self.n_teams = n_teams
        self.user_team_index = user_team_index
        self.starter_requirements = starter_requirements.copy()
        self.bench_spots = bench_spots
        self.picks: list[tuple[int, Player]] = []

        # Compute number of rounds based on roster requirements
        self.n_rounds = calculate_rounds(self.starter_requirements, bench_spots)
        # Generate the draft order (list of team indices)
        self.draft_order: List[int] = generate_snake_order(n_teams, self.n_rounds)
        # Create a TeamRoster for each team
        self.teams: List[TeamRoster] = [
            TeamRoster(team_id=i, starter_requirements=self.starter_requirements, bench_spots=self.bench_spots)
            for i in range(n_teams)
        ]
        # Track the overall pick index (0-based into draft_order)
        self.current_pick_index: int = 0

    def is_draft_over(self) -> bool:
        """Return True if all picks in the draft have been made."""
        return self.current_pick_index >= len(self.draft_order)

    def get_current_team_index(self) -> Optional[int]:
        """Return the index of the team whose turn it is to pick.

        Returns None if the draft has completed.
        """
        if self.is_draft_over():
            return None
        return self.draft_order[self.current_pick_index]

    def remove_player_from_available(self, player: Player) -> None:
        """Remove a player from the available pool if present.

        Parameters
        ----------
        player : Player
            The player to remove.  Comparison is by object identity or
            equality.
        """
        try:
            self.available_players.remove(player)
        except ValueError:
            # Player not found; ignore
            pass

    def make_pick(self, team_index: int, player: Player) -> bool:
        """Record that a team has drafted a given player.

        This method updates both the team's roster and the available player
        pool.  If the pick is invalid (e.g., the team cannot draft the
        player), it returns False and does nothing else.  Otherwise, it
        returns True.

        Note: This method does **not** advance the current pick index.
        Call :meth:`advance_pick` separately after making a pick.
        """
        # Validate team index
        if not (0 <= team_index < self.n_teams):
            return False
        team = self.teams[team_index]
        # Check if team can draft this player
        if not team.can_draft(player):
            return False
        # Draft player and remove from available pool
        drafted = team.draft_player(player)
        if drafted:
            self.remove_player_from_available(player)
            self.picks.append((team_index, player))
            return True
        return False

    def advance_pick(self) -> None:
        """Advance to the next pick in the draft order.

        If the draft has already completed, this method does nothing.
        """
        if not self.is_draft_over():
            self.current_pick_index += 1

    def get_pick_number_for_team(self, team_index: int) -> List[int]:
        """Return a list of overall pick indices where a given team picks.

        Parameters
        ----------
        team_index : int
            The team index to search for.

        Returns
        -------
        List[int]
            A list of indices into the draft order representing the pick
            positions for the specified team.
        """
        return [i for i, t in enumerate(self.draft_order) if t == team_index]

    def reset(self) -> None:
        """Reset the draft state to its initial configuration.

        This method restores all teams' rosters, the available player pool,
        and the current pick index as they were at initialisation.  It can be
        used prior to running a new simulation without instantiating a new
        ``DraftState``.
        """
        """Reset the draft state to its initial configuration.

        This method restores all teams' rosters, the available player pool,
        and the current pick index as they were at initialisation.  It can be
        used prior to running a new simulation without instantiating a new
        :class:`DraftState`.
        """
        # Restore the available players to the original list
        self.available_players = list(self._original_players)
        # Reset each team's roster and slot counters
        for team in self.teams:
            team.__init__(team.team_id, self.starter_requirements, self.bench_spots)
        # Reset pick index
        self.current_pick_index = 0

    def undo_last_pick(self) -> bool:
        if not self.picks:
            return False

        # move pointer back one slot
        if self.current_pick_index > 0:
            self.current_pick_index -= 1

        team_idx, player = self.picks.pop()

        # reverse roster effects
        self.teams[team_idx].undraft_player(player)

        # restore to available pool (optionally re-sort)
        self.available_players.append(player)
        # self.available_players.sort(key=lambda p: (p.adp, -p.fpts))

        return True