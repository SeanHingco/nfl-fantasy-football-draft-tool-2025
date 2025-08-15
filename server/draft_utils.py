"""
draft_utils.py
==============

Helper functions related to draft mechanics, such as generating a
snake-draft order and computing the number of rounds required for a
given roster configuration.
"""

from __future__ import annotations

from typing import Dict, List


def calculate_rounds(starter_requirements: Dict[str, int], bench_spots: int) -> int:
    """Compute the total number of draft rounds given roster needs.

    Parameters
    ----------
    starter_requirements : Dict[str, int]
        Mapping of position codes to the number of required starters for that position.
    bench_spots : int
        Number of bench slots available.

    Returns
    -------
    int
        Total number of draft rounds (players) each team will select.
    """
    return sum(starter_requirements.values()) + bench_spots


def generate_snake_order(n_teams: int, n_rounds: int) -> List[int]:
    """Generate a pick order for a snake draft.

    In a snake draft, the order of picks reverses every round.  For example,
    with four teams and three rounds, the pick order (by team index) is
    ``[0, 1, 2, 3,  3, 2, 1, 0,  0, 1, 2, 3]``.  This helper produces
    such an order for an arbitrary number of teams and rounds.

    Parameters
    ----------
    n_teams : int
        Number of teams in the league.
    n_rounds : int
        Number of rounds in the draft (players each team will select).

    Returns
    -------
    List[int]
        A list of length ``n_teams * n_rounds`` where each entry is the
        zero-based index of the team whose turn it is to pick at that slot.
    """
    order: List[int] = []
    for round_num in range(n_rounds):
        if round_num % 2 == 0:
            order.extend(range(n_teams))
        else:
            order.extend(reversed(range(n_teams)))
    return order
