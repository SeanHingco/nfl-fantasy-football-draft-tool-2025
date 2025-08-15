# server/draft_manager.py
import uuid
from typing import Dict, Optional
from draft_state import DraftState
from data_loader import load_players_as_objects

class DraftManager:
    """
    Holds multiple DraftState objects in memory, keyed by session_id.
    Later you can swap this for Redis/DB if you want persistence.
    """
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.players = load_players_as_objects(self.data_dir)  # load once at boot
        self._sessions: Dict[str, DraftState] = {}

    def create_session(
        self,
        league_size: int,
        user_team_index: int,
        bench_spots: int,
        starter_requirements: Dict[str, int],
    ) -> str:
        session_id = str(uuid.uuid4())
        state = DraftState(
            self.players,
            starter_requirements,
            bench_spots,
            league_size,
            user_team_index,
        )
        self._sessions[session_id] = state
        return session_id

    def get(self, session_id: str) -> Optional[DraftState]:
        return self._sessions.get(session_id)
