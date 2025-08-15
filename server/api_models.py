from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class InitStateRequest(BaseModel):
    league_size: int = Field(..., ge=2, le=16)
    user_team_index: int = Field(..., ge=0)
    bench_spots: int = Field(7, ge=0)
    starter_requirements: Dict[str, int]

class InitStateResponse(BaseModel):
    session_id: str
    current_pick_index: int
    draft_order: List[int]

class PickRequest(BaseModel):
    session_id: str
    team_index: int = Field(..., ge=0)
    player_name: str

class StateSummary(BaseModel):
    session_id: str
    current_pick_index: int
    on_the_clock_team: int
    slots_remaining: Dict[str, int]

class PlayerOut(BaseModel):
    name: str
    team: str
    position: str
    fpts: float
    adp: float

class RecsRequest(BaseModel):
    session_id: str
    candidate_pool_size: int = Field(12, ge=3, le=50)
    num_simulations: int = Field(150, ge=10, le=1000)
    use_vor: bool = True
    weight_adp: float = -1.2
    candidate_adp_margin: int = 4
    candidate_pool_balanced: bool = False
    candidate_pool_dynamic: bool = True
    per_position_caps: Optional[Dict[str, int]] = None
    opponent_top_n: int = Field(5, ge=1, le=20)
    user_strategy_weight_adp: float = -0.5
    user_strategy_top_n: int = Field(5, ge=1, le=20)
    allow_flex_early: bool = True
    flex_threshold: float = 0.7
    allow_bench_early: bool = False
    bench_threshold: float = 1.0

class RecommendationOut(BaseModel):
    player: PlayerOut
    ev_if_available: float
    p_available: float
    ev_unconditional: float
    vor: float
    adp: float
    adp_delta: float
    fills_slot: str
    rationale: List[str]

class RecsEnvelope(BaseModel):
    duration_ms: float
    params: RecsRequest
    results: list[RecommendationOut]

class SummaryOut(BaseModel):
    session_id: str
    current_pick_index: int
    on_the_clock_team: int
    slots_remaining: Dict[str, int]
    my_roster: Dict[str, List[PlayerOut]]
