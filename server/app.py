# server/app.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import os, logging, uuid, time

app = FastAPI(title="Fantasy Draft Recommender", version="0.1.0")
logger = logging.getLogger("uvicorn.error")
logger.info("CPU cores detected: %s", os.cpu_count())

# allow your Vite dev server (default: http://localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:8080", "http://localhost:5173", "https://fantasy-ui.onrender.com"],
    allow_credentials=False,
    allow_methods=["*"],   # lets the preflight say POST is allowed
    allow_headers=["*"],   # lets the preflight say your headers are allowed
)


@app.middleware("http")
async def log_requests(request, call_next):
    rid = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    try:
        # read body once (small payload), to log session_id/num_simulations if present
        body = await request.body()
        logger.info("REQ %s %s %s body=%s", rid, request.method, request.url.path, body[:300])
        response = await call_next(request)
        return response
    finally:
        dur = time.perf_counter() - start
        logger.info("RES %s %s %.2fs %s", rid, request.url.path, dur, getattr(response, "status_code", "?"))

# server/app.py
from fastapi import FastAPI, HTTPException
from api_models import (
    InitStateRequest, InitStateResponse,
    PickRequest, StateSummary, PlayerOut,
    RecsRequest, RecommendationOut, RecsEnvelope,
)
from draft_manager import DraftManager
from draft_simulator import recommend_players, Recommendation, select_player_for_team, evaluate_team_value
import math
from time import perf_counter
import re
from draft_state import DraftState
from data_loader import load_players_as_objects

SESSIONS: dict[str, DraftState] = {}
DATA_DIR = "data"

def _norm(s: str) -> str:
    return re.sub(r"[’`]", "'", s).strip().lower()

def _safe_str(x: object, default: str = "") -> str:
    # turn None/NaN into a safe empty string
    if x is None:
        return default
    if isinstance(x, float) and math.isnan(x):
        return default
    s = str(x)
    return "" if s.lower() == "nan" else s

def to_player_out(p) -> PlayerOut:
    return PlayerOut(
        name=_safe_str(p.name),
        team=_safe_str(p.team),
        position=_safe_str(p.position),
        fpts=float(p.fpts),
        adp=float(p.adp),
    )

def build_my_roster(state) -> dict[str, list[PlayerOut]]:
    me = state.teams[state.user_team_index]
    return {slot: [to_player_out(p) for p in lst] for slot, lst in me.roster.items()}

def to_rec_out(r: Recommendation) -> RecommendationOut:
    return RecommendationOut(
        player=to_player_out(r.player),
        ev_if_available=r.ev_if_available,
        p_available=r.p_available,
        ev_unconditional=r.ev_unconditional,
        vor=r.vor,
        adp=r.adp,
        adp_delta=r.adp_delta,
        fills_slot=r.fills_slot,
        rationale=r.rationale,
    )

@app.get("/health")
def health():
    return {"ok": True}

manager = DraftManager(data_dir="data")

@app.post("/state/init", response_model=InitStateResponse)
def init_state(req: InitStateRequest):
    if req.user_team_index >= req.league_size:
        raise HTTPException(status_code=400, detail="user_team_index >= league_size")

    sid = manager.create_session(
        league_size=req.league_size,
        user_team_index=req.user_team_index,
        bench_spots=req.bench_spots,
        starter_requirements=req.starter_requirements,
    )
    state = manager.get(sid)
    return InitStateResponse(
        session_id=sid,
        current_pick_index=state.current_pick_index,
        draft_order=state.draft_order,
    )

@app.post("/state/pick", response_model=StateSummary)
def make_pick(req: PickRequest):
    state = manager.get(req.session_id)
    if not state:
        raise HTTPException(status_code=404, detail="session not found")

    wanted = _norm(req.player_name)
    player = next((p for p in state.available_players if _norm(p.name) == wanted), None)
    if not player:
        raise HTTPException(status_code=400, detail="player not available")
    if not state.make_pick(req.team_index, player):
        raise HTTPException(status_code=400, detail="team cannot draft this player")
    state.advance_pick()

    user_team = state.teams[state.user_team_index]
    return StateSummary(
        session_id=req.session_id,
        current_pick_index=state.current_pick_index,
        on_the_clock_team=state.get_current_team_index(),
        slots_remaining=user_team.slots_remaining,
        my_roster=build_my_roster(state),             # <-- add
    )

@app.post("/recs", response_model=list[RecommendationOut])
def get_recommendations(req: RecsRequest):
    state = manager.get(req.session_id)
    if not state:
        raise HTTPException(status_code=404, detail="session not found")

    recs = recommend_players(
        state,
        starter_requirements=state.starter_requirements,
        candidate_pool_size=req.candidate_pool_size,
        num_simulations=req.num_simulations,
        use_vor=req.use_vor,
        weight_adp=req.weight_adp,
        candidate_adp_margin=req.candidate_adp_margin,
        candidate_pool_balanced=req.candidate_pool_balanced,
        candidate_pool_dynamic=req.candidate_pool_dynamic,
        per_position_caps=req.per_position_caps,
        opponent_top_n=req.opponent_top_n,
        user_strategy_weight_adp=req.user_strategy_weight_adp,
        user_strategy_top_n=req.user_strategy_top_n,
        allow_flex_early=req.allow_flex_early,
        flex_threshold=req.flex_threshold,
        allow_bench_early=req.allow_bench_early,
        bench_threshold=req.bench_threshold,
    )
    return [to_rec_out(r) for r in recs]

@app.post("/recs2", response_model=RecsEnvelope)
def get_recommendations2(req: RecsRequest):
    state = manager.get(req.session_id)
    if not state:
        raise HTTPException(status_code=404, detail="session not found")
    t0 = perf_counter()
    recs = recommend_players(
        state,
        starter_requirements=state.starter_requirements,
        candidate_pool_size=req.candidate_pool_size,
        num_simulations=req.num_simulations,
        use_vor=req.use_vor,
        weight_adp=req.weight_adp,
        candidate_adp_margin=req.candidate_adp_margin,
        candidate_pool_balanced=req.candidate_pool_balanced,
        candidate_pool_dynamic=req.candidate_pool_dynamic,
        per_position_caps=req.per_position_caps,
        opponent_top_n=req.opponent_top_n,
        user_strategy_weight_adp=req.user_strategy_weight_adp,
        user_strategy_top_n=req.user_strategy_top_n,
        allow_flex_early=req.allow_flex_early,
        flex_threshold=req.flex_threshold,
        allow_bench_early=req.allow_bench_early,
        bench_threshold=req.bench_threshold,
    )
    dt = (perf_counter() - t0) * 1000.0
    return RecsEnvelope(
        duration_ms=dt,
        params=req,
        results=[to_rec_out(r) for r in recs]
    )

@app.get("/state/summary", response_model=StateSummary)
def summary(session_id: str):
    state = manager.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="session not found")
    user_team = state.teams[state.user_team_index]
    return StateSummary(
        session_id=session_id,
        current_pick_index=state.current_pick_index,
        on_the_clock_team=state.get_current_team_index(),
        slots_remaining=user_team.slots_remaining,
        my_roster=build_my_roster(state),             # <-- add
    )

# @app.get("/players")
# def list_players(
#     session_id: str,
#     position: str | None = Query(None, description="RB/WR/QB/TE/DST/K"),
#     q: str | None = Query(None, description="Search on name or team (case-insensitive)"),
#     limit: int = Query(50, ge=1, le=200),
# ):
#     state = manager.get(session_id)
#     if state is None:
#         raise HTTPException(status_code=404, detail="session not found")

#     players = list(state.available_players)

#     # position filter
#     if position:
#         pos = position.upper()
#         players = [p for p in players if p.position.upper() == pos]

#     # text filter
#     if q:
#         needle = q.casefold()
#         def to_str(x):
#             try:
#                 return (x or "").casefold()
#             except Exception:
#                 return ""
#         players = [p for p in players if needle in to_str(p.name) or needle in to_str(p.team)]

#     # sort & limit (ADP asc, then FPTS desc)
#     players.sort(key=lambda p: (p.adp, -p.fpts))
#     players = players[:limit]

#     # map to your response model (PlayerOut) or plain dicts:
#     def safe_str(x):
#         return "" if x is None or (isinstance(x, float) and x != x) else str(x)

#     return [
#         {
#             "name": safe_str(p.name),
#             "team": safe_str(p.team),
#             "position": p.position,
#             "fpts": float(p.fpts),
#             "adp": float(p.adp),
#         }
#         for p in players
#     ]


@app.get("/draftboard")
def draftboard(session_id: str):
    state = manager.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="session not found")
    # expose a simple list of picks done so far
    picks = getattr(state, "picks", [])  # adjust if you track differently
    # each pick should include: overall_index, team_index, player (name/pos/team)
    return {"picks": [
        {
            "overall_index": i,
            "team_index": t,
            "player": {
                "name": p.name, "team": _safe_str(p.team), "position": p.position
            }
        }
        for i, (t, p) in enumerate(picks)
    ]}

@app.post("/state/undo", response_model=StateSummary)
def undo(session_id: str):
    state = manager.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="session not found")
    ok = state.undo_last_pick()
    if not ok:
        raise HTTPException(status_code=400, detail="nothing to undo")
    user_team = state.teams[state.user_team_index]
    return StateSummary(
        session_id=session_id,
        current_pick_index=state.current_pick_index,
        on_the_clock_team=state.get_current_team_index(),
        slots_remaining=user_team.slots_remaining,
        my_roster=build_my_roster(state),             # <-- add
    )

@app.post("/state/autopick", response_model=StateSummary)
def autopick(session_id: str, top_n: int = 5):
    state = manager.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="session not found")
    team_idx = state.get_current_team_index()
    p = select_player_for_team(state, team_idx, top_n=top_n)
    if not p:
        raise HTTPException(status_code=400, detail="no eligible players")
    if not state.make_pick(team_idx, p):
        raise HTTPException(status_code=400, detail="cannot draft player")
    state.advance_pick()

    user_team = state.teams[state.user_team_index]
    return StateSummary(
        session_id=session_id,
        current_pick_index=state.current_pick_index,
        on_the_clock_team=state.get_current_team_index(),
        slots_remaining=user_team.slots_remaining,
        my_roster=build_my_roster(state),             # <-- add
    )

@app.get("/players", response_model=list[PlayerOut])
def list_players(session_id: str, position: str | None = None, q: str | None = None, limit: int = 50):
    state = manager.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="session not found")

    pool = state.available_players
    if position:
        pool = [p for p in pool if p.position.upper() == position.upper()]
    if q:
        needle = q.strip().lower()
        pool = [p for p in pool if needle in p.name.lower()]

    pool = sorted(pool, key=lambda p: (p.adp, -p.fpts))[:max(1, min(limit, 200))]
    return [to_player_out(p) for p in pool]
