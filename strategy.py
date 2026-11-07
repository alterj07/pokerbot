"""
strategy.py - a Tit-for-Tat strategy for heads-up Short Deck PLO5.

Tit-for-Tat, translated to poker:
    1. COOPERATE first: play a solid, equity-based baseline (the Params defaults).
    2. RETALIATE: when the opponent deviates from "polite" play, shift the
       thresholds toward the counter-strategy, in proportion to the deviation
       measured by OpponentModel (opponent_model.py):
           hyper-aggressive / bluffy  -> call wider, raise thinner for value, trap
                                         monsters by check-calling, stop bluffing
           tight / folds too much     -> open and steal more, c-bet/bluff more,
                                         bet thinner for value
           calling station            -> never bluff, value-bet thinner
           passive                    -> respect their rare bets
    3. FORGIVE: the model's statistics decay, so when the opponent returns to
       baseline the adjustments fade and we are back to step 1.

"Generous" Tit-for-Tat: poker is a noisy game, so deviations smaller than
DEAD_ZONE are ignored. Strict Tit-for-Tat retaliates against noise and loses
value; the generous version only reacts to deviations that are clearly real.

Entry points used by agent.py:
    fresh_memory(config)        -> dict stored as the half's memory
    ensure_memory(memory)       -> fills any missing keys (never leaves the bot stuck)
    decide(obs, memory)         -> Action (always legal)
    hand_ended(result, memory)  -> None
"""

from __future__ import annotations

import os
import random
import sys
import time
from dataclasses import dataclass

import equity
import preflop
from evaluator import (FLUSH, FOUR_OF_A_KIND, FULL_HOUSE, HIGH_CARD, ONE_PAIR, STRAIGHT,
                       STRAIGHT_FLUSH, THREE_OF_A_KIND, TWO_PAIR, category_of, omaha_score)
from game import Action, AgentConfig, HandResult, Observation, Street
from opponent_model import OpponentModel


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() not in ("", "0", "false", "no")


# ----- switches (read once at import; all default to the tournament setting) --
TFT_DISABLED = _flag("TFT_DISABLED", "0")      # pure baseline, for A/B tests
RANGE_FILTER = _flag("RANGE_FILTER", "1")      # range-aware opponent sampling
DEBUG = _flag("TFT_DEBUG", "0")                # print the opponent model every 100 hands
STYLE = os.environ.get("STYLE", "static").strip().lower()   # static | tight | loose (sparring only)
if STYLE not in ("static", "tight", "loose"):
    STYLE = "static"
# TIGHTNESS shifts every baseline threshold: +1 = the "tight" style, -1 = "loose", 0 = Params defaults.
# Sparring/tuning only; the tournament bot uses the defaults below (already tuned with this knob).
_TIGHT_DEFAULT = {"static": 0.0, "tight": 1.0, "loose": -1.0}[STYLE]

# ----- time budget -------------------------------------------------------------
TIME_TARGET = 5.0          # seconds of the 8 s bank we allow ourselves per half (judge-charged time runs ~0.3 s above this)
EMERGENCY_RESERVE = 0.3    # below this, stop simulating and use cheap heuristics
EST_PER_HAND = 1.8         # expected Monte Carlo estimates per hand (one per street seen)
MIN_SAMPLES = 30
MAX_SAMPLES = 400
MAX_REQUEST_SECONDS = 0.25  # never plan to spend more than this on one decision (limit is 1 s)
FALLBACK_COST = 60e-6

try:
    COST_PER_SAMPLE = equity.measure_cost_per_sample()   # measured on this machine at startup
    if not (COST_PER_SAMPLE > 0):
        COST_PER_SAMPLE = FALLBACK_COST
except Exception:
    COST_PER_SAMPLE = FALLBACK_COST

# ----- Tit-for-Tat tuning (env overrides exist only for local experiments) --------
def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


TIGHTNESS = _env_float("TIGHTNESS", _TIGHT_DEFAULT)
DEAD_ZONE = _env_float("TFT_DEAD_ZONE", 0.15)        # deviations smaller than this are forgiven outright
ALL_IN_MARGIN = _env_float("TFT_ALL_IN_MARGIN", 0.04)  # equity above pot odds required to call off the stack
COMMIT_FRACTION = 0.6      # a raise adding >= this share of our stack is a "stack-off"
COMMIT_WINDOW = _env_float("TFT_COMMIT_WINDOW", 0.0)   # ... downgraded if equity is within this of the threshold (0 = off: it cost ~2 bb/deal vs loose callers)

RAISES = (Action.RAISE_HALF, Action.RAISE_POT)


@dataclass
class Params:
    """All thresholds of the baseline. Deviations move them; defaults = cooperate."""
    # preflop (percentile of our 5 cards among all starting hands, 1.0 = best)
    # Tuned 2026-11-07 with the TIGHTNESS ladder (see STRATEGY.md): this is one step tighter
    # than the first version, which lost ~0.7 bb/deal to its own tighter twin.
    open_raise: float = 0.50     # button opens with a pot raise above this
    open_limp: float = 0.20      # button limps above this, folds below
    bb_iso: float = 0.80         # big blind raises a limp above this
    threebet: float = 0.90
    call_vs_open: float = 0.35
    fourbet: float = 0.96
    call_vs_3bet: float = 0.65
    call_vs_4bet: float = 0.88
    # postflop (Monte Carlo equity vs a random hand; real ranges are stronger, hence the margins)
    bet_pot: float = 0.80
    bet_half: float = 0.64
    raise_pot: float = 0.86
    raise_half: float = 0.74
    call_margin: float = 0.08    # equity needed above pot odds to call
    river_margin: float = 0.04   # extra margin on the river (no more cards to come)
    bluff_rate: float = 0.05     # chance to semi-bluff a half-pot bet with a weak hand
    trap_prob: float = 0.0       # chance to check-call a monster out of position
    maniac: float = 0.0          # softened aggression deviation when positive (0..1), informational


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def base_params() -> Params:
    """Baseline, shifted by TIGHTNESS (sparring partners / tuning only; tournament = 0)."""
    p = Params()
    s = TIGHTNESS
    if s == 0.0:
        return p
    for name in ("open_raise", "open_limp", "bb_iso", "call_vs_open", "call_vs_3bet"):
        setattr(p, name, _clamp(getattr(p, name) + 0.10 * s, 0.0, 0.99))
    for name in ("bet_pot", "bet_half", "raise_pot", "raise_half"):
        setattr(p, name, _clamp(getattr(p, name) + 0.08 * s, 0.0, 0.99))
    p.call_margin = _clamp(p.call_margin + 0.05 * s, -0.1, 0.3)
    p.bluff_rate = _clamp(p.bluff_rate - 0.07 * s, 0.0, 0.5)
    return p


RESPONSE_POWER = _env_float("TFT_RESPONSE_POWER", 1.0)   # 1 = linear (best in 5-seed sparring), 2 = quadratic


def soften(d: float) -> float:
    """Generous Tit-for-Tat: ignore deviations inside the dead zone, then respond with a
    power curve: a moderately different (but competent) opponent gets a small adjustment,
    an extreme one (maniac, calling station) gets the full counter-strategy."""
    mag = abs(d)
    if mag <= DEAD_ZONE:
        return 0.0
    x = ((mag - DEAD_ZONE) / (1.0 - DEAD_ZONE)) ** RESPONSE_POWER
    return x if d > 0 else -x


def build_params(model: OpponentModel) -> Params:
    """Baseline params, shifted by the opponent's measured deviations (the retaliation)."""
    p = base_params()
    if TFT_DISABLED:
        return p
    a = soften(model.aggression_dev())   # > 0 maniac, < 0 passive
    f = soften(model.fold_dev())         # > 0 tight,  < 0 station
    b = soften(model.bluff_dev())        # > 0 shows down bluffs

    if a > 0:
        # Maniac: their bets mean little, so call wider, raise thinner for value and trap.
        p.maniac = a
        p.call_margin -= 0.12 * a
        p.raise_pot -= 0.10 * a
        p.raise_half -= 0.10 * a
        p.trap_prob = 1.2 * max(0.0, a - 0.5)   # trapping only pays against a true maniac
        p.bluff_rate *= (1.0 - a)
        p.call_vs_open -= 0.10 * a
        p.call_vs_3bet -= 0.10 * a
        p.threebet -= 0.05 * a
    else:
        # Passive: their rare bets mean strength.
        p.call_margin += 0.05 * (-a)
    if b > 0:
        p.call_margin -= 0.05 * b
        p.river_margin -= 0.04 * b

    if f > 0:
        # Tight / folds too much: steal, c-bet and bet thinner.
        p.open_raise -= 0.25 * f
        p.open_limp -= 0.10 * f
        p.bb_iso -= 0.15 * f
        p.bluff_rate = 0.12 + 0.30 * f
        p.bet_half -= 0.06 * f
    else:
        # Calling station: never bluff, value-bet much thinner.
        d = -f
        p.bluff_rate *= (1.0 - d)
        p.bet_pot -= 0.12 * d
        p.bet_half -= 0.12 * d
        p.call_margin += 0.03 * d

    p.open_raise = _clamp(p.open_raise, 0.05, 0.65)
    p.open_limp = _clamp(p.open_limp, 0.0, 0.35)
    p.bb_iso = _clamp(p.bb_iso, 0.45, 0.90)
    p.threebet = _clamp(p.threebet, 0.80, 0.95)
    p.call_vs_open = _clamp(p.call_vs_open, 0.10, 0.50)
    p.call_vs_3bet = _clamp(p.call_vs_3bet, 0.40, 0.75)
    p.bet_pot = _clamp(p.bet_pot, 0.60, 0.88)
    p.bet_half = _clamp(p.bet_half, 0.45, 0.72)
    p.raise_pot = _clamp(p.raise_pot, 0.68, 0.92)
    p.raise_half = _clamp(p.raise_half, 0.58, 0.80)
    p.call_margin = _clamp(p.call_margin, -0.08, 0.17)
    p.river_margin = _clamp(p.river_margin, 0.0, 0.08)
    p.bluff_rate = _clamp(p.bluff_rate, 0.0, 0.45)
    p.trap_prob = _clamp(p.trap_prob, 0.0, 0.6)
    return p


# ----- memory ------------------------------------------------------------------

def fresh_memory(config: AgentConfig | None = None) -> dict:
    try:
        num_hands = int(getattr(config, "num_hands", 1000) or 1000)
    except Exception:
        num_hands = 1000
    return {
        "model": OpponentModel(),
        "rng": random.Random(20251107),
        "time_used": 0.0,
        "decisions": 0,
        "hands_played": 0,
        "net": 0,
        "num_hands": num_hands,
        "eq_cache": {},
    }


def ensure_memory(memory: dict) -> None:
    """Fill in anything missing, so a failed new_half() or a corrupted entry can never
    leave the bot stuck on the fallback action for the rest of the half."""
    defaults = fresh_memory()
    for key, value in defaults.items():
        if key not in memory or not isinstance(memory[key], type(value)):
            memory[key] = value


# ----- decision ----------------------------------------------------------------

def decide(obs: Observation, memory: dict) -> Action:
    t0 = time.perf_counter()
    ensure_memory(memory)
    model: OpponentModel = memory["model"]
    try:
        model.observe(obs)
    except Exception:
        memory["model"] = model = OpponentModel()   # start over rather than crash
    try:
        p = build_params(model)
    except Exception:
        p = base_params()
    if obs.street == Street.PREFLOP:
        act = _preflop(obs, p)
    else:
        act = _postflop(obs, p, memory)
    act = _legalize(act, obs)
    try:
        model.record_my_action(act)
    except Exception:
        pass
    memory["decisions"] += 1
    memory["time_used"] += time.perf_counter() - t0
    return act


def _legalize(act: Action, obs: Observation) -> Action:
    legal = obs.legal_actions
    if act in legal:
        return act
    if act in RAISES and Action.RAISE_HALF in legal:
        return Action.RAISE_HALF
    return Action.CHECK_CALL  # always legal


def _percentile(obs: Observation) -> float:
    try:
        return float(preflop.percentile(obs.hole_cards))
    except Exception:
        return 0.5


def _preflop(obs: Observation, p: Params) -> Action:
    pct = _percentile(obs)
    r = obs.raises_this_street
    if obs.to_call == 0:
        # Big blind after the button limped: isolate strong hands, otherwise check.
        return Action.RAISE_POT if pct > p.bb_iso else Action.CHECK_CALL
    if r == 0:
        # Button's first decision: raise, limp or fold.
        if pct > p.open_raise:
            return Action.RAISE_POT
        if pct > p.open_limp:
            return Action.CHECK_CALL
        return Action.FOLD
    if r == 1:
        # Facing one raise (we're the big blind, or we limped and got raised).
        if pct > p.threebet:
            return Action.RAISE_POT
        if pct > p.call_vs_open:
            return Action.CHECK_CALL
        return Action.FOLD
    if r == 2:
        # Facing a 3-bet: a 4-bet is effectively all-in.
        if pct > p.fourbet:
            return Action.RAISE_POT
        if pct > p.call_vs_3bet:
            return Action.CHECK_CALL
        return Action.FOLD
    # Facing a 4-bet or more: only the very top continues.
    return Action.CHECK_CALL if pct > p.call_vs_4bet else Action.FOLD


def _commit_guard(act: Action, e: float, threshold: float, obs: Observation) -> Action:
    """Downgrade a raise that would commit most of our stack on a marginal edge.
    Lower variance for almost no EV: thin value goes in as a smaller bet or a call."""
    if act not in RAISES or act not in obs.raise_to:
        return act
    added = obs.raise_to[act] - obs.my_bet
    if obs.my_stack > 0 and added >= COMMIT_FRACTION * obs.my_stack and e < threshold + COMMIT_WINDOW:
        return Action.RAISE_HALF if act == Action.RAISE_POT else Action.CHECK_CALL
    return act


def _postflop(obs: Observation, p: Params, memory: dict) -> Action:
    rng: random.Random = memory["rng"]
    e = _equity(obs, memory)
    if obs.to_call == 0:
        if p.trap_prob > 0 and e > 0.85 and not obs.is_button and rng.random() < p.trap_prob:
            return Action.CHECK_CALL   # trap: let the aggressor bluff into us
        if e > p.bet_pot:
            return _commit_guard(Action.RAISE_POT, e, p.bet_pot, obs)
        if e > p.bet_half:
            return _commit_guard(Action.RAISE_HALF, e, p.bet_half, obs)
        if (obs.street != Street.RIVER and obs.raises_this_street == 0
                and 0.30 < e < p.bet_half and rng.random() < p.bluff_rate):
            return Action.RAISE_HALF   # semi-bluff
        return Action.CHECK_CALL

    pot_after = obs.pot + obs.to_call
    po = obs.to_call / pot_after if pot_after > 0 else 0.5
    all_in_call = obs.to_call >= obs.my_stack
    if not all_in_call:
        if e > p.raise_pot:
            return _commit_guard(Action.RAISE_POT, e, p.raise_pot, obs)
        if e > p.raise_half:
            return _commit_guard(Action.RAISE_HALF, e, p.raise_half, obs)
        margin = p.call_margin + (p.river_margin if obs.street == Street.RIVER else 0.0)
    else:
        # No more decisions: pot odds plus a little respect for their range. A maniac's
        # all-in range is close to random, so the respect shrinks with their aggression.
        margin = ALL_IN_MARGIN * (1.0 - p.maniac)
    if e > po + margin:
        return Action.CHECK_CALL
    return Action.FOLD


# ----- equity with caching, time budget and emergency fallback ---------------------

_CATEGORY_PROXY = {
    HIGH_CARD: 0.10, ONE_PAIR: 0.22, TWO_PAIR: 0.40, STRAIGHT: 0.55, THREE_OF_A_KIND: 0.60,
    FULL_HOUSE: 0.80, FLUSH: 0.86, FOUR_OF_A_KIND: 0.97, STRAIGHT_FLUSH: 0.995,
}


def _samples(obs: Observation, memory: dict) -> int:
    """How many Monte Carlo samples we can afford right now (0 = emergency)."""
    remaining = TIME_TARGET - memory["time_used"]
    if remaining < EMERGENCY_RESERVE:
        return 0
    hands_left = max(1, memory["num_hands"] - obs.hand_index)
    budget = min(remaining / (hands_left * EST_PER_HAND + 1.0), MAX_REQUEST_SECONDS)
    n = int(budget / max(COST_PER_SAMPLE, 1e-6))
    return int(_clamp(n, MIN_SAMPLES, MAX_SAMPLES))


def _opp_min_pct(memory: dict) -> float | None:
    """Preflop-percentile floor for sampled opponent hands, from their preflop raises.
    A maniac's raises carry less information, so the floor shrinks with aggression."""
    if not RANGE_FILTER:
        return None
    model: OpponentModel = memory["model"]
    k = model.opp_raised_preflop
    if k == 0:
        return None
    floor = 0.35 if k == 1 else 0.70
    a = soften(model.aggression_dev()) if not TFT_DISABLED else 0.0
    if a > 0:
        floor *= (1.0 - a)
    return floor if floor > 0.05 else None


def _equity(obs: Observation, memory: dict) -> float:
    cache: dict = memory["eq_cache"]
    key = (obs.hand_index, int(obs.street))
    hit = cache.get(key)
    if hit is not None:
        return hit
    if cache and next(iter(cache))[0] != obs.hand_index:
        cache.clear()
    n = _samples(obs, memory)
    e = None
    if n > 0:
        try:
            e = equity.estimate(obs.hole_cards, obs.board, n, memory["rng"], _opp_min_pct(memory))
        except Exception:
            e = None
    if e is None:
        e = _cheap_equity(obs)
    cache[key] = e
    return e


def _cheap_equity(obs: Observation) -> float:
    try:
        if len(obs.board) == 5:
            return _CATEGORY_PROXY[category_of(omaha_score(obs.hole_cards, obs.board))]
    except Exception:
        pass
    return 0.25 + 0.5 * _percentile(obs)


# ----- end of hand ------------------------------------------------------------------

def hand_ended(result: HandResult, memory: dict) -> None:
    ensure_memory(memory)
    model: OpponentModel = memory["model"]
    model.hand_ended(result)
    memory["hands_played"] += 1
    memory["net"] += result.my_net
    if DEBUG and memory["hands_played"] % 100 == 0:
        print(f"[tft] hand {memory['hands_played']} net={memory['net']:+d} "
              f"time={memory['time_used']:.2f}s decisions={memory['decisions']} "
              f"model={model.summary()}", file=sys.stderr, flush=True)
