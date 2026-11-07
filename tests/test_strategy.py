"""Tests for the Tit-for-Tat strategy modules. Run from the repo root: python -m pytest -q"""

import os
import random
import sys
from itertools import combinations

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import equity  # noqa: E402
import preflop  # noqa: E402
import strategy  # noqa: E402
from evaluator import cards_from_str  # noqa: E402
from game import Action, AgentConfig, Game, Street, make_deal  # noqa: E402
from game import ObservedAction, Observation  # noqa: E402
from opponent_model import OpponentModel  # noqa: E402


# ----- preflop -------------------------------------------------------------------

def test_canonical_key_is_suit_invariant():
    a = cards_from_str("Ah Kh Qd Jd 6c")
    b = cards_from_str("As Ks Qc Jc 6h")   # same hand, suits permuted
    c = cards_from_str("Ah Kd Qd Jh 6c")   # different suit pattern
    assert preflop.canonical_key(a) == preflop.canonical_key(b)
    assert preflop.canonical_key(a) != preflop.canonical_key(c)


def test_table_loaded_and_covers_every_hand():
    assert preflop.TABLE_LOADED, "run tools/build_preflop_table.py first"
    missing = 0
    for hand in combinations(range(36), 5):
        if preflop.canonical_key(hand) not in preflop._TABLE:
            missing += 1
    assert missing == 0


def test_table_orders_hands_sensibly():
    premium = preflop.percentile(cards_from_str("Ah Ad Kh Kd Qs"))
    mid = preflop.percentile(cards_from_str("6c 7d 8h 9s Jc"))
    junk = preflop.percentile(cards_from_str("6c 6d 6h 9s Jc"))   # trips = dead cards
    assert premium > 0.95
    assert premium > mid > junk


def test_heuristic_fallback_never_crashes():
    eq, pct = preflop._heuristic(cards_from_str("6c 6d 6h 6s Ac"))
    assert 0.0 <= eq <= 1.0 and 0.0 <= pct <= 1.0


# ----- equity ---------------------------------------------------------------------

def test_equity_extremes_on_river():
    rng = random.Random(1)
    board = cards_from_str("Ah Kh Qh 7c 6d")
    nuts = cards_from_str("Jh Th 9c 8c 6s")       # royal-ish straight flush (J T hearts)
    air = cards_from_str("8s 9d 6c 7d Js")        # 9-high straight at best? no: needs 2 hole
    e_nuts = equity.estimate(nuts, board, 200, rng)
    e_air = equity.estimate(air, board, 200, rng)
    assert e_nuts > 0.95
    assert e_air < e_nuts


def test_equity_range_filter_runs():
    rng = random.Random(2)
    hole = cards_from_str("Ah Ad Kh Kd Qs")
    e = equity.estimate(hole, cards_from_str("6c 7d 8h"), 100, rng, min_opp_pct=0.7)
    assert 0.0 <= e <= 1.0


# ----- decisions are always legal ---------------------------------------------------

def _play(num_hands, seed, raise_rate):
    memory = strategy.fresh_memory(AgentConfig())
    rng = random.Random(seed)
    for h in range(num_hands):
        game = Game(make_deal(seed, h), button_seat=h % 2, hand_index=h)
        while not game.is_terminal:
            seat = game.to_act
            obs = game.observation(seat)
            if seat == 0:
                act = strategy.decide(obs, memory)
                assert act in obs.legal_actions
            else:
                raises = [a for a in obs.legal_actions if a in (Action.RAISE_HALF, Action.RAISE_POT)]
                act = rng.choice(raises) if raises and rng.random() < raise_rate else Action.CHECK_CALL
            game.apply(act)
        strategy.hand_ended(game.hand_result(0), memory)
    return memory


def test_decisions_legal_vs_random_opponent():
    mem = _play(300, "legal", 0.3)
    assert mem["hands_played"] == 300
    assert mem["time_used"] < 3.0


def test_decisions_legal_vs_maniac():
    mem = _play(150, "maniac", 1.0)
    assert mem["model"].aggression_dev() > 0.5, mem["model"].summary()


def test_time_budget_adapts():
    mem = strategy.fresh_memory(AgentConfig())
    mem["time_used"] = strategy.TIME_TARGET - 0.1   # almost out of time
    game = Game(make_deal("t", 0), 0, 0)
    game.apply(Action.CHECK_CALL); game.apply(Action.CHECK_CALL)  # to the flop
    obs = game.observation(game.to_act)
    assert strategy._samples(obs, mem) == 0          # emergency mode
    act = strategy.decide(obs, mem)
    assert act in obs.legal_actions


# ----- the Tit-for-Tat sensor: retaliation and forgiveness ------------------------------

def _obs_with_history(hand_index, history):
    return Observation(hand_index=hand_index, street=Street.FLOP, hole_cards=(0, 5, 10, 15, 20),
                       board=(25, 30, 35), pot=10, to_call=0, my_bet=0, opp_bet=0, my_stack=75,
                       opp_stack=75, is_button=False, legal_actions=(Action.CHECK_CALL,),
                       raise_to={}, min_raise_to=2, raises_this_street=0,
                       action_history=tuple(history), big_blind=2)


def test_model_retaliates_then_forgives():
    m = OpponentModel()
    assert m.aggression_dev() == 0.0            # cooperate first
    for h in range(40):                          # 40 hands of opponent pot-raising every street
        hist = [ObservedAction(Street.PREFLOP, False, Action.RAISE_POT, 5, 6),
                ObservedAction(Street.PREFLOP, True, Action.CHECK_CALL, 4, 6),
                ObservedAction(Street.FLOP, False, Action.RAISE_POT, 12, 12)]
        m.observe(_obs_with_history(h, hist))
    assert m.aggression_dev() > 0.8
    for h in range(40, 100):                     # then 60 hands of pure passivity
        hist = [ObservedAction(Street.PREFLOP, False, Action.CHECK_CALL, 1, 2),
                ObservedAction(Street.PREFLOP, True, Action.CHECK_CALL, 0, 2),
                ObservedAction(Street.FLOP, False, Action.CHECK_CALL, 0, 0)]
        m.observe(_obs_with_history(h, hist))
    assert m.aggression_dev() < 0.0               # forgiven and then some


def test_params_shift_in_the_right_direction():
    base = strategy.Params()
    m = OpponentModel()
    for _ in range(60):
        m.agg.add(1.0); m.pfr.add(1.0)
    p = strategy.build_params(m)
    if not strategy.TFT_DISABLED:
        assert p.call_margin < base.call_margin       # call wider vs a maniac
        assert p.trap_prob > 0
        assert p.bluff_rate < base.bluff_rate
    m2 = OpponentModel()
    for _ in range(60):
        m2.fold_to_bet.add(1.0)
    p2 = strategy.build_params(m2)
    if not strategy.TFT_DISABLED:
        assert p2.open_raise < base.open_raise        # steal more vs a folder
        assert p2.bluff_rate > base.bluff_rate


# ----- hardening: the bot must never get stuck or crash ------------------------------------

def test_choose_action_with_empty_memory():
    import importlib.util
    spec = importlib.util.spec_from_file_location("agent_h", os.path.join(ROOT, "agent.py"))
    agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent)
    memory = {}                                   # as if new_half had failed
    game = Game(make_deal("empty", 0), 0, 0)
    obs = game.observation(0)
    act = agent.choose_action(obs, memory)
    assert act in obs.legal_actions
    assert "model" in memory and "rng" in memory  # repaired
    memory["eq_cache"] = "corrupt"                # wrong type gets replaced too
    strategy.ensure_memory(memory)
    assert isinstance(memory["eq_cache"], dict)
    agent.on_hand_end(game.hand_result(0) if game.is_terminal else _finish(game, 0), memory)


def _finish(game, seat):
    while not game.is_terminal:
        game.apply(Action.CHECK_CALL)
    return game.hand_result(seat)


def test_fuzz_many_hands_mixed_opponent():
    memory = strategy.fresh_memory(AgentConfig())
    rng = random.Random("fuzz")
    for h in range(1000):
        rate = rng.choice((0.0, 0.3, 0.7, 1.0))
        game = Game(make_deal("fuzz", h), button_seat=h % 2, hand_index=h)
        while not game.is_terminal:
            seat = game.to_act
            obs = game.observation(seat)
            if seat == 0:
                act = strategy.decide(obs, memory)
                assert act in obs.legal_actions
            else:
                raises = [a for a in obs.legal_actions if a in (Action.RAISE_HALF, Action.RAISE_POT)]
                if raises and rng.random() < rate:
                    act = rng.choice(raises)
                elif Action.FOLD in obs.legal_actions and rng.random() < 0.1:
                    act = Action.FOLD
                else:
                    act = Action.CHECK_CALL
            game.apply(act)
        strategy.hand_ended(game.hand_result(0), memory)
    assert memory["hands_played"] == 1000
    assert memory["time_used"] < 6.0


def test_model_tolerates_odd_histories():
    m = OpponentModel()
    long_hist = [ObservedAction(Street.PREFLOP, False, Action.RAISE_POT, 5, 6),
                 ObservedAction(Street.PREFLOP, True, Action.CHECK_CALL, 4, 6)]
    m.observe(_obs_with_history(3, long_hist))
    m.observe(_obs_with_history(3, long_hist[:1]))   # shorter than processed: must not raise
    m.observe(_obs_with_history(3, long_hist))        # repeated
    from game import HandResult
    # a hand we never acted in (opponent folded preflop before our turn)
    m.hand_ended(HandResult(hand_index=7, my_net=1, pot=3, went_to_showdown=False, board=(), opp_hole=None, winner="me"))
    # showdown with a short board (should be impossible, but must not raise)
    m.hand_ended(HandResult(hand_index=8, my_net=0, pot=4, went_to_showdown=True, board=(1, 2, 3), opp_hole=(4, 5, 6, 7, 8), winner="split"))
    assert m.hands == 2


def test_soften_dead_zone():
    assert strategy.soften(0.1) == 0.0 and strategy.soften(-0.1) == 0.0
    assert strategy.soften(1.0) == pytest.approx(1.0)
    assert 0.0 < strategy.soften(0.5) < 0.5


def test_commit_guard_downgrades_marginal_stack_off():
    game = Game(make_deal("cg", 0), 0, 0)
    game.apply(Action.RAISE_POT); game.apply(Action.RAISE_POT); game.apply(Action.CHECK_CALL)  # 3-bet pot
    obs = game.observation(game.to_act)              # flop, pot 36, stacks 62: pot bet commits 36/62 = 58%
    assert obs.street == Street.FLOP and obs.to_call == 0
    big = obs.raise_to[Action.RAISE_POT] - obs.my_bet
    if big >= strategy.COMMIT_FRACTION * obs.my_stack:
        assert strategy._commit_guard(Action.RAISE_POT, 0.73, 0.72, obs) == Action.RAISE_HALF
    assert strategy._commit_guard(Action.RAISE_POT, 0.95, 0.72, obs) == Action.RAISE_POT


def test_styles_shift_params():
    tight = strategy.Params(); loose = strategy.Params()
    saved = strategy.TIGHTNESS
    try:
        strategy.TIGHTNESS = 1.0; tight = strategy.base_params()
        strategy.TIGHTNESS = -1.0; loose = strategy.base_params()
    finally:
        strategy.TIGHTNESS = saved
    assert tight.bet_pot > strategy.Params().bet_pot > loose.bet_pot
    assert tight.open_raise > loose.open_raise
