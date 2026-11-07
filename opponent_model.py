"""
opponent_model.py - the Tit-for-Tat "sensor".

Tracks how the opponent plays with exponentially weighted moving averages
(EWMA). The decay is the FORGIVENESS of Tit-for-Tat: when the opponent stops
deviating, the statistics drift back to baseline within a few dozen hands
and the bot's adjustments fade with them. Sample-count gating ("ramp") means
the bot COOPERATES (plays its baseline) until it has seen enough to judge.

Stats (all frequencies in [0, 1]):
    pfr          opponent raised when acting preflop
    agg          opponent bet/raised when acting postflop (checks/calls = 0)
    fold_to_bet  opponent folded when facing our bet/raise
    bluff        at showdown, opponent had bet/raised the river with a weak hand
    wtsd         hand went to showdown

Deviations (each clamped to [-1, 1], 0 = "polite" baseline):
    aggression_dev()   > 0 hyper-aggressive, < 0 passive
    fold_dev()         > 0 folds too much (tight), < 0 calling station
    bluff_dev()        > 0 shows down bluffs more often than a sound player

Opponent actions are harvested from obs.action_history during our decisions
(each history entry is processed exactly once), and the final action of a
hand, which we never see in a history, is inferred in hand_ended().
"""

from __future__ import annotations

from evaluator import TWO_PAIR, category_of, omaha_score
from game import Action, HandResult, Observation, Street

RAISES = (Action.RAISE_HALF, Action.RAISE_POT)

# "Polite" baselines = what OUR OWN non-adaptive baseline measures when it plays itself
# (1000-deal self-match, TFT_DISABLED=1 TFT_DEBUG=1, see STRATEGY.md). A mirror opponent
# therefore produces zero deviation, and only real differences trigger a response.
AGG_BASE = 0.24      # postflop bet/raise frequency       (measured 0.21-0.27)
PFR_BASE = 0.40      # preflop raise frequency            (measured 0.38-0.42)
FTB_BASE = 0.36      # fold-to-bet frequency              (measured 0.30-0.42)
BLUFF_BASE = 0.05    # weak hand shown after river aggression (measured 0.005; 0.05 leaves slack)
SCALE = 0.30         # deviation of this size = a full ±1

ALPHA_ACTION = 0.02  # EWMA step for per-action stats (half-life ~35 actions)
ALPHA_HAND = 0.04    # EWMA step for per-hand stats (half-life ~17 hands)

# Minimum observations before a stat has full weight (linear ramp from 0).
RAMP_ACTIONS = 25
RAMP_FOLDS = 15
RAMP_BLUFF = 8


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


class Ewma:
    __slots__ = ("value", "n", "alpha")

    def __init__(self, start: float, alpha: float) -> None:
        self.value = start
        self.n = 0
        self.alpha = alpha

    def add(self, x: float) -> None:
        self.n += 1
        self.value += self.alpha * (x - self.value)

    def weight(self, ramp: int) -> float:
        return min(1.0, self.n / ramp)


class OpponentModel:
    def __init__(self) -> None:
        self.pfr = Ewma(PFR_BASE, ALPHA_ACTION)
        self.agg = Ewma(AGG_BASE, ALPHA_ACTION)
        self.fold_to_bet = Ewma(FTB_BASE, ALPHA_ACTION)
        self.bluff = Ewma(BLUFF_BASE, ALPHA_HAND)
        self.wtsd = Ewma(0.5, ALPHA_HAND)
        self.hands = 0
        # Per-hand scratch state, reset by _start_hand().
        self.hand_index = -1
        self.processed = 0          # history entries already counted
        self.my_last: Action | None = None
        self.opp_river_aggr = False
        self.opp_raised_preflop = 0  # number of opp preflop raises this hand

    # ----- during the hand ---------------------------------------------------

    def _start_hand(self, hand_index: int) -> None:
        self.hand_index = hand_index
        self.processed = 0
        self.my_last = None
        self.opp_river_aggr = False
        self.opp_raised_preflop = 0

    def observe(self, obs: Observation) -> None:
        """Count every not-yet-seen opponent action in obs.action_history."""
        if obs.hand_index != self.hand_index:
            self._start_hand(obs.hand_index)
        hist = obs.action_history
        if len(hist) < self.processed:      # inconsistent history: start the hand's count over
            self.processed = 0
        prev_me_raised = False
        if self.processed > 0:
            prev = hist[self.processed - 1]
            prev_me_raised = prev.is_me and prev.action in RAISES
        for rec in hist[self.processed:]:
            if rec.is_me:
                prev_me_raised = rec.action in RAISES
                continue
            is_raise = rec.action in RAISES
            if rec.street == Street.PREFLOP:
                self.pfr.add(1.0 if is_raise else 0.0)
                if is_raise:
                    self.opp_raised_preflop += 1
            else:
                self.agg.add(1.0 if is_raise else 0.0)
                if is_raise and rec.street == Street.RIVER:
                    self.opp_river_aggr = True
            if prev_me_raised:
                # The opponent faced our bet/raise and did not fold (folds end the hand).
                self.fold_to_bet.add(0.0)
            prev_me_raised = False
        self.processed = len(hist)

    def record_my_action(self, action: Action) -> None:
        self.my_last = action

    # ----- at the end of the hand ---------------------------------------------

    def hand_ended(self, result: HandResult) -> None:
        if result.hand_index != self.hand_index:
            # We never acted in this hand (e.g. the opponent folded preflop before our turn).
            self._start_hand(result.hand_index)
        self.hands += 1
        self.wtsd.add(1.0 if result.went_to_showdown else 0.0)
        # The opponent's last action is never in a history we see: infer it.
        if self.my_last in RAISES:
            if result.went_to_showdown:
                self.fold_to_bet.add(0.0)        # they called our bet
            elif result.winner == "me":
                self.fold_to_bet.add(1.0)        # they folded to our bet
        if (result.went_to_showdown and result.opp_hole is not None
                and len(result.board) == 5 and len(result.opp_hole) == 5):
            if self.opp_river_aggr:
                score = omaha_score(result.opp_hole, result.board)
                weak = category_of(score) < TWO_PAIR
                self.bluff.add(1.0 if weak else 0.0)

    # ----- deviations (the signals Tit-for-Tat reacts to) ----------------------

    def aggression_dev(self) -> float:
        post = (self.agg.value - AGG_BASE) / SCALE * self.agg.weight(RAMP_ACTIONS)
        pre = (self.pfr.value - PFR_BASE) / SCALE * self.pfr.weight(RAMP_ACTIONS)
        return _clamp(0.6 * post + 0.4 * pre)

    def fold_dev(self) -> float:
        return _clamp((self.fold_to_bet.value - FTB_BASE) / SCALE * self.fold_to_bet.weight(RAMP_FOLDS))

    def bluff_dev(self) -> float:
        return _clamp((self.bluff.value - BLUFF_BASE) / SCALE * self.bluff.weight(RAMP_BLUFF))

    def summary(self) -> dict:
        return {
            "hands": self.hands,
            "pfr": round(self.pfr.value, 3), "agg": round(self.agg.value, 3),
            "fold_to_bet": round(self.fold_to_bet.value, 3), "bluff": round(self.bluff.value, 3),
            "wtsd": round(self.wtsd.value, 3),
            "aggression_dev": round(self.aggression_dev(), 2),
            "fold_dev": round(self.fold_dev(), 2), "bluff_dev": round(self.bluff_dev(), 2),
        }
