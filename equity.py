"""
equity.py - Monte Carlo equity for one hand against a random (or range-
filtered) opponent hand in Short Deck PLO5.

estimate(hole, board, n, rng, min_opp_pct=None) -> float in [0, 1]

    hole         our 5 hole cards
    board        0, 3, 4 or 5 board cards
    n            number of samples
    rng          a random.Random (kept in memory so halves are repeatable)
    min_opp_pct  if given, opponent hands whose preflop percentile is below
                 this are re-drawn (at most MAX_REDRAWS times). Used to model
                 an opponent who raised preflop and so holds a stronger range.

Each sample costs ~200 dict lookups: our best hand and the opponent's best
hand on the sampled board (exactly-2-from-hand + 3-from-board rule).
"""

from __future__ import annotations

import random
import time
from itertools import combinations
from typing import Sequence

from evaluator import NUM_CARDS, get_table

import preflop

_PAIRS = tuple(combinations(range(5), 2))
_TRIPLES = tuple(combinations(range(5), 3))
MAX_REDRAWS = 3


def estimate(hole: Sequence[int], board: Sequence[int], n: int, rng: random.Random,
             min_opp_pct: float | None = None) -> float:
    """Share of sampled showdowns we win (ties count half)."""
    table = get_table()
    known = set(hole) | set(board)
    deck = [c for c in range(NUM_CARDS) if c not in known]
    hb = [1 << c for c in hole]
    my_pairs = [hb[i] | hb[j] for i, j in _PAIRS]
    fixed = [1 << c for c in board]
    need = 5 - len(board)
    pct = preflop.percentile
    wins = 0.0
    sample = rng.sample
    for _ in range(n):
        draw = sample(deck, need + 5)
        opp = draw[need:]
        if min_opp_pct is not None:
            tries = 0
            while tries < MAX_REDRAWS and pct(opp) < min_opp_pct:
                draw = sample(deck, need + 5)
                opp = draw[need:]
                tries += 1
        bb = fixed + [1 << c for c in draw[:need]]
        bm = [bb[i] | bb[j] | bb[k] for i, j, k in _TRIPLES]
        my = 0
        for hm in my_pairs:
            for b in bm:
                s = table[hm | b]
                if s > my:
                    my = s
        ob = [1 << c for c in opp]
        theirs = 0
        for i, j in _PAIRS:
            hm = ob[i] | ob[j]
            for b in bm:
                s = table[hm | b]
                if s > theirs:
                    theirs = s
        if my > theirs:
            wins += 1.0
        elif my == theirs:
            wins += 0.5
    return wins / n


def measure_cost_per_sample(samples: int = 300) -> float:
    """Seconds per Monte Carlo sample on this machine (measured at startup)."""
    get_table()                       # build the lookup table first: that is not sampling cost
    rng = random.Random(1)
    hole = (35, 31, 25, 21, 0)
    board = (34, 17, 8)
    estimate(hole, board, 20, rng)    # warm caches
    t0 = time.perf_counter()
    estimate(hole, board, samples, rng)
    return (time.perf_counter() - t0) / samples
