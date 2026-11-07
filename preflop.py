"""
preflop.py - starting-hand strength for 5-card Short Deck PLO.

Looks a 5-card starting hand up in a precomputed table of Monte Carlo
equities against a random hand (built offline by tools/build_preflop_table.py
and shipped as data/preflop_equity.json).

Hands that differ only by a permutation of the suits play identically, so
the table is keyed by a suit-canonical form: the multiset of "ranks held in
each suit". E.g. "Ah Kh Qd Jd 6c" -> groups {A K}, {Q J}, {6} -> key "87|65|0"
(ranks as numbers, groups sorted big-first, separated by '|').

Public API:
    canonical_key(cards)  -> str
    strength(cards)       -> (equity_vs_random, percentile)   both in [0, 1]
    percentile(cards)     -> float
"""

from __future__ import annotations

import json
import os
from typing import Sequence

_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "preflop_equity.json")

# key -> (equity, percentile). Filled by _load() at import.
_TABLE: dict[str, tuple[float, float]] = {}
TABLE_LOADED = False


def canonical_key(cards: Sequence[int]) -> str:
    """Suit-isomorphism-invariant key for a set of cards (see module doc)."""
    by_suit: list[list[int]] = [[], [], [], []]
    for c in cards:
        by_suit[c & 3].append(c >> 2)
    groups = []
    for ranks in by_suit:
        if ranks:
            ranks.sort(reverse=True)
            groups.append(tuple(ranks))
    # Bigger groups first, then higher ranks first. Tuples compare fine.
    groups.sort(key=lambda g: (len(g), g), reverse=True)
    return "|".join("".join(str(r) for r in g) for g in groups)


def _heuristic(cards: Sequence[int]) -> tuple[float, float]:
    """Crude fallback if a key is somehow missing: never crash."""
    ranks = sorted((c >> 2 for c in cards), reverse=True)
    suits = [c & 3 for c in cards]
    score = 0.0
    score += sum(ranks) / (8 * 5) * 0.4                     # high cards
    counts = {r: ranks.count(r) for r in set(ranks)}
    pairs = sum(1 for v in counts.values() if v == 2)
    trips = sum(1 for v in counts.values() if v >= 3)
    score += 0.15 * pairs - 0.1 * trips                      # pairs good, trips bad (dead cards)
    suit_counts = {s: suits.count(s) for s in set(suits)}
    score += 0.1 * sum(1 for v in suit_counts.values() if v == 2)
    distinct = sorted(set(ranks))
    gaps = sum(1 for a, b in zip(distinct, distinct[1:]) if b - a == 1)
    score += 0.05 * gaps
    p = max(0.0, min(1.0, score))
    return 0.35 + 0.3 * p, p


def _load() -> None:
    global _TABLE, TABLE_LOADED
    try:
        with open(_DATA_PATH, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        _TABLE = {k: (float(v[0]), float(v[1])) for k, v in raw["table"].items()}
        TABLE_LOADED = True
    except Exception:  # missing/corrupt file: fall back to the heuristic
        _TABLE = {}
        TABLE_LOADED = False


def strength(cards: Sequence[int]) -> tuple[float, float]:
    """(equity vs a random hand, percentile among all starting hands)."""
    hit = _TABLE.get(canonical_key(cards))
    if hit is None:
        return _heuristic(cards)
    return hit


def percentile(cards: Sequence[int]) -> float:
    """Percentile of this starting hand among all 376,992 hands (1.0 = best)."""
    return strength(cards)[1]


_load()
