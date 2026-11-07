"""
tools/build_preflop_table.py - OFFLINE builder for data/preflop_equity.json.

For every suit-canonical 5-card starting hand it estimates, by Monte Carlo,
the equity against one random opponent hand on a random board, then turns
those equities into percentiles over all C(36,5) = 376,992 raw hands.

Run from the repo root (takes a few minutes on a multi-core machine):
    python tools/build_preflop_table.py --samples 1500

The output is committed to the repo; the bot only reads it.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from itertools import combinations
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluator import NUM_CARDS, get_table  # noqa: E402
from preflop import canonical_key  # noqa: E402

_PAIRS = tuple(combinations(range(5), 2))
_TRIPLES = tuple(combinations(range(5), 3))


def equity_vs_random(hole: tuple[int, ...], samples: int, seed: int) -> float:
    """Monte Carlo equity of `hole` against a random 5-card hand."""
    table = get_table()
    rng = random.Random(seed)
    deck = [c for c in range(NUM_CARDS) if c not in hole]
    hb = [1 << c for c in hole]
    my_pairs = [hb[i] | hb[j] for i, j in _PAIRS]
    wins = 0.0
    for _ in range(samples):
        draw = rng.sample(deck, 10)
        bb = [1 << c for c in draw[:5]]
        ob = [1 << c for c in draw[5:]]
        bm = [bb[i] | bb[j] | bb[k] for i, j, k in _TRIPLES]
        my = 0
        for hm in my_pairs:
            for b in bm:
                s = table[hm | b]
                if s > my:
                    my = s
        opp = 0
        for i, j in _PAIRS:
            hm = ob[i] | ob[j]
            for b in bm:
                s = table[hm | b]
                if s > opp:
                    opp = s
        if my > opp:
            wins += 1.0
        elif my == opp:
            wins += 0.5
    return wins / samples


def _work(args):
    key, rep, samples, seed = args
    return key, equity_vs_random(rep, samples, seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "preflop_equity.json"))
    args = ap.parse_args()

    t0 = time.time()
    reps: dict[str, tuple[int, ...]] = {}
    counts: dict[str, int] = {}
    for hand in combinations(range(NUM_CARDS), 5):
        k = canonical_key(hand)
        if k not in reps:
            reps[k] = hand
        counts[k] = counts.get(k, 0) + 1
    total = sum(counts.values())
    print(f"{len(reps)} suit-canonical classes covering {total} hands ({time.time() - t0:.1f}s)")

    jobs = [(k, rep, args.samples, i) for i, (k, rep) in enumerate(sorted(reps.items()))]
    equities: dict[str, float] = {}
    t1 = time.time()
    with Pool(args.workers, initializer=get_table) as pool:
        for n, (k, eq) in enumerate(pool.imap_unordered(_work, jobs, chunksize=16), 1):
            equities[k] = eq
            if n % 1000 == 0:
                print(f"  {n}/{len(jobs)} classes, {time.time() - t1:.0f}s", flush=True)
    print(f"simulation done in {time.time() - t1:.0f}s")

    # Percentile = share of all raw hands with strictly lower equity, plus half of the ties.
    ordered = sorted(equities.items(), key=lambda kv: kv[1])
    below = 0
    table_out: dict[str, list[float]] = {}
    i = 0
    while i < len(ordered):
        j = i
        while j < len(ordered) and ordered[j][1] == ordered[i][1]:
            j += 1
        tie_count = sum(counts[k] for k, _ in ordered[i:j])
        pct = (below + 0.5 * tie_count) / total
        for k, eq in ordered[i:j]:
            table_out[k] = [round(eq, 4), round(pct, 4)]
        below += tie_count
        i = j

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"samples": args.samples, "classes": len(table_out), "hands": total,
                   "table": table_out}, fh, separators=(",", ":"))
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.2f} MB) in {time.time() - t0:.0f}s total")


if __name__ == "__main__":
    main()
