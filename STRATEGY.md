# TitForTatBot: how the strategy works

The bot plays **Tit-for-Tat**, the strategy that won Axelrod's iterated
prisoner's-dilemma tournaments, translated to heads-up poker:

| Tit-for-Tat | In this bot |
|---|---|
| Cooperate first | Play a solid equity-based baseline (`Params` defaults in `strategy.py`). Adjustments are zero until enough actions have been seen. |
| Retaliate in proportion | `OpponentModel` measures the opponent's deviation from "polite" play; `build_params()` shifts the thresholds toward the counter-strategy by that amount. |
| Forgive | All statistics are exponentially weighted moving averages. When the opponent stops deviating, the stats decay back to baseline within a few dozen hands and the adjustments fade. |
| **Generous** | Poker is noisy. Deviations inside a dead zone (0.15) are ignored, so a normal opponent on a hot streak does not trigger retaliation. Strict Tit-for-Tat loses value in noisy games; the generous variant does not. |

## Files
| File | Role |
|---|---|
| `agent.py` | Template server. Our code only calls into `strategy.py`; every entry point is wrapped so an exception can never cost an error (`CHECK_CALL` is always legal), and memory is repaired if `new_half` ever fails. |
| `strategy.py` | Baseline thresholds, the Tit-for-Tat adjustment table, time-budget control, emergency fallbacks. |
| `opponent_model.py` | EWMA stats: preflop raise %, postflop aggression %, fold-to-bet %, showdown bluff %, and the three deviation signals. |
| `equity.py` | Monte Carlo equity (2-from-hand + 3-from-board), optional preflop-range filter for opponents who raised. |
| `preflop.py` + `data/preflop_equity.json` | Lookup of every suit-canonical 5-card starting hand's equity and percentile (built by `tools/build_preflop_table.py`). |
| `tests/test_strategy.py` | Legality fuzz (1000 hands, mixed opponents), empty-memory recovery, odd histories, table coverage, retaliation-then-forgiveness, time budget. |
| `tools/run_match.sh` | Start our bot + an opponent and run the judge in one command. |
| `tools/sparring.sh` | Full bot vs the 4 baselines and vs 3 non-adaptive copies of itself (static / tight / loose), one line per match. |

## Retaliation table
| Opponent deviation | Counter |
|---|---|
| Hyper-aggressive / shows down bluffs | call wider, raise thinner for value, defend preflop wider, stop bluffing; check-call monsters out of position (trap) only against a true maniac |
| Tight / folds too much | open and steal more, c-bet/semi-bluff more, value-bet thinner |
| Calling station | never bluff, value-bet much thinner |
| Passive | respect their rare bets |

## Calibration ("polite" = us)
The baselines the model compares against are what our own non-adaptive bot measures when it
plays itself (`AGENT1_ENV="TFT_DISABLED=1 TFT_DEBUG=1" AGENT2_ENV="TFT_DISABLED=1" ./tools/run_match.sh agent.py`):
preflop raise 0.40, postflop aggression 0.24, fold-to-bet 0.36, shown-down bluffs ~0.
So a mirror opponent produces zero deviation. Readings of the baselines: AlwaysPot/AlwaysHalf
aggression +1.0, fold -1.0; CallingStation aggression -1.0, fold -1.0; RandomBot aggression +1.0.

## Tuning history (what was measured, 1000-deal duplicate matches)
- **Tightness ladder.** The first baseline lost ~0.7 bb/deal to a copy of itself one step tighter
  on every seed. Two steps tighter gained only ~0.2 more but halved the margins against weak bots.
  The defaults are now one step tighter; the adaptive layer loosens again against stations/maniacs.
- **Commit guard** (downgrade pot-sized bets that commit most of the stack on marginal equity):
  cost ~2 bb/deal against loose callers. Off by default (`TFT_COMMIT_WINDOW=0`).
- **All-in margin** 0.04 above pot odds when calling off the stack, scaled to 0 against a maniac.
- **Response curve.** Linear beat quadratic over 5 seeds of self-sparring. `TFT_RESPONSE_POWER=1`.

## Switches (environment variables, read at startup; tournament uses the defaults)
- `TFT_DISABLED=1` plays the pure baseline (A/B testing). `STYLE=tight|loose` or `TIGHTNESS=<float>` shifts the baseline (sparring partners).
- `TFT_DEAD_ZONE`, `TFT_ALL_IN_MARGIN`, `TFT_COMMIT_WINDOW`, `TFT_RESPONSE_POWER`: tuning overrides.
- `RANGE_FILTER=0` disables range-aware opponent sampling. `TFT_DEBUG=1` prints the model every 100 hands to stderr.

## Results (final version, 1000 deals, both halves, 0 errors, judge-charged thinking <= 5.3 s of 8 s)
| Opponent | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| RandomBot | +8.1 bb/deal | +6.0 | +6.8 |
| AlwaysPot | +6.7 | +4.9 | +6.1 |
| AlwaysHalf | +8.9 | +8.4 | +8.8 |
| CallingStation | +9.9 | +9.0 | +9.7 |
| self (non-adaptive, same thresholds) | -0.3 (chop) | -0.2 (chop) | -0.1 (chop) |
| self tight | -0.2 (chop) | -0.4 (chop) | -0.4 (chop) |
| self loose | +0.1 (chop) | +0.2 (chop) | +0.5 (chop) |

All weak-bot matches are wins; all matches against competent copies of itself are chops within noise.

## Rebuilding the preflop table
```
python tools/build_preflop_table.py --samples 1500     # ~3.5 min on 16 cores
```
