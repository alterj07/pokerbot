# TitForTatBot: how the strategy works

The bot plays **Tit-for-Tat**, the strategy that won Axelrod's iterated
prisoner's-dilemma tournaments, translated to heads-up poker:

| Tit-for-Tat | In this bot |
|---|---|
| Cooperate first | Play a solid equity-based baseline (`Params` defaults in `strategy.py`). |
| Retaliate in proportion | `OpponentModel` measures the opponent's deviation from "polite" play; `build_params()` shifts the thresholds toward the counter-strategy by that amount. |
| Forgive | All statistics are exponentially weighted moving averages. When the opponent stops deviating, the stats decay back to baseline within a few dozen hands and the adjustments fade. |

## Files
| File | Role |
|---|---|
| `agent.py` | Template server. Our code only calls into `strategy.py` and falls back to `CHECK_CALL` on any exception. |
| `strategy.py` | Baseline thresholds, the Tit-for-Tat adjustment table, time-budget control. |
| `opponent_model.py` | EWMA stats: preflop raise %, postflop aggression %, fold-to-bet %, showdown bluff %, and the three deviation signals. |
| `equity.py` | Monte Carlo equity (2-from-hand + 3-from-board), optional preflop-range filter for opponents who raised. |
| `preflop.py` + `data/preflop_equity.json` | Lookup of every suit-canonical 5-card starting hand's equity and percentile (built by `tools/build_preflop_table.py`). |
| `tests/test_strategy.py` | Legality, table coverage, retaliation-then-forgiveness, time budget. |
| `tools/run_match.sh` | Start our bot + an opponent and run the judge in one command. |

## Retaliation table
| Opponent deviation | Counter |
|---|---|
| Hyper-aggressive / shows down bluffs | call wider, raise thinner for value, check-call monsters out of position (trap), stop bluffing, defend preflop wider |
| Tight / folds too much | open and steal more, c-bet/semi-bluff more, value-bet thinner |
| Calling station | never bluff, value-bet thinner |
| Passive | respect their rare bets |

Deviations are 0 until enough actions have been observed (linear ramp), so the bot cooperates first.

## Switches (environment variables, read at startup)
- `TFT_DISABLED=1` plays the pure baseline (for A/B testing the adaptive layer).
- `RANGE_FILTER=0` disables range-aware opponent sampling.
- `TFT_DEBUG=1` prints the opponent model every 100 hands to stderr.

## Local results (1000 deals, both halves, 0 errors, ~4.3 s of the 8 s time bank per half)
| Opponent | seed 1 | seed 2 |
|---|---|---|
| RandomBot | +6.7 bb/deal | +7.4 |
| AlwaysPot | +5.7 | +5.4 |
| AlwaysHalf | +8.8 | +9.5 |
| CallingStation | +11.6 | +11.1 |

## Rebuilding the preflop table
```
python tools/build_preflop_table.py --samples 1500     # ~3.5 min on 16 cores
```
