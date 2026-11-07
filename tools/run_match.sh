#!/bin/bash
# tools/run_match.sh - play our bot against an opponent bot file and print the judge report.
#
#   ./tools/run_match.sh baseline-bots/baseline_random.py [judge args...]
#   ./tools/run_match.sh baseline-bots/baseline_always_pot.py --deals 300 --seed 3
#   AGENT2_ENV="TFT_DISABLED=1 STYLE=tight" ./tools/run_match.sh agent.py --seed 2   # sparring vs ourselves
#
# Env: AGENT1_ENV / AGENT2_ENV = extra "VAR=value" pairs for each bot; P1/P2 = ports;
#      PY = interpreter (default .venv/bin/python). Bot logs: /tmp/bot1_$P1.log, /tmp/bot2_$P2.log
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
OPP="$1"; shift
P1="${P1:-5008}"; P2="${P2:-5009}"
env ${AGENT1_ENV:-} PORT=$P1 "$PY" agent.py >/tmp/bot1_$P1.log 2>&1 &
B1=$!
env ${AGENT2_ENV:-} PORT=$P2 "$PY" "$OPP" >/tmp/bot2_$P2.log 2>&1 &
B2=$!
trap 'kill $B1 $B2 2>/dev/null; wait $B1 $B2 2>/dev/null' EXIT
for i in $(seq 1 60); do
  curl -s "http://localhost:$P1/" >/dev/null 2>&1 && curl -s "http://localhost:$P2/" >/dev/null 2>&1 && break
  sleep 0.5
done
"$PY" judge_engine.py --agent1 "$P1" --agent2 "$P2" "$@"
