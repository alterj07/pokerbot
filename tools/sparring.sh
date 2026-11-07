#!/bin/bash
# tools/sparring.sh - run the full bot against every practice opponent and print one line per match.
#
#   ./tools/sparring.sh [seed] [extra judge args...]
#
# Opponents: the four baseline bots, plus our own bot with adaptation disabled in three styles
# (static / tight / loose), which stand in for other teams' non-adaptive equity bots.
# Matches run in parallel on separate ports. Output: name, bb/deal, CI, errors, max thinking time, RESULT.
set -uo pipefail
cd "$(dirname "$0")/.."
SEED="${1:-1}"; shift || true
OUT="${OUT:-/tmp/sparring_$SEED}"
mkdir -p "$OUT"
declare -a NAMES OPPS ENVS
NAMES=(random always_pot always_half calling_station self_static self_tight self_loose)
OPPS=(baseline-bots/baseline_random.py baseline-bots/baseline_always_pot.py baseline-bots/baseline_always_half.py
      baseline-bots/baseline_calling_station.py agent.py agent.py agent.py)
ENVS=("" "" "" "" "TFT_DISABLED=1 STYLE=static" "TFT_DISABLED=1 STYLE=tight" "TFT_DISABLED=1 STYLE=loose")
port=5400
for i in "${!NAMES[@]}"; do
  AGENT2_ENV="${ENVS[$i]}" P1=$port P2=$((port+1)) ./tools/run_match.sh "${OPPS[$i]}" --seed "$SEED" "$@" \
      >"$OUT/${NAMES[$i]}.txt" 2>&1 &
  port=$((port+10))
done
wait
printf "%-16s %-10s %-22s %-8s %-9s %s\n" opponent bb/deal CI errors max_think RESULT
for n in "${NAMES[@]}"; do
  f="$OUT/$n.txt"
  bb=$(grep -oE 'agent1: [+-][0-9.]+' "$f" | awk '{print $2}')
  ci=$(grep -oE '\[[+-][0-9.]+, [+-][0-9.]+\]' "$f" | head -1)
  err=$(grep -E 'agent1 half' "$f" | grep -oE 'errors [0-9]+' | awk '{s+=$2} END{print s}')
  think=$(grep -E 'agent1 half' "$f" | grep -oE 'thinking [0-9.]+s' | awk '{gsub("s","",$2); if($2>m)m=$2} END{print m"s"}')
  res=$(grep -oE 'RESULT: [A-Z0-9_]+' "$f" | awk '{print $2}')
  flags=$(grep -E 'agent1 half' "$f" | grep -oE 'FORFEITED|TIME BANK EMPTY' | sort -u | tr '\n' ' ')
  printf "%-16s %-10s %-22s %-8s %-9s %s %s\n" "$n" "$bb" "$ci" "$err" "$think" "$res" "$flags"
done
