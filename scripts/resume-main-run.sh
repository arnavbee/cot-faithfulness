#!/bin/bash
# Keeps the main run going until every planned call is on disk.
#
# The predecessor of this script relaunched every 120 seconds regardless of why
# the run had stopped, which meant it spent the day re-discovering the same
# rate-limit wall and logging a dead row each time. This one reads the exit code:
#
#   0  finished          -> stop
#   3  daily budget gone  -> sleep out the window, then resume
#   *  crashed            -> short backoff, resume, give up after a few in a row
#
# Resuming is free: errored records are never cached, so the run picks up
# exactly where it left off.
set -u
cd "$(dirname "$0")/.." || exit 1

RUN=main-40x3
LOG=/tmp/cotf-main-run.log
COOL=1800          # 30m, one roll of Groq's day window
CRASH_BACKOFF=60
MAX_CRASHES=5
crashes=0

log() { echo "[supervisor $(date '+%H:%M:%S')] $*" >> "$LOG"; }

while :; do
  .venv/bin/python -m cotf run "$RUN" --provider groq --dataset mmlu \
    --n 40 --repeats 3 --hints sycophancy metadata authority reward_hack \
    --temperature 0.0 --max-tokens 2048 --seed 0 --concurrency 3 >> "$LOG" 2>&1
  code=$?
  case $code in
    0) log "run complete"; exit 0 ;;
    3) crashes=0; log "daily budget gone, sleeping ${COOL}s for the window to roll"
       sleep "$COOL" ;;
    *) crashes=$((crashes + 1))
       log "exit $code (crash $crashes/$MAX_CRASHES), retrying in ${CRASH_BACKOFF}s"
       [ "$crashes" -ge "$MAX_CRASHES" ] && { log "giving up, needs a human"; exit 1; }
       sleep "$CRASH_BACKOFF" ;;
  esac
done
