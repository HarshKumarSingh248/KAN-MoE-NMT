#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
status=0
for lang in hindi bengali malayalam; do
  checkpoint="checkpoints/$lang/best_model2.pt"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Skipping $lang: checkpoint unavailable at $checkpoint" >&2
    continue
  fi
  for split in test challenge; do
    if ! python evaluate.py --config "configs/$lang.yaml" --checkpoint "$checkpoint" --split "$split"; then
      status=1
    fi
  done
done
exit "$status"
