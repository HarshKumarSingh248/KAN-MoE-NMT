#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python evaluate.py --config configs/hindi.yaml --checkpoint "${1:-checkpoints/hindi/best_model2.pt}" --split "${2:-test}" "${@:3}"
