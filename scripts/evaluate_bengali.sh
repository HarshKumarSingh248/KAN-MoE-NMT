#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python evaluate.py --config configs/bengali.yaml --checkpoint "${1:-checkpoints/bengali/best_model2.pt}" --split "${2:-test}" "${@:3}"
