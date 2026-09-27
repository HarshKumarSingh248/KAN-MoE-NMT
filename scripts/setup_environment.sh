#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v conda >/dev/null || { echo "conda is required; install Miniconda/Anaconda first." >&2; exit 1; }
conda env create -f environment.yml
echo "Environment created. Activate it with: conda activate kan-moe-nmt"
echo "Then install Indic NLP Library and its resources as documented in README.md."
echo "For advisor scoring, also obtain RIBES.py 1.02.4 as documented in docs/EVALUATION.md."
