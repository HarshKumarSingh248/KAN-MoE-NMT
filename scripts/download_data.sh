#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {hindi|bengali|malayalam}" >&2
  exit 2
fi
case "$1" in
  hindi|bengali|malayalam) ;;
  *) echo "Unsupported language: $1" >&2; exit 2 ;;
esac
cat <<EOF
Dataset downloads are hosted by the WAT/ÚFAL maintainers. Open the official
task page and follow the $1 Visual Genome download link:

https://ufal.mff.cuni.cz/wat2025english-indicmultimodaltranslation

This script does not guess a LINDAT bitstream URL or bypass any access terms.
After downloading and extracting the archive, run:
  bash scripts/prepare_data.sh $1 /path/to/extracted/dataset
EOF
