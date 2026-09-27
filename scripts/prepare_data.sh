#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 {hindi|bengali|malayalam} EXTRACTED_DATA_DIRECTORY" >&2
  exit 2
fi
lang="$1"
source_dir="$(realpath "$2")"
case "$lang" in
  hindi|bengali|malayalam) ;;
  *) echo "Unsupported language: $lang" >&2; exit 2 ;;
esac
root="$(cd "$(dirname "$0")/.." && pwd)"
dest="$root/data/$lang"
prefix="${lang}-visual-genome"
mkdir -p "$dest"
for split in train dev test challenge-test-set; do
  file="$prefix-$split.txt"
  if [[ ! -s "$source_dir/$file" ]]; then
    echo "Missing expected TSV: $source_dir/$file" >&2
    exit 1
  fi
  install -m 0644 "$source_dir/$file" "$dest/$file"
done
echo "Prepared $lang TSV files in $dest"
echo "Verified rows:"
for split in train dev test challenge-test-set; do
  file="$dest/$prefix-$split.txt"
  printf '  %-20s %s\n' "$split" "$(wc -l < "$file")"
done
