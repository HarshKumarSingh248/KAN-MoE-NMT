#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
library_dir="$repo_root/third_party/indic_nlp_library"
resources_dir="$library_dir/indicnlp_resources"
library_revision=4cead0ae6c78fe9a19a51ef679f586206df9c476

command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
command -v python >/dev/null || { echo "Activate the Python environment first" >&2; exit 1; }

if [[ ! -e "$library_dir" ]]; then
    git clone https://github.com/anoopkunchukuttan/indic_nlp_library.git "$library_dir"
    git -C "$library_dir" checkout "$library_revision"
fi

if [[ ! -f "$library_dir/indicnlp/__init__.py" ]]; then
    echo "Expected Indic NLP Library source at $library_dir" >&2
    exit 1
fi
if [[ "$(git -C "$library_dir" rev-parse --show-toplevel)" != "$library_dir" ]]; then
    echo "Expected an independent Indic NLP Library git checkout at $library_dir" >&2
    exit 1
fi
if [[ "$(git -C "$library_dir" rev-parse HEAD)" != "$library_revision" ]]; then
    echo "Indic NLP Library at $library_dir is not at the recorded revision $library_revision" >&2
    exit 1
fi

if [[ ! -e "$resources_dir" ]]; then
    git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git "$resources_dir"
fi
if [[ ! -f "$resources_dir/README.md" ]]; then
    echo "Expected Indic NLP resources at $resources_dir" >&2
    exit 1
fi

python -m pip install "$library_dir"
echo "Indic NLP Library: $(git -C "$library_dir" rev-parse HEAD)"
if [[ "$(git -C "$resources_dir" rev-parse --show-toplevel)" == "$resources_dir" ]]; then
    echo "Indic NLP resources: $(git -C "$resources_dir" rev-parse HEAD)"
else
    echo "Indic NLP resources: local files (commit unavailable)"
fi
