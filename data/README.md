# Dataset files

Do not commit downloaded corpora here. The TSVs and images are distributed by
the WAT/ÚFAL dataset maintainers; follow the official download instructions
linked in the repository README, then use `scripts/prepare_data.sh` to place
the four split files in `data/<language>/`.

The loader consumes TSV files only. Each row must have seven tab-separated
columns in this order: `image_id`, `x`, `y`, `w`, `h`, English source text, and
target text. Image files are not read by this implementation; bounding-box
coordinates are used as spatial metadata.
