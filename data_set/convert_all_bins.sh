#!/usr/bin/env bash
# Convert LFW, CFP-FP, and AgeDB-30 .bin files to images. Use quoted paths so
# names like CFP-FP and AgeDB-30 are not split by the shell.
# Run from project root: bash data_set/convert_all_bins.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
PY="${ROOT}/.venv/bin/python"
CONVERT="${SCRIPT_DIR}/load_images_from_bin.py"

"$PY" "$CONVERT" --mode bin \
  --bin "$SCRIPT_DIR/lfw.bin" \
  --save_bin "$SCRIPT_DIR/LFW/lfw_align_112" \
  --pair_file pairs.txt

"$PY" "$CONVERT" --mode bin \
  --bin "$SCRIPT_DIR/cfp_fp.bin" \
  --save_bin "$SCRIPT_DIR/CFP-FP/CFP_FP_aligned_112" \
  --pair_file cfp_fp_pair.txt

"$PY" "$CONVERT" --mode bin \
  --bin "$SCRIPT_DIR/agedb_30.bin" \
  --save_bin "$SCRIPT_DIR/AgeDB-30/agedb30_align_112" \
  --pair_file agedb_30_pair.txt

echo "Done. Eval datasets: LFW, CFP-FP, AgeDB-30."
