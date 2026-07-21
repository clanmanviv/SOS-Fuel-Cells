#!/usr/bin/env bash
# Reproduce the full pipeline end-to-end.
set -e
cd "$(dirname "$0")/src"
python generate_data.py
python train.py
echo "Done. See ../results/ for figures and metrics.json"
