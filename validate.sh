#!/usr/bin/env bash
# Data/order/configuration tests only; no training, backward pass, or GPU access.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$project_dir"
python_bin="${PYTHON:-python3}"
"$python_bin" -m unittest discover -s tests -v
"$python_bin" -m compileall -q model.py data.py canonical.py train.py plot.py hardware.py tests
bash -n run.sh validate.sh
