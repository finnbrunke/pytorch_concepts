#!/usr/bin/env bash
# One-time setup on the cluster login node, from the repository root:
#     bash cluster/setup_env.sh
# Creates a virtualenv in $CMR_ENV (default: $HOME/venvs/cmr_ts), installs the
# package with the data and tracking extras, and downloads the MIT-BIH records
# into $CMR_DATA (default: data/mitdb) so jobs never download concurrently.
set -euo pipefail

CMR_ENV=${CMR_ENV:-$HOME/venvs/cmr_ts}
CMR_DATA=${CMR_DATA:-data/mitdb}

# Euler (ETH): load a Python + CUDA stack and the proxy for internet access.
# Adjust or remove on other clusters (`module avail python` lists options).
if command -v module >/dev/null 2>&1; then
    module load stack/2024-06 python_cuda/3.11.6 2>/dev/null || true
    module load eth_proxy 2>/dev/null || true
fi

python -m venv "$CMR_ENV"
source "$CMR_ENV/bin/activate"
pip install --upgrade pip
pip install torch  # CUDA wheels on Linux x86_64
pip install -e ".[data,tracking,tests]"

python - "$CMR_DATA" <<'PY'
import importlib, sys
from pathlib import Path
from torch_concepts.data.datasets.mit_bih import MITBIHArrhythmiaDataset
mit = importlib.import_module("examples.high_level.13_mit_bih_cmr_ts_training")
for records in (mit.TRAIN_RECORDS, mit.TEST_RECORDS):
    data = MITBIHArrhythmiaDataset(root=Path(sys.argv[1]), records=records, downsample=4)
    print(f"{len(data)} windows from {len(records)} records")
PY

python -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"
echo "done. activate with: source $CMR_ENV/bin/activate"
