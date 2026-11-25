#!/usr/bin/env bash
# Quick check that a GPU job sees CUDA and can run a short training:
#     sbatch cluster/check_gpu.sbatch
#SBATCH --job-name=cmr-check
#SBATCH --time=00:20:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=cluster/logs/%x-%j.out
set -euo pipefail
source cluster/env.sh

nvidia-smi || true
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
python -m examples.high_level.14_cmr_ts_interpretability --epochs 2 --n-repeats 1 --n-samples 2 \
    --log-dir "$CMR_LOGS/check" --data-root "$CMR_DATA" --no-download
