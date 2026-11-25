# Sourced by the job scripts. Override any variable before `sbatch`
# (e.g. `export CMR_ENV=...`); Slurm passes the environment on.
CMR_ENV=${CMR_ENV:-$HOME/venvs/cmr_ts}
CMR_DATA=${CMR_DATA:-data/mitdb}
CMR_LOGS=${CMR_LOGS:-${SCRATCH:-$PWD}/cmr_runs}
export CMR_ENV CMR_DATA CMR_LOGS

if command -v module >/dev/null 2>&1; then  # Euler (ETH); adjust elsewhere
    module load stack/2024-06 python_cuda/3.11.6 2>/dev/null || true
    module load eth_proxy 2>/dev/null || true  # needed for online wandb
fi
source "$CMR_ENV/bin/activate"
mkdir -p "$CMR_LOGS" cluster/logs
