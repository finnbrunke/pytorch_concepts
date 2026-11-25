# Running the temporal CMR experiments on a Slurm cluster

From the repository root on the login node:

```bash
bash cluster/setup_env.sh            # venv, package install, MIT-BIH download (once)
sbatch cluster/check_gpu.sbatch      # 2-epoch smoke test on one GPU
sbatch cluster/sweep.sbatch          # 48-task grid of example 14
python cluster/collect.py "$SCRATCH/cmr_runs"   # mean ± std over seeds
```

- Paths are set in `cluster/env.sh`: `CMR_ENV` (venv), `CMR_DATA` (MIT-BIH
  records) and `CMR_LOGS` (default `$SCRATCH/cmr_runs`). Export any of them
  before `sbatch` to override.
- The `module load` lines are for ETH Euler; adjust them on other clusters.
- W&B: run `wandb login` once, then `WANDB=1 sbatch cluster/sweep.sbatch`.
  Without network access on compute nodes, add `--offline` in
  `sweep.sbatch` and later run `wandb sync` on the log directory.
- Epochs: `PP_EPOCHS` (default 60) and `MIT_EPOCHS` (default 30).
- Job output goes to `cluster/logs/`.
