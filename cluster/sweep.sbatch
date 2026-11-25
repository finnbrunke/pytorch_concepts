#!/usr/bin/env bash
# Sweep of example 14 as a job array, one configuration per task:
#
#     sbatch cluster/sweep.sbatch                 # all 48 configurations
#     sbatch --array=0-23 cluster/sweep.sbatch    # predator-prey only
#     WANDB=1 sbatch cluster/sweep.sbatch         # also log to W&B (wandb login first)
#
# Grid: dataset x {soft, straight_through} x {joint, independent}
#       x {one template, all templates} x 3 seeds.
# Each task writes $CMR_LOGS/<run>/.../report.json;
# summarize with: python cluster/collect.py "$CMR_LOGS"
#SBATCH --job-name=cmr-sweep
#SBATCH --array=0-47
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --output=cluster/logs/%x-%A_%a.out
set -euo pipefail
source cluster/env.sh

DATASETS=(predator_prey mit_bih)
DISCRETE=(soft straight_through)
SOURCES=(predicted ground_truth)
TEMPLATES=(one all)
SEEDS=(0 1 2)

i=${SLURM_ARRAY_TASK_ID:-0}
seed=${SEEDS[i % 3]};            i=$((i / 3))
templates=${TEMPLATES[i % 2]};   i=$((i / 2))
source=${SOURCES[i % 2]};        i=$((i / 2))
discrete=${DISCRETE[i % 2]};     i=$((i / 2))
dataset=${DATASETS[i % 2]}

args=(--dataset "$dataset" --discrete "$discrete" --concept-source "$source"
      --seed "$seed" --bidirectional --latent-size 32
      --log-dir "$CMR_LOGS" --data-root "$CMR_DATA" --no-download)
[[ $templates == all ]] && args+=(--all-templates)
if [[ $dataset == mit_bih ]]; then
    args+=(--epochs "${MIT_EPOCHS:-30}" --pos-weight)
else
    args+=(--epochs "${PP_EPOCHS:-60}")
fi
[[ ${WANDB:-0} == 1 ]] && args+=(--wandb --wandb-project "${WANDB_PROJECT:-cmr-ts}")

run="${dataset}_${discrete}_${source}_${templates}_s${seed}"
echo "task ${SLURM_ARRAY_TASK_ID:-0}: $run"
python -m examples.high_level.14_cmr_ts_interpretability --run-name "$run" "${args[@]}"
