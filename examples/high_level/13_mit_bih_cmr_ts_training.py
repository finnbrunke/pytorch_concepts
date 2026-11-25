"""Train learned temporal CMR rules on MIT-BIH Arrhythmia data.

Run from the repository root:

    python -m examples.high_level.13_mit_bih_cmr_ts_training
    python -m examples.high_level.13_mit_bih_cmr_ts_training --wandb

The first run downloads the selected PhysioNet records through WFDB. Without
``--wandb``, metrics are written as CSV and figures as PNG files under
``lightning_logs/``.
"""

import argparse
import importlib
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Subset

from torch_concepts import seed_everything
from torch_concepts.data.datasets.mit_bih import MITBIHArrhythmiaDataset
from torch_concepts.nn import (
    ConceptMemoryReasonerTS,
    SNFConfig,
    TemporalCMRLearner,
    TemporalCMRMonitor,
)

# Reuse the logger and summary helpers of the predator-prey example.
_shared = importlib.import_module("examples.high_level.12_cmr_ts_training")

# Inter-patient DS1/DS2 split (de Chazal et al., 2004); paced records are
# excluded, so no patient appears in both sets.
TRAIN_RECORDS = (
    "101 106 108 109 112 114 115 116 118 119 122 "
    "124 201 203 205 207 208 209 215 220 223 230"
).split()
TEST_RECORDS = (
    "100 103 105 111 113 117 121 123 200 202 210 "
    "212 213 214 219 221 222 228 231 232 233 234"
).split()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-windows", type=int, default=2048)
    parser.add_argument("--val-windows", type=int, default=512)
    parser.add_argument("--test-windows", type=int, default=1024)
    parser.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    parser.add_argument("--wandb-project", default="cmr-ts")
    parser.add_argument("--offline", action="store_true", help="W&B offline mode")
    parser.add_argument("--plot-every", type=int, default=2, help="figure frequency (epochs)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(7)
    # 4 s windows at 90 Hz (downsample 4) keep traces at 360 steps.
    options = dict(root=Path("data/mitdb"), window_seconds=4.0, downsample=4, download=True)
    train_dataset = MITBIHArrhythmiaDataset(records=TRAIN_RECORDS, **options)
    test_dataset = MITBIHArrhythmiaDataset(records=TEST_RECORDS, **options)

    # Windows are stored record by record: sample them at random. Validation
    # windows come from the training patients; the test set is DS2.
    order = torch.randperm(len(train_dataset), generator=torch.Generator().manual_seed(0))
    train_indices = order[: args.train_windows].tolist()
    val_indices = order[args.train_windows: args.train_windows + args.val_windows].tolist()
    test_order = torch.randperm(len(test_dataset), generator=torch.Generator().manual_seed(1))
    loader = dict(batch_size=32, collate_fn=train_dataset.collate)
    train_loader = DataLoader(Subset(train_dataset, train_indices), shuffle=True, **loader)
    val_loader = DataLoader(Subset(train_dataset, val_indices), **loader)
    test_loader = DataLoader(
        Subset(test_dataset, test_order[: args.test_windows].tolist()), **loader
    )

    model = ConceptMemoryReasonerTS(
        input_size=1,
        annotations=train_dataset.concept_annotations,
        task_names=MITBIHArrhythmiaDataset.task_names,
        n_rules=2,
        latent_size=16,
        encoder_kind="gru",
        snf_config=SNFConfig(max_horizon=360),
    )
    learner = TemporalCMRLearner(model)
    trainer = Trainer(
        max_epochs=args.epochs,
        logger=_shared.make_logger(args, "mit_bih"),
        callbacks=[TemporalCMRMonitor(every_n_epochs=args.plot_every)],
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    trainer.fit(learner, train_loader, val_loader)
    trainer.test(learner, test_loader, verbose=False)
    _shared.print_summary(learner)
    print(f"\nlogs and figures: {trainer.log_dir or 'W&B run'}")


if __name__ == "__main__":
    main()
