"""Small end-to-end training example for learned temporal CMR rules.

Run from the repository root:

    python -m examples.high_level.12_cmr_ts_training
    python -m examples.high_level.12_cmr_ts_training --wandb            # log to W&B
    python -m examples.high_level.12_cmr_ts_training --wandb --offline  # sync later

The example generates predator-prey data, learns one-clause SNF rules, and
applies them to held-out sequences. Without ``--wandb``, metrics are written as
CSV and figures as PNG files under ``lightning_logs/``.
"""

import argparse

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, Subset

from torch_concepts import seed_everything
from torch_concepts.data.datasets.predator_prey import PredatorPreyDataset
from torch_concepts.nn import (
    ConceptMemoryReasonerTS,
    SNFConfig,
    TemporalCMRLearner,
    TemporalCMRMonitor,
    format_rule,
)


def make_logger(args, name: str):
    log_dir = getattr(args, "log_dir", "lightning_logs")
    if not args.wandb:
        return CSVLogger(log_dir, name=name)
    try:
        from pytorch_lightning.loggers import WandbLogger
    except ImportError as exc:  # pragma: no cover
        raise ImportError("--wandb requires: pip install wandb") from exc
    return WandbLogger(
        project=args.wandb_project, name=name, offline=args.offline,
        save_dir=log_dir,  # keeps ./wandb out of the repository root
    )


def print_summary(learner: TemporalCMRLearner) -> None:
    print("\nlearned rules:")
    for rule in learner.model.rule_decoder.export_rules():
        print(f"  {rule['task']:<24} {format_rule(rule)}")
    print("\ntest metrics (majority-class accuracy in brackets):")
    for task, values in learner.test_metrics.items():
        auroc = f"{values['auroc']:.3f}" if "auroc" in values else "  n/a"
        print(
            f"  {task:<24} acc={values['acc']:.3f} [{values['majority_acc']:.3f}]"
            f"  f1={values['f1']:.3f}  auroc={auroc}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    parser.add_argument("--wandb-project", default="cmr-ts")
    parser.add_argument("--offline", action="store_true", help="W&B offline mode")
    parser.add_argument("--plot-every", type=int, default=5, help="figure frequency (epochs)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(7)

    dataset = PredatorPreyDataset(num_samples=120, time_steps=40, dt=0.1)
    train_end, val_end = 80, 100
    loader = dict(collate_fn=dataset.collate)
    train_loader = DataLoader(Subset(dataset, range(train_end)), batch_size=16, shuffle=True, **loader)
    val_loader = DataLoader(Subset(dataset, range(train_end, val_end)), batch_size=20, **loader)
    test_loader = DataLoader(Subset(dataset, range(val_end, len(dataset))), batch_size=20, **loader)

    model = ConceptMemoryReasonerTS(
        input_size=2,
        annotations=dataset.concept_annotations,
        task_names=["prey_recovery", "predator_crash", "system_stability"],
        n_rules=2,
        latent_size=16,
        encoder_kind="gru",
        snf_config=SNFConfig(max_horizon=40),
    )
    learner = TemporalCMRLearner(model, learning_rate=3e-3, concept_weight=1.0)
    trainer = Trainer(
        max_epochs=args.epochs,
        logger=make_logger(args, "predator_prey"),
        callbacks=[TemporalCMRMonitor(every_n_epochs=args.plot_every)],
        log_every_n_steps=1,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    trainer.fit(learner, train_loader, val_loader)
    trainer.test(learner, test_loader, verbose=False)
    print_summary(learner)
    print(f"\nlogs and figures: {trainer.log_dir or 'W&B run'}")


if __name__ == "__main__":
    main()
