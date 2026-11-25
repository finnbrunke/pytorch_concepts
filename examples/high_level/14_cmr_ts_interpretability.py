"""Interpretability metrics for a temporal CMR.

Trains a temporal CMR with learned SNF rules and a concept-free black box with
the same encoder, then reports on the test set:

* task and concept scores (per step and per event),
* concept completeness (CMR vs. black box) and rule-language completeness,
* concept intervention curves (whole traces and time steps),
* rule interventions, rule fidelity of the exported rules, rule compactness,

each for the soft model, its most likely discrete rulebook ("argmax") and
rulebooks sampled from the fuzzy concept and selection distributions.

Run from the repository root:

    python -m examples.high_level.14_cmr_ts_interpretability
    python -m examples.high_level.14_cmr_ts_interpretability --dataset mit_bih
    python -m examples.high_level.14_cmr_ts_interpretability --concept-source ground_truth

``cluster/sweep.sbatch`` runs a grid of these settings as a Slurm job array.

The full report is saved as ``report.json`` in the run's log directory.
"""

import argparse
import importlib
import json
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, Subset

from torch_concepts import seed_everything
from torch_concepts.nn import (
    ConceptMemoryReasonerTS,
    SequenceBlackBox,
    SNFConfig,
    TemporalCMRLearner,
    TemporalCMRMonitor,
    TemporalInterpretabilityReport,
    concept_pos_weight,
    format_rule,
)
from torch_concepts.nn.modules.high.rules import TEMPLATES

_shared = importlib.import_module("examples.high_level.12_cmr_ts_training")


def predator_prey_data(args):
    from torch_concepts.data.datasets.predator_prey import PredatorPreyDataset

    dataset = PredatorPreyDataset(num_samples=200, time_steps=40, dt=0.1)
    loader = dict(collate_fn=dataset.collate, batch_size=20)
    splits = (range(0, 120), range(120, 160), range(160, 200))
    loaders = [DataLoader(Subset(dataset, s), shuffle=i == 0, **loader) for i, s in enumerate(splits)]
    # The predator-prey labels use slightly different window semantics than
    # STLRuleDecoder, so its rulebook is not an exact reference.
    task_names = list(dict.fromkeys(rule["task"] for rule in dataset.rulebook))
    return dataset.concept_annotations, task_names, 2, 40, loaders, None


def mit_bih_data(args):
    from torch_concepts.data.datasets.mit_bih import MITBIHArrhythmiaDataset

    mit = importlib.import_module("examples.high_level.13_mit_bih_cmr_ts_training")
    options = dict(root=Path(args.data_root), window_seconds=4.0, downsample=4,
                   download=not args.no_download)
    train = MITBIHArrhythmiaDataset(records=mit.TRAIN_RECORDS, **options)
    test = MITBIHArrhythmiaDataset(records=mit.TEST_RECORDS, **options)
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(0)).tolist()
    test_order = torch.randperm(len(test), generator=torch.Generator().manual_seed(1)).tolist()
    loader = dict(collate_fn=train.collate, batch_size=32)
    loaders = [
        DataLoader(Subset(train, order[:2048]), shuffle=True, **loader),
        DataLoader(Subset(train, order[2048:2560]), **loader),
        DataLoader(Subset(test, test_order[:512]), **loader),
    ]
    return (train.concept_annotations, list(train.task_names), 1, 360, loaders,
            train.rulebook)


def fit(model, args, loaders, name, callbacks=(), **learner_options):
    learner = TemporalCMRLearner(model, **learner_options)
    trainer = Trainer(
        max_epochs=args.epochs,
        logger=_shared.make_logger(args, f"{args.run_name}_{name}" if args.run_name else name),
        callbacks=list(callbacks),
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    trainer.fit(learner, loaders[0], loaders[1])
    return learner, trainer


def print_report(report: dict) -> None:
    def line(label, value):
        print(f"  {label:<46} {value}")

    def fmt(row, keys):
        return "  ".join(f"{k}={row[k]:.3f}" for k in keys if k in row and row[k] == row[k])

    print("\n== tasks (test AUROC)")
    for task, row in report["tasks"].items():
        line(task, f"{row['score']:.3f}")
    print("\n== concepts")
    for name, row in report["concepts"].items():
        line(name, fmt(row, ("auroc", "f1", "event_f1", "event_recall", "mse", "r2")))
    if "completeness" in report:
        print("\n== concept completeness (CMR vs. black box)")
        for task, row in report["completeness"].items():
            line(task, fmt(row, ("whitebox", "reference", "completeness", "normalized_completeness")))
    print("\n== rule-language completeness (rules on true concepts)")
    for task, row in report["rule_language_completeness"].items():
        line(task, fmt(row, ("whitebox", "reference", "completeness", "normalized_completeness")))
    print("\n== concept interventions (macro task AUROC)")
    for name, curve in report["interventions"]["curves"].items():
        points = " ".join(f"{s:.3f}" for s in curve["score"])
        line(name, f"area={curve['auc']:.3f} gain={curve['gain']:+.3f} | {points}")
    effects = report["interventions"]["single_concept_effects"]
    line("single-concept effects", " ".join(f"{k}={v:+.3f}" for k, v in effects.items()))
    if "rules" in report:
        print("\n== rule interventions")
        for task, row in report["rules"]["interventions"].items():
            forced = " ".join(f"{s:.3f}" for s in row["forced"])
            line(task, f"selected={row['selected']:.3f} forced=[{forced}] gain={row['gain']:+.3f}")
        print("\n== rule fidelity (exported discrete rules vs. soft model)")
        for source in ("predicted", "ground_truth"):
            row = report["rules"][f"fidelity_{source}"]["macro"]
            line(f"{source} concepts", fmt(row, (
                "top_rule_agreement", "soft_selection_agreement", "model_score",
                "top_rule_score")))
        print("\n== rule compactness")
        for task, row in report["rules"]["compactness"].items():
            line(task, fmt(row, ("concept_entropy", "selection_entropy", "rules_used",
                                 "duplicate_rules")))


def print_comparison(reports: dict) -> None:
    """Key numbers side by side for the soft, argmax and sampled rules."""
    def get(report, *path):
        node = report
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return None
            node = node[key]
        return node

    rows = [
        ("task AUROC (macro)", ("tasks", "macro", "score")),
        ("concept completeness", ("completeness", "macro", "completeness")),
        ("rule-language completeness", ("rule_language_completeness", "macro", "completeness")),
        ("intervention area, whole traces", ("interventions", "curves", "concepts_random", "auc")),
        ("intervention gain, whole traces", ("interventions", "curves", "concepts_random", "gain")),
        ("intervention gain, time steps", ("interventions", "curves", "time_random", "gain")),
    ]
    modes = list(reports)
    print("\n== soft vs. discrete rules" + "".join(f"{m:>16}" for m in modes))
    for label, path in rows:
        cells = []
        for mode in modes:
            value = get(reports[mode], *path)
            std = reports[mode].get("sample_std", {}).get("/".join(path))
            cell = "-" if value is None else f"{value:.3f}"
            cells.append(cell + (f"±{std:.3f}" if std is not None else ""))
        print(f"  {label:<34}" + "".join(f"{c:>16}" for c in cells))
    if "argmax" in reports:
        print("\n== argmax rulebook")
        for rule in reports["argmax"]["rulebook"]:
            print(f"  {rule['task']:<24} {format_rule(rule)}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=("predator_prey", "mit_bih"), default="predator_prey")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--concept-source", choices=("predicted", "ground_truth"),
                        default="predicted", help="joint or independent training of the rules")
    parser.add_argument("--discrete", choices=("soft", "straight_through"), default="soft",
                        help="train with soft or straight-through discrete rules and selection")
    parser.add_argument("--all-templates", action="store_true",
                        help="let rules choose G(A->F B), F(A and F B) or G(not A)")
    parser.add_argument("--bidirectional", action="store_true", help="bidirectional GRU encoder")
    parser.add_argument("--latent-size", type=int, default=16)
    parser.add_argument("--encoder-layers", type=int, default=1)
    parser.add_argument("--pos-weight", action="store_true",
                        help="up-weight rare binary concept steps in the concept loss")
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=5, help="rulebooks drawn in 'sample' mode")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-dir", default="lightning_logs", help="where logs and report.json go")
    parser.add_argument("--run-name", default=None, help="prefix of the logger run names")
    parser.add_argument("--data-root", default="data/mitdb", help="MIT-BIH record directory")
    parser.add_argument("--no-download", action="store_true",
                        help="fail instead of downloading missing MIT-BIH records")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="cmr-ts")
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    data = predator_prey_data if args.dataset == "predator_prey" else mit_bih_data
    annotations, task_names, input_size, horizon, loaders, rulebook = data(args)
    encoder = dict(latent_size=args.latent_size, encoder_kind="gru",
                   encoder_layers=args.encoder_layers, encoder_bidirectional=args.bidirectional)
    templates = TEMPLATES if args.all_templates else ("always_implies_eventually",)

    blackbox, _ = fit(SequenceBlackBox(input_size, task_names, **encoder), args, loaders,
                      f"{args.dataset}_blackbox")
    report = TemporalInterpretabilityReport(
        blackbox=blackbox.model, reference_rules=rulebook,
        rule_modes=("soft", "argmax", "sample"),
        n_repeats=args.n_repeats, n_samples=args.n_samples,
    )
    model = ConceptMemoryReasonerTS(
        input_size=input_size, annotations=annotations, task_names=task_names, n_rules=2,
        snf_config=SNFConfig(max_horizon=horizon, templates=templates, discrete=args.discrete),
        selection=args.discrete, **encoder,
    )
    learner, trainer = fit(
        model, args, loaders, f"{args.dataset}_cmr",
        callbacks=[TemporalCMRMonitor(every_n_epochs=max(1, args.epochs // 4)), report],
        concept_source=args.concept_source,
        concept_pos_weight=concept_pos_weight(loaders[0]) if args.pos_weight else None,
    )
    trainer.test(learner, loaders[2], verbose=False)
    print("\n######## soft rules (the trained model as is)")
    print_report(report.report["soft"])
    print_comparison(report.report)

    if isinstance(trainer.logger, CSVLogger):  # <log-dir>/<run>/version_N
        out = Path(trainer.logger.log_dir) / "report.json"
    else:  # W&B shares its save_dir between runs
        run = f"{args.run_name}_{args.dataset}_cmr" if args.run_name else f"{args.dataset}_cmr"
        out = Path(args.log_dir) / run / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.report, indent=2, default=float))
    print(f"\nfull report: {out}")


if __name__ == "__main__":
    main()
