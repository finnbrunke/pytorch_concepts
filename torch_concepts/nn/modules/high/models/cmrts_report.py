"""One call that runs every interpretability metric for a temporal CMR."""

from typing import Dict, Iterable, List, Optional, Sequence

import torch

from . import cmrts_evaluation as evaluation
from . import cmrts_interventions as interventions


def interpretability_report(
    model,
    dataloader: Iterable,
    blackbox: Optional[torch.nn.Module] = None,
    reference_rules: Optional[list] = None,
    scorer: str = "auroc",
    n_repeats: int = 3,
    event_tolerance: int = 0,
    time_fractions: Sequence[float] = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
    rule_mode: str = "soft",
    n_samples: int = 5,
    seed: int = 0,
) -> Dict[str, object]:
    """Interpretability metrics of ``model`` on ``dataloader``.

    ``rule_mode`` sets which model the metrics describe:

    * ``"soft"``: the trained model as is (fuzzy concept choice in the
      rules, soft rule selection);
    * ``"argmax"``: the single discrete rulebook a reader would see (most
      likely concepts, integer intervals, exact STL, top-selected rule per
      sample), via :func:`discrete_rules`;
    * ``"sample"``: ``n_samples`` rulebooks drawn from the fuzzy concept
      distributions, with the rule per sample drawn from the selection
      distribution; numbers are averaged over the draws and
      ``report["sample_std"]`` holds the standard deviation of every scalar.

    The discrete modes add ``"rulebook"`` (or ``"rulebooks"``) and skip rule
    fidelity, which compares the soft model with its discrete version.

    Sections (rule sections only for a learned SNF bank):

    * ``tasks``: per-task scores of the model;
    * ``concepts``: :func:`concept_scores` (per-step and event-level);
    * ``completeness``: :func:`concept_completeness` against ``blackbox``
      (a trained :class:`SequenceBlackBox`), when given;
    * ``rule_language_completeness``: against ``reference_rules`` or a
      perfect score;
    * ``interventions``: concept intervention curves (whole traces, by count;
      time steps, by fraction; random and uncertainty order) and
      :func:`single_concept_effects`;
    * ``rules``: :func:`rule_intervention_scores`, :func:`rule_fidelity` on
      predicted and on true concepts, and :func:`rule_compactness`.
    """
    from ..rules.snf import SNFRuleDecoder
    from .cmrts import discrete_rules

    options = dict(
        blackbox=blackbox, reference_rules=reference_rules, scorer=scorer,
        n_repeats=n_repeats, event_tolerance=event_tolerance, time_fractions=time_fractions,
    )
    if rule_mode == "soft":
        return _report(model, dataloader, fidelity=True, **options)
    if not isinstance(model.rule_decoder, SNFRuleDecoder):
        raise TypeError("Discrete rule modes need a model with a learned SNF rule bank.")
    if rule_mode == "argmax":
        with discrete_rules(model, "argmax", "top") as rulebook:
            report = _report(model, dataloader, fidelity=False, **options)
        report["rulebook"] = rulebook
        return report
    if rule_mode != "sample":
        raise ValueError("rule_mode must be 'soft', 'argmax' or 'sample'.")
    reports, rulebooks = [], []
    for draw in range(n_samples):
        with discrete_rules(model, "sample", "sample", seed=seed + draw) as rulebook:
            reports.append(_report(model, dataloader, fidelity=False, **options))
        rulebooks.append(rulebook)
    report = average_reports(reports)
    flat = [flatten_report(r, prefix="") for r in reports]
    report["sample_std"] = {
        key: float(torch.tensor([f[key] for f in flat if key in f]).std(unbiased=False))
        for key in flat[0]
    }
    report["rulebooks"] = rulebooks
    return report


def average_reports(reports: List[Dict[str, object]]) -> Dict[str, object]:
    """Element-wise mean of reports with the same structure."""
    first = reports[0]
    if isinstance(first, dict):
        return {
            key: average_reports([r[key] for r in reports])
            for key in first if all(isinstance(r, dict) and key in r for r in reports)
        }
    if isinstance(first, (list, tuple)) and first and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in first
    ):
        return [sum(values) / len(values) for values in zip(*reports)]
    if isinstance(first, (int, float)) and not isinstance(first, bool):
        return sum(reports) / len(reports)
    return first


def _report(model, dataloader, blackbox, reference_rules, scorer, n_repeats,
            event_tolerance, time_fractions, fidelity: bool) -> Dict[str, object]:
    from ..rules.snf import SNFRuleDecoder

    predictions = evaluation.collect_predictions(model, dataloader, keep_concepts=True)
    report: Dict[str, object] = {
        "tasks": evaluation._completeness_table(
            predictions.probabilities, None, predictions.targets, model.task_names,
            scorer, reference_score=1.0,
        ),
        "concepts": evaluation.concept_scores(model, predictions, event_tolerance),
    }
    report["tasks"] = {
        task: {"score": row["whitebox"]} for task, row in report["tasks"].items()
    }
    if blackbox is not None:
        report["completeness"] = evaluation.concept_completeness(
            predictions,
            evaluation.collect_predictions(blackbox, dataloader),
            model.task_names,
            scorer,
        )
    report["rule_language_completeness"] = evaluation.rule_language_completeness(
        model, dataloader, reference_rules, scorer
    )
    curves = {}
    for order in ("random", "uncertainty"):
        curves[f"concepts_{order}"] = interventions.concept_intervention_curve(
            model, dataloader, order=order, n_repeats=n_repeats, scorer=scorer
        )
        curves[f"time_{order}"] = interventions.time_intervention_curve(
            model, dataloader, time_fractions, order=order, n_repeats=n_repeats, scorer=scorer
        )
    report["interventions"] = {
        "curves": curves,
        "single_concept_effects": interventions.single_concept_effects(model, dataloader, scorer),
    }
    if isinstance(model.rule_decoder, SNFRuleDecoder):
        rules = {
            "interventions": interventions.rule_intervention_scores(model, dataloader, scorer),
            "compactness": evaluation.rule_compactness(model, dataloader),
        }
        if fidelity:
            for source in ("predicted", "ground_truth"):
                rules[f"fidelity_{source}"] = evaluation.rule_fidelity(
                    model, dataloader, source, scorer
                )
        report["rules"] = rules
    return report


def flatten_report(report: Dict[str, object], prefix: str = "interp") -> Dict[str, float]:
    """Scalar entries of a report as ``{"interp/section/.../name": value}``.

    Lists (curve points, per-rule scores) and NaNs are left out; a curve
    contributes its ``auc`` and ``gain``.
    """
    flat: Dict[str, float] = {}

    def walk(node, path):
        if path.endswith(("/rulebook", "/rulebooks", "/sample_std")):
            return
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}/{key}")  # lists (curve points) are not scalars
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            if node == node:  # not NaN
                flat[path] = float(node)

    walk(report, prefix)
    return {key.lstrip("/"): value for key, value in flat.items()}
