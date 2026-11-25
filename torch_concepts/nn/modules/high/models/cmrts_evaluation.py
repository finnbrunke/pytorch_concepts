"""Interpretability metrics for temporal CMR models.

Completeness (this module, step 1):

* :func:`concept_completeness` compares a concept-based model with a
  concept-free black box trained on the same data. It reports the library's
  ratio ``score(whitebox) / score(blackbox)``
  (:func:`torch_concepts.nn.functional.completeness_score`) and the
  normalised form of Yeh et al. (2020),
  ``(score(whitebox) - score(random)) / (score(blackbox) - score(random))``,
  which does not reward a weak black box.
* :func:`rule_language_completeness` asks how much the rule *language* loses:
  tasks are decoded from ground-truth concept traces, so concept errors play
  no part, and compared with a reference (by default a perfect score, since
  the labels of rule-generated datasets are a function of the true concepts).
"""

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from ....functional import completeness_score


def _auroc(y_true, y_score, average=None):
    return roc_auc_score(y_true, y_score)


def _accuracy(y_true, y_score, average=None):
    return accuracy_score(y_true, y_score >= 0.5)


SCORERS: Dict[str, Callable] = {"auroc": _auroc, "accuracy": _accuracy}


@dataclass
class Predictions:
    """Task probabilities and targets collected over a dataloader."""

    probabilities: torch.Tensor  # (n, tasks)
    targets: torch.Tensor  # (n, tasks), 0/1
    concepts: Optional[torch.Tensor] = None  # (n, time, concepts), predicted
    true_concepts: Optional[torch.Tensor] = None  # (n, time, concepts)


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    dataloader: Iterable,
    concept_source: str = "predicted",
    keep_concepts: bool = False,
) -> Predictions:
    """Run ``model`` over ``dataloader`` in eval mode.

    Args:
        model: :class:`ConceptMemoryReasonerTS` or :class:`SequenceBlackBox`.
        concept_source: ``"predicted"`` or ``"ground_truth"`` (decode tasks
            from the true concept traces).
        keep_concepts: Also return predicted and true concept traces.
    """
    from .cmrts_learner import TemporalCMRLearner

    if concept_source not in {"predicted", "ground_truth"}:
        raise ValueError("concept_source must be 'predicted' or 'ground_truth'.")
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    probabilities, targets, concepts, true_concepts = [], [], [], []
    for batch in dataloader:
        inputs, true, target = (t.to(device) for t in TemporalCMRLearner.unpack(batch))
        values = true if concept_source == "ground_truth" else None
        output = model(inputs, concept_values=values)
        probabilities.append(output.tasks.tensor.cpu())
        targets.append(target.cpu())
        if keep_concepts and output.concepts is not None:
            concepts.append(output.concepts.tensor.cpu())
            true_concepts.append(true.cpu())
    model.train(was_training)
    return Predictions(
        torch.cat(probabilities),
        torch.cat(targets),
        torch.cat(concepts) if concepts else None,
        torch.cat(true_concepts) if true_concepts else None,
    )


def _random_score(targets: torch.Tensor, scorer: str) -> float:
    if scorer == "auroc":
        return 0.5
    rate = float(targets.float().mean())
    return max(rate, 1.0 - rate)  # best constant predictor


def _completeness_table(
    whitebox: torch.Tensor,
    reference: Optional[torch.Tensor],
    targets: torch.Tensor,
    task_names: List[str],
    scorer: str,
    reference_score: Optional[float] = None,
) -> Dict[str, Dict[str, float]]:
    if scorer not in SCORERS:
        raise ValueError(f"scorer must be one of {sorted(SCORERS)}.")
    score = SCORERS[scorer]
    table: Dict[str, Dict[str, float]] = {}
    for index, task in enumerate(task_names):
        y = targets[:, index]
        if y.min() == y.max():
            continue  # a single class: AUROC undefined, accuracy uninformative
        white = whitebox[:, index]
        if reference is not None:
            ratio = completeness_score(y, reference[:, index], white, scorer=score)
            black_score = float(score(y.numpy(), reference[:, index].numpy()))
        else:
            black_score = float(reference_score)
            ratio = float(score(y.numpy(), white.numpy())) / (black_score + 1e-10)
        white_score = float(score(y.numpy(), white.numpy()))
        random = _random_score(y, scorer)
        span = black_score - random
        table[task] = {
            "whitebox": white_score,
            "reference": black_score,
            "random": random,
            "completeness": float(ratio),
            "normalized_completeness": (
                (white_score - random) / span if abs(span) > 1e-8 else float("nan")
            ),
        }
    if table:
        table["macro"] = {
            key: sum(row[key] for row in table.values()) / len(table)
            for key in next(iter(table.values()))
        }
    return table


def concept_completeness(
    whitebox: Predictions,
    blackbox: Predictions,
    task_names: List[str],
    scorer: str = "auroc",
) -> Dict[str, Dict[str, float]]:
    """Concept completeness of a concept-based model w.r.t. a black box.

    Both prediction sets must come from the same samples in the same order.
    Tasks with a single class in the evaluation data are skipped.

    Returns:
        ``{task: {"whitebox", "reference", "random", "completeness",
        "normalized_completeness"}, ..., "macro": {...}}`` where
        ``reference`` is the black-box score.
    """
    if not torch.equal(whitebox.targets, blackbox.targets):
        raise ValueError("whitebox and blackbox predictions must share their targets.")
    return _completeness_table(
        whitebox.probabilities, blackbox.probabilities, whitebox.targets, task_names, scorer
    )


def rule_language_completeness(
    model: torch.nn.Module,
    dataloader: Iterable,
    reference_rules: Optional[list] = None,
    scorer: str = "auroc",
) -> Dict[str, Dict[str, float]]:
    """How well the model's rules explain the tasks given *true* concepts.

    Tasks are decoded from ground-truth concept traces (the rule selection
    still comes from the inputs), so only the rule language and the learned
    rules limit the score. The reference is the rulebook ``reference_rules``
    evaluated exactly (hard STL) on the same traces, or a perfect score when
    no rulebook is given.

    A value well below 1 means the rules the model can express (or has
    learned) do not capture the labelling rules, independently of how well the
    concepts are predicted.
    """
    from .cmrts import STLRuleDecoder

    predictions = collect_predictions(model, dataloader, "ground_truth", keep_concepts=True)
    task_names = model.task_names
    reference = None
    reference_score = None
    if reference_rules is not None:
        decoder = STLRuleDecoder(
            model.annotations, task_names, reference_rules, smoothness=None
        )
        rules_per_task = len(reference_rules) // len(task_names)
        from .....tensor import AnnotatedTensor

        true = AnnotatedTensor(
            predictions.true_concepts, model.annotations.to_concept_space(), axis=-1
        )
        selection = torch.full(
            (true.shape[0], len(task_names), rules_per_task), 1.0 / rules_per_task
        )
        _, robustness, _ = decoder(true, selection)
        # Hard rule satisfaction: rank by robustness (AUROC) or sign (accuracy).
        reference = torch.sigmoid(1e3 * robustness).mean(-1)
    else:
        reference_score = 1.0
    return _completeness_table(
        predictions.probabilities, reference, predictions.targets, task_names, scorer,
        reference_score=reference_score,
    )


# ---------------------------------------------------------------------------
# Rule fidelity
# ---------------------------------------------------------------------------


def exported_rule_specs(model) -> list:
    """The learned SNF bank as one discrete rulebook (most likely concepts).

    Intervals are ``[round(start), round(end)]``, matching the soft gates.
    """
    return model.rule_decoder.discrete_specs("argmax")


def rule_fidelity(
    model,
    dataloader: Iterable,
    concept_source: str = "predicted",
    scorer: str = "auroc",
) -> Dict[str, Dict[str, float]]:
    """Do the exported discrete rules compute what the soft model computes?

    The model is evaluated as is and with its most likely discrete rulebook
    (:func:`discrete_rules`), exact STL semantics, and two selections:

    * ``"soft_selection"``: the learned selection weights;
    * ``"top_rule"``: only each sample's most selected rule, i.e. the single
      rule a reader would take as the explanation.

    Per task it reports ``agreement`` (same thresholded prediction as the
    soft model) and the task ``score`` of the soft model and of both discrete
    variants. Concept traces are the predicted or the true ones
    (``concept_source``).
    """
    from .cmrts import discrete_rules
    from ..rules.snf import SNFRuleDecoder

    if not isinstance(model.rule_decoder, SNFRuleDecoder):
        raise TypeError("rule_fidelity needs a model with a learned SNF rule bank.")
    soft = collect_predictions(model, dataloader, concept_source)
    variants = {}
    for name, selection in (("soft_selection", "soft"), ("top_rule", "top")):
        with discrete_rules(model, "argmax", selection):
            variants[name] = collect_predictions(model, dataloader, concept_source).probabilities
    score = SCORERS[scorer]
    table: Dict[str, Dict[str, float]] = {}
    for index, task in enumerate(model.task_names):
        reference = soft.probabilities[:, index] >= 0.5
        row = {
            f"{name}_agreement": float(((values[:, index] >= 0.5) == reference).float().mean())
            for name, values in variants.items()
        }
        y = soft.targets[:, index]
        if y.min() != y.max():
            row["model_score"] = float(score(y.numpy(), soft.probabilities[:, index].numpy()))
            for name, values in variants.items():
                row[f"{name}_score"] = float(score(y.numpy(), values[:, index].numpy()))
        table[task] = row
    rows = list(table.values())
    keys = set.intersection(*(set(row) for row in rows)) if rows else set()
    table["macro"] = {key: sum(row[key] for row in rows) / len(rows) for key in sorted(keys)}
    return table


# ---------------------------------------------------------------------------
# Concept quality
# ---------------------------------------------------------------------------


def _runs(active: torch.Tensor) -> List[tuple]:
    """``[(start, end)]`` (inclusive) of consecutive ``True`` steps of a 1-D mask."""
    padded = torch.cat([active.new_zeros(1), active.int(), active.new_zeros(1)])
    edges = padded.diff()
    starts = (edges == 1).nonzero().flatten().tolist()
    ends = (edges == -1).nonzero().flatten().tolist()
    return [(s, e - 1) for s, e in zip(starts, ends)]


def event_scores(
    predicted: torch.Tensor, true: torch.Tensor, tolerance: int = 0, threshold: float = 0.5
) -> Dict[str, float]:
    """Event-level precision, recall and F1 for one binary concept.

    An *event* is a run of consecutive active steps. A true event counts as
    detected when a predicted active step lies within ``tolerance`` steps of
    it, and a predicted event is correct when it lies within ``tolerance``
    steps of a true event. Unlike per-step scores, these are not dominated by
    the (many) inactive background steps.

    Args:
        predicted, true: ``(n, time)`` probabilities and 0/1 targets.
    """
    detected = true_events = correct = predicted_events = 0
    for p, t in zip(predicted >= threshold, true >= 0.5):
        near_p = torch.nn.functional.max_pool1d(
            p.float()[None, None], 2 * tolerance + 1, 1, tolerance
        )[0, 0].bool()
        near_t = torch.nn.functional.max_pool1d(
            t.float()[None, None], 2 * tolerance + 1, 1, tolerance
        )[0, 0].bool()
        for start, end in _runs(t):
            true_events += 1
            detected += bool(near_p[start:end + 1].any())
        for start, end in _runs(p):
            predicted_events += 1
            correct += bool(near_t[start:end + 1].any())
    recall = detected / true_events if true_events else float("nan")
    precision = correct / predicted_events if predicted_events else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if true_events and predicted_events and precision + recall > 0 else 0.0
    )
    return {
        "event_precision": precision, "event_recall": recall, "event_f1": f1,
        "true_events": true_events, "predicted_events": predicted_events,
    }


def concept_scores(
    model, predictions: Predictions, tolerance: int = 0
) -> Dict[str, Dict[str, float]]:
    """Per-concept quality of the predicted traces.

    Binary concepts: per-step AUROC and F1 plus :func:`event_scores`.
    Continuous concepts: MSE and R². ``predictions`` must come from
    :func:`collect_predictions` with ``keep_concepts=True``.
    """
    from sklearn.metrics import f1_score

    if predictions.concepts is None:
        raise ValueError("Collect the predictions with keep_concepts=True.")
    table = {}
    for index, name in enumerate(model.annotations.labels):
        p = predictions.concepts[..., index]
        t = predictions.true_concepts[..., index]
        if model.annotations.concept(name).type == "binary":
            flat_t, flat_p = t.flatten().numpy() >= 0.5, p.flatten().numpy()
            row = {"f1": float(f1_score(flat_t, flat_p >= 0.5, zero_division=0))}
            if flat_t.min() != flat_t.max():
                row["auroc"] = float(roc_auc_score(flat_t, flat_p))
            row.update(event_scores(p, t, tolerance))
        else:
            mse = float(((p - t) ** 2).mean())
            variance = float(t.var())
            row = {"mse": mse, "r2": 1 - mse / variance if variance > 0 else float("nan")}
        table[name] = row
    return table


# ---------------------------------------------------------------------------
# Rule compactness
# ---------------------------------------------------------------------------


def _normalised_entropy(weights: torch.Tensor) -> torch.Tensor:
    n = weights.shape[-1]
    entropy = -(weights * weights.clamp_min(1e-12).log()).sum(-1)
    return entropy / torch.log(torch.tensor(float(n))) if n > 1 else torch.zeros_like(entropy)


@torch.no_grad()
def rule_compactness(
    model, dataloader: Optional[Iterable] = None, overlap: float = 0.5
) -> Dict[str, Dict[str, float]]:
    """How crisp, distinct and selectively used the learned rules are.

    Per task:

    * ``concept_entropy``: mean normalised entropy (0 = one concept, 1 =
      uniform) of the antecedent and consequent concept weights. High values
      mean the soft rule mixes concepts, so its exported (argmax) version
      says less than the model uses.
    * ``template_entropy``: mean normalised entropy of the template choice
      (only with more than one template).
    * ``duplicate_rules``: pairs of rules with the same template, antecedent
      and consequent concepts whose intervals overlap with IoU >= ``overlap``.
    * ``selection_entropy`` (needs ``dataloader``): mean normalised entropy
      of the per-sample rule selection (0 = one rule per sample).
    * ``rules_used`` (needs ``dataloader``): rules that are some sample's
      top-selected rule.
    """
    from ..rules.snf import SNFRuleDecoder

    decoder = model.rule_decoder
    if not isinstance(decoder, SNFRuleDecoder):
        raise TypeError("rule_compactness needs a model with a learned SNF rule bank.")
    concept_entropy = (
        _normalised_entropy(decoder.antecedent_logits.softmax(-1))
        + _normalised_entropy(decoder.consequent_logits.softmax(-1))
    ) / 2
    start, end = decoder._intervals()
    antecedent = decoder.antecedent_logits.argmax(-1)
    consequent = decoder.consequent_logits.argmax(-1)
    template = (
        decoder.template_logits.argmax(-1) if decoder.template_logits is not None
        else torch.zeros_like(antecedent)
    )
    selections = None
    if dataloader is not None:
        from .cmrts_learner import TemporalCMRLearner

        was_training = model.training
        model.eval()
        device = next(model.parameters()).device
        selections = torch.cat([
            model(TemporalCMRLearner.unpack(batch)[0].to(device)).rule_selection.cpu()
            for batch in dataloader
        ])
        model.train(was_training)
    table = {}
    for k, task in enumerate(model.task_names):
        duplicates = 0
        for a in range(model.n_rules):
            for b in range(a + 1, model.n_rules):
                same = (antecedent[k, a] == antecedent[k, b]
                        and consequent[k, a] == consequent[k, b]
                        and template[k, a] == template[k, b])
                inter = (torch.minimum(end[k, a], end[k, b]) - torch.maximum(start[k, a], start[k, b])).clamp_min(0)
                union = torch.maximum(end[k, a], end[k, b]) - torch.minimum(start[k, a], start[k, b])
                duplicates += bool(same and inter / union.clamp_min(1e-6) >= overlap)
        row = {
            "concept_entropy": float(concept_entropy[k].mean()),
            "duplicate_rules": float(duplicates),
        }
        if decoder.template_logits is not None:
            row["template_entropy"] = float(
                _normalised_entropy(decoder.template_logits[k].softmax(-1)).mean()
            )
        if selections is not None:
            row["selection_entropy"] = float(_normalised_entropy(selections[:, k]).mean())
            row["rules_used"] = float(selections[:, k].argmax(-1).unique().numel())
        table[task] = row
    return table
