"""Concept interventions on temporal CMR models.

Interventions go through the library's :func:`~torch_concepts.nn.intervention`
on ``model.concept_head``, whose output is ``(batch, time, concepts)``
pre-activations (logits for binary concepts, raw values for continuous ones),
so every intervention strategy and policy of the library applies per time step:

* whole concept traces: ``GroundTruthIntervention`` + ``UniformPolicy`` on a
  group of concepts (the standard concept-bottleneck protocol);
* concepts chosen per time step: e.g. ``UncertaintyInterventionPolicy`` with
  ``quantile``;
* a subset of time steps: :class:`TimeMaskedIntervention` (ground truth on
  masked ``(time, concept)`` cells, predictions elsewhere).

The rule selection is computed from the inputs, so concept interventions change
which concepts the rules see, not which rules are selected.
"""

from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Sequence, Union

import torch

from ...low.base.intervention import ConceptInterventionStrategy, intervention
from ...low.intervention.policy.uniform import UniformPolicy
from ...low.intervention.strategy.ground_truth import GroundTruthIntervention
from .cmrts_evaluation import SCORERS


class TimeMaskedIntervention(ConceptInterventionStrategy):
    """Ground truth on the masked ``(…, time, concept)`` cells, predictions elsewhere.

    Args:
        ground_truth: Target values in the layer's output space, ``(batch,
            time, concepts)``.
        mask: Boolean, broadcastable to ``ground_truth``; ``True`` marks the
            cells to replace.
    """

    def __init__(self, ground_truth: torch.Tensor, mask: torch.Tensor):
        super().__init__()
        self.ground_truth = ground_truth
        self.mask = mask

    def forward(self, predictions, *args, **kwargs):
        mask = self.mask.to(device=predictions.device, dtype=torch.bool)
        return torch.where(mask, self.ground_truth.to(predictions), predictions)


def _binary(model) -> torch.Tensor:
    return torch.tensor(
        [model.annotations.concept(name).type == "binary" for name in model.annotations.labels]
    )


def to_head_space(model, values: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Concept values -> ``concept_head`` outputs (logits for binary concepts)."""
    binary = _binary(model).to(values.device)
    return torch.where(binary, torch.logit(values.clamp(eps, 1 - eps)), values)


def concept_indices(model, concepts: Optional[Sequence[Union[str, int]]] = None) -> List[int]:
    labels = list(model.annotations.labels)
    if concepts is None:
        return list(range(len(labels)))
    return [labels.index(c) if isinstance(c, str) else int(c) for c in concepts]


@contextmanager
def intervene_on_concepts(
    model,
    ground_truth: torch.Tensor,
    concepts: Optional[Sequence[Union[str, int]]] = None,
    policy=None,
    quantile: float = 1.0,
    mask: Optional[torch.Tensor] = None,
):
    """Inside the block, ``model`` sees ground-truth concept values.

    Args:
        model: :class:`ConceptMemoryReasonerTS`.
        ground_truth: ``(batch, time, concepts)`` true concept values (0/1
            for binary concepts) of the batch that will be passed to ``model``.
        concepts: Names or positions that may be intervened on (default all).
        policy: Library intervention policy; default ``UniformPolicy`` (every
            selected concept at every time step).
        quantile: Fraction of the selected concepts replaced per time step.
        mask: Optional boolean ``(batch, time)`` or ``(batch, time, concepts)``
            mask of cells to replace; uses :class:`TimeMaskedIntervention`
            (``policy`` and ``quantile`` must then be left at their defaults).
    """
    indices = concept_indices(model, concepts)
    target = to_head_space(model, ground_truth)
    if mask is None:
        strategy = GroundTruthIntervention(target)
    else:
        if policy is not None or quantile != 1.0:
            raise ValueError("A mask selects the cells itself; leave policy and quantile unset.")
        mask = mask.bool()
        if mask.dim() == ground_truth.dim() - 1:
            mask = mask.unsqueeze(-1)
        strategy = TimeMaskedIntervention(target, mask.expand_as(ground_truth))
    with intervention(
        model.concept_head, strategy, policy or UniformPolicy(), indices, quantile=quantile
    ):
        yield


def uncertainty(model, probabilities: torch.Tensor) -> torch.Tensor:
    """Per-cell uncertainty: ``1 - |2p - 1|`` for binary concepts, 0 otherwise."""
    binary = _binary(model).to(probabilities.device)
    return torch.where(binary, 1 - (2 * probabilities - 1).abs(), torch.zeros_like(probabilities))


def time_mask(
    scores: torch.Tensor, fraction: float, generator: Optional[torch.Generator] = None,
    order: str = "random",
) -> torch.Tensor:
    """Select ``fraction`` of the time steps per sample and concept.

    Args:
        scores: ``(batch, time, concepts)`` uncertainty (higher first) for
            ``order="uncertainty"``; only its shape is used for ``"random"``.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1].")
    batch, time, n = scores.shape
    k = int(round(fraction * time))
    if order == "random":
        ranking = torch.rand(scores.shape, generator=generator).to(scores.device)
    elif order == "uncertainty":
        ranking = scores + 1e-6 * torch.rand(scores.shape, generator=generator).to(scores.device)
    else:
        raise ValueError("order must be 'random' or 'uncertainty'.")
    mask = torch.zeros_like(scores, dtype=torch.bool)
    if k:
        top = ranking.topk(k, dim=1).indices
        mask.scatter_(1, top, True)
    return mask


def concept_mask(
    scores: torch.Tensor, n_concepts: int, generator: Optional[torch.Generator] = None,
    order: str = "random",
) -> torch.Tensor:
    """Whole traces of ``n_concepts`` concepts per sample (most uncertain first)."""
    batch, time, n = scores.shape
    if order == "random":
        ranking = torch.rand(batch, n, generator=generator).to(scores.device)
    elif order == "uncertainty":
        ranking = scores.mean(1) + 1e-6 * torch.rand(batch, n, generator=generator).to(scores.device)
    else:
        raise ValueError("order must be 'random' or 'uncertainty'.")
    mask = torch.zeros(batch, n, dtype=torch.bool, device=scores.device)
    if n_concepts:
        mask.scatter_(1, ranking.topk(n_concepts, dim=1).indices, True)
    return mask[:, None, :].expand(batch, time, n)


def _score(probabilities: torch.Tensor, targets: torch.Tensor, task_names, scorer: str):
    score = SCORERS[scorer]
    per_task = {}
    for index, task in enumerate(task_names):
        y = targets[:, index]
        if y.min() != y.max():
            per_task[task] = float(score(y.numpy(), probabilities[:, index].numpy()))
    macro = sum(per_task.values()) / len(per_task) if per_task else float("nan")
    return macro, per_task


@torch.no_grad()
def _run(model, dataloader, make_mask, seed: int):
    """Task probabilities with ``make_mask(probabilities, uncertainty, generator)``."""
    from .cmrts_learner import TemporalCMRLearner

    generator = torch.Generator().manual_seed(seed)
    device = next(model.parameters()).device
    probabilities, targets = [], []
    for batch in dataloader:
        inputs, true, target = (t.to(device) for t in TemporalCMRLearner.unpack(batch))
        predicted = model(inputs).concepts.tensor
        mask = make_mask(predicted, uncertainty(model, predicted), generator)
        if mask is None or not mask.any():
            output = model(inputs)
        else:
            with intervene_on_concepts(model, true, mask=mask):
                output = model(inputs)
        probabilities.append(output.tasks.tensor.cpu())
        targets.append(target.cpu())
    return torch.cat(probabilities), torch.cat(targets)


def concept_intervention_curve(
    model,
    dataloader: Iterable,
    order: str = "random",
    n_repeats: int = 3,
    scorer: str = "auroc",
    seed: int = 0,
) -> Dict[str, object]:
    """Task score after replacing the whole traces of ``k`` concepts.

    ``k`` runs from 0 to the number of concepts; concepts are chosen per
    sample at random (averaged over ``n_repeats`` draws) or most uncertain
    first. Tasks with a single class are skipped.

    Returns:
        ``{"x": [0..C], "score": [...], "per_task": {task: [...]},
        "auc": mean score, "gain": score(C) - score(0)}``.
    """
    return _curve(
        model, dataloader, list(range(len(model.annotations.labels) + 1)),
        lambda p, u, g, k: concept_mask(u, k, g, order), order, n_repeats, scorer, seed,
    )


def time_intervention_curve(
    model,
    dataloader: Iterable,
    fractions: Sequence[float] = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
    order: str = "random",
    n_repeats: int = 3,
    scorer: str = "auroc",
    seed: int = 0,
) -> Dict[str, object]:
    """Task score after replacing a fraction of the time steps of every concept.

    With ``order="uncertainty"`` the time steps whose predictions are least
    certain are corrected first. Same return format as
    :func:`concept_intervention_curve`, with ``x`` the fractions.
    """
    return _curve(
        model, dataloader, list(fractions),
        lambda p, u, g, f: time_mask(u, f, g, order), order, n_repeats, scorer, seed,
    )


def _curve(model, dataloader, xs, mask_for, order, n_repeats, scorer, seed):
    if scorer not in SCORERS:
        raise ValueError(f"scorer must be one of {sorted(SCORERS)}.")
    was_training = model.training
    model.eval()
    repeats = n_repeats if order == "random" else 1
    scores, per_task = [], {}
    for x in xs:
        runs = []
        for repeat in range(repeats):
            probabilities, targets = _run(
                model, dataloader, lambda p, u, g: mask_for(p, u, g, x), seed + repeat
            )
            runs.append(_score(probabilities, targets, model.task_names, scorer))
        scores.append(sum(r[0] for r in runs) / repeats)
        for task in runs[0][1]:
            per_task.setdefault(task, []).append(sum(r[1][task] for r in runs) / repeats)
    model.train(was_training)
    return {
        "x": xs,
        "score": scores,
        "per_task": per_task,
        "auc": sum(scores) / len(scores),
        "gain": scores[-1] - scores[0],
    }


def single_concept_effects(
    model, dataloader: Iterable, scorer: str = "auroc"
) -> Dict[str, float]:
    """Change in macro task score when only one concept's trace is corrected."""
    was_training = model.training
    model.eval()
    baseline = _score(*_run(model, dataloader, lambda p, u, g: None, 0), model.task_names, scorer)[0]
    effects = {}
    for index, name in enumerate(model.annotations.labels):
        def only(p, u, g, index=index):
            mask = torch.zeros_like(p, dtype=torch.bool)
            mask[..., index] = True
            return mask

        effects[name] = _score(*_run(model, dataloader, only, 0), model.task_names, scorer)[0] - baseline
    model.train(was_training)
    return effects


# ---------------------------------------------------------------------------
# Rule interventions
# ---------------------------------------------------------------------------

_FORCE = 30.0  # selector logit gap that makes the softmax effectively one-hot


def _selector_logits(model, task: Union[str, int], values: Dict[int, float]) -> torch.Tensor:
    task_index = model.task_names.index(task) if isinstance(task, str) else int(task)
    logits = torch.zeros(len(model.task_names) * model.n_rules)
    positions = []
    for rule, value in values.items():
        if not 0 <= rule < model.n_rules:
            raise ValueError(f"rule must be in [0, {model.n_rules}).")
        position = task_index * model.n_rules + rule
        logits[position] = value
        positions.append(position)
    return logits, positions


@contextmanager
def force_rule(model, task: Union[str, int], rule: int):
    """Inside the block, every sample uses rule ``rule`` for ``task``.

    A library ``DoIntervention`` on ``model.rule_selector`` sets that task's
    selector logits; other tasks keep their learned selection.
    """
    from ...low.intervention.strategy.do import DoIntervention

    values = {r: (_FORCE if r == rule else -_FORCE) for r in range(model.n_rules)}
    logits, positions = _selector_logits(model, task, values)
    with intervention(model.rule_selector, DoIntervention(logits), UniformPolicy(), positions):
        yield


@contextmanager
def remove_rule(model, task: Union[str, int], rule: int):
    """Inside the block, rule ``rule`` of ``task`` is never selected.

    Its selector logit is set very low, so the softmax renormalises the
    selection over the remaining rules.
    """
    from ...low.intervention.strategy.do import DoIntervention

    if model.n_rules < 2:
        raise ValueError("Cannot remove the only rule of a task.")
    logits, positions = _selector_logits(model, task, {rule: -1e4})
    with intervention(model.rule_selector, DoIntervention(logits), UniformPolicy(), positions):
        yield


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return float(torch.logit(torch.tensor(p)))


@contextmanager
def edit_rule(model, task: Union[str, int], rule: int, spec: dict):
    """Inside the block, a learned SNF rule is replaced by an expert rule.

    ``spec`` uses the rulebook format of kind ``"always_implies_eventually"``
    (the form the SNF bank can represent): ``antecedent`` / ``consequent``
    ``{"concept", "threshold"}`` and ``interval`` ``[start, end]`` in time
    steps. The rule's parameters are restored when the block exits.
    """
    from ..rules.snf import SNFRuleDecoder

    decoder = model.rule_decoder
    if not isinstance(decoder, SNFRuleDecoder):
        raise TypeError("edit_rule needs a model with a learned SNF rule bank.")
    if spec.get("kind") != "always_implies_eventually":
        raise ValueError("The SNF bank only represents 'always_implies_eventually' rules.")
    task_index = model.task_names.index(task) if isinstance(task, str) else int(task)
    labels = list(decoder.annotations.labels)
    horizon = float(decoder.config.max_horizon)
    min_width = decoder.config.min_interval
    start, end = (float(v) for v in spec["interval"])
    if not (0 <= start and start + min_width <= end <= horizon):
        raise ValueError(
            f"interval must satisfy 0 <= start, start + {min_width} <= end <= {horizon}."
        )
    names = (
        "antecedent_logits", "consequent_logits", "antecedent_threshold_logits",
        "consequent_threshold_logits", "start_logits", "width_logits",
    )
    saved = {name: getattr(decoder, name).detach().clone() for name in names}
    try:
        with torch.no_grad():
            for role in ("antecedent", "consequent"):
                concept = labels.index(spec[role]["concept"])
                logits = getattr(decoder, f"{role}_logits")
                logits[task_index, rule] = -20.0
                logits[task_index, rule, concept] = 20.0
                getattr(decoder, f"{role}_threshold_logits")[task_index, rule] = _logit(
                    float(spec[role].get("threshold", 0.5))
                )
            decoder.start_logits[task_index, rule] = _logit(start / (horizon - min_width))
            decoder.width_logits[task_index, rule] = _logit(
                (end - start - min_width) / max(horizon - start - min_width, 1e-6)
            )
        yield
    finally:
        with torch.no_grad():
            for name, value in saved.items():
                getattr(decoder, name).copy_(value)


@torch.no_grad()
def _task_probabilities(model, dataloader, context=None):
    from .cmrts_learner import TemporalCMRLearner

    device = next(model.parameters()).device
    probabilities, targets = [], []
    for batch in dataloader:
        inputs, _, target = (t.to(device) for t in TemporalCMRLearner.unpack(batch))
        if context is None:
            output = model(inputs)
        else:
            with context():
                output = model(inputs)
        probabilities.append(output.tasks.tensor.cpu())
        targets.append(target.cpu())
    return torch.cat(probabilities), torch.cat(targets)


def rule_intervention_scores(
    model, dataloader: Iterable, scorer: str = "auroc"
) -> Dict[str, Dict[str, object]]:
    """Per task: the learned selection's score and each rule's score alone.

    Returns ``{task: {"selected": s, "forced": [s_r0, s_r1, ...],
    "removed": [...], "best_rule": r, "gain": max(forced) - selected}}``
    (``removed`` only with more than one rule). ``gain > 0`` means a single
    rule beats the learned mixture: the selector is not using it well.
    Tasks with a single class are skipped.
    """
    if scorer not in SCORERS:
        raise ValueError(f"scorer must be one of {sorted(SCORERS)}.")
    score = SCORERS[scorer]
    was_training = model.training
    model.eval()
    base, targets = _task_probabilities(model, dataloader)
    table = {}
    for index, task in enumerate(model.task_names):
        y = targets[:, index]
        if y.min() == y.max():
            continue
        evaluate = lambda p: float(score(y.numpy(), p[:, index].numpy()))
        forced = [
            evaluate(_task_probabilities(
                model, dataloader, lambda r=r: force_rule(model, task, r))[0])
            for r in range(model.n_rules)
        ]
        row = {"selected": evaluate(base), "forced": forced}
        if model.n_rules > 1:
            row["removed"] = [
                evaluate(_task_probabilities(
                    model, dataloader, lambda r=r: remove_rule(model, task, r))[0])
                for r in range(model.n_rules)
            ]
        row["best_rule"] = int(max(range(model.n_rules), key=forced.__getitem__))
        row["gain"] = max(forced) - row["selected"]
        table[task] = row
    model.train(was_training)
    return table
