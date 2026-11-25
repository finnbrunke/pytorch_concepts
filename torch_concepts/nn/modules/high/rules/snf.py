"""Differentiable one-clause temporal rules (a learned rule bank).

Each task-rule pair is one clause over two concept predicates ``A = c_a >
theta_a`` and ``B = c_b > theta_b`` in one of the templates

* ``always_implies_eventually``: ``G(A -> F_[a,b] B)``,
* ``eventually_and_eventually``: ``F(A and F_[a,b] B)``,
* ``always_not``: ``G(not A)``,

chosen per rule among ``SNFConfig.templates``. The concepts, the template,
the thresholds and the interval are learned.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .....annotations import Annotations
from .....tensor import AnnotatedTensor
from .operators import smooth_and, smooth_or

TEMPLATES = ("always_implies_eventually", "eventually_and_eventually", "always_not")


@dataclass(frozen=True)
class SNFConfig:
    """Configuration for globally learned SNF rules.

    Args:
        max_horizon: Largest temporal offset (in time steps) an interval may reach.
        smoothness: Inverse temperature of the soft min/max over robustness
            values. Robustness lives roughly in ``[-1, 1]``.
        robustness_scale: Initial slope of ``sigmoid(scale * robustness)``.
        learn_robustness_scale: Whether the slope is a trainable parameter.
        min_interval: Minimum interval width in time steps.
        time_temperature: Softness (in time steps) of the interval gates. Kept
            separate from ``smoothness`` because it is measured in time steps,
            not robustness units. Defaults to ``max_horizon / 20``.
        init_std: Standard deviation of the random parameter initialisation.
            A non-zero value breaks the symmetry between rules of one task.
        templates: Rule templates a rule may take (see module docstring).
            With more than one, each rule learns which one it is.
        discrete: ``"soft"`` mixes concepts (and templates) with softmax
            weights. ``"straight_through"`` uses hard one-hot Gumbel samples
            in training (soft gradients, straight-through) and the argmax in
            eval mode, so the trained model computes with discrete rules.
        gumbel_temperature: Temperature of the Gumbel-softmax samples.
    """

    max_horizon: int = 40
    smoothness: float = 10.0
    robustness_scale: float = 5.0
    learn_robustness_scale: bool = True
    min_interval: float = 1.0
    time_temperature: Optional[float] = None
    init_std: float = 1.0
    templates: tuple = ("always_implies_eventually",)
    discrete: str = "soft"
    gumbel_temperature: float = 1.0

    def __post_init__(self):
        if self.max_horizon <= 0:
            raise ValueError("max_horizon must be positive.")
        if self.smoothness <= 0 or self.robustness_scale <= 0:
            raise ValueError("smoothness and robustness_scale must be positive.")
        if not 0 < self.min_interval <= self.max_horizon:
            raise ValueError("min_interval must be in (0, max_horizon].")
        if self.time_temperature is not None and self.time_temperature <= 0:
            raise ValueError("time_temperature must be positive.")
        if self.init_std < 0:
            raise ValueError("init_std must be non-negative.")
        templates = tuple(self.templates)
        if not templates or len(set(templates)) != len(templates) or not set(templates) <= set(TEMPLATES):
            raise ValueError(f"templates must be distinct entries of {TEMPLATES}.")
        object.__setattr__(self, "templates", templates)
        if self.discrete not in {"soft", "straight_through"}:
            raise ValueError("discrete must be 'soft' or 'straight_through'.")
        if self.gumbel_temperature <= 0:
            raise ValueError("gumbel_temperature must be positive.")

    @property
    def resolved_time_temperature(self) -> float:
        if self.time_temperature is not None:
            return self.time_temperature
        return self.max_horizon / 20.0


class SNFRuleDecoder(nn.Module):
    """Learn a global bank of one-clause temporal rules (see module docstring)."""

    def __init__(
        self,
        annotations: Annotations,
        task_names: list[str],
        n_rules: int,
        config: SNFConfig | None = None,
    ):
        super().__init__()
        if n_rules <= 0:
            raise ValueError("n_rules must be positive.")
        self.annotations = annotations
        self.task_names = list(task_names)
        self.n_rules = n_rules
        self.config = config or SNFConfig()
        self._indices = {name: i for i, name in enumerate(annotations.labels)}
        shape = (len(self.task_names), n_rules, len(annotations.labels))
        self.antecedent_logits = nn.Parameter(torch.empty(shape))
        self.consequent_logits = nn.Parameter(torch.empty(shape))
        interval_shape = (len(self.task_names), n_rules)
        self.start_logits = nn.Parameter(torch.empty(interval_shape))
        self.width_logits = nn.Parameter(torch.empty(interval_shape))
        self.antecedent_threshold_logits = nn.Parameter(torch.empty(shape))
        self.consequent_threshold_logits = nn.Parameter(torch.empty(shape))
        self.templates = self.config.templates
        if len(self.templates) > 1:
            self.template_logits = nn.Parameter(
                torch.empty(len(self.task_names), n_rules, len(self.templates))
            )
        else:
            self.template_logits = None
        log_scale = torch.tensor(math.log(self.config.robustness_scale))
        if self.config.learn_robustness_scale:
            self.log_robustness_scale = nn.Parameter(log_scale)
        else:
            self.register_buffer("log_robustness_scale", log_scale)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Randomly initialise rule parameters so rules of a task differ."""
        std = self.config.init_std
        for parameter in (
            self.antecedent_logits,
            self.consequent_logits,
            self.start_logits,
            self.width_logits,
        ):
            nn.init.normal_(parameter, std=std)
        for parameter in (
            self.antecedent_threshold_logits,
            self.consequent_threshold_logits,
        ):
            nn.init.normal_(parameter, std=0.25 * std)
        if self.template_logits is not None:
            nn.init.normal_(self.template_logits, std=0.25 * std)

    @property
    def robustness_scale(self) -> torch.Tensor:
        return self.log_robustness_scale.exp()

    def _intervals(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``0 <= start < start + min_interval <= end <= max_horizon``.

        Built without clamps, so both logits always receive gradients.
        """
        horizon = float(self.config.max_horizon)
        min_width = self.config.min_interval
        start = (horizon - min_width) * torch.sigmoid(self.start_logits)
        available = horizon - start - min_width
        end = start + min_width + available * torch.sigmoid(self.width_logits)
        return start, end

    def choice(self, logits: torch.Tensor) -> torch.Tensor:
        """Weights of a categorical choice (concepts or templates) over the last axis.

        Softmax for ``discrete="soft"``. For ``"straight_through"``: a hard
        one-hot Gumbel-softmax sample in training mode (gradients of the soft
        sample) and the one-hot argmax in eval mode.
        """
        if self.config.discrete == "soft":
            return torch.softmax(logits, dim=-1)
        if self.training:
            return F.gumbel_softmax(logits, tau=self.config.gumbel_temperature, hard=True, dim=-1)
        return F.one_hot(logits.argmax(-1), logits.shape[-1]).to(logits.dtype)

    def template_weights(self) -> torch.Tensor:
        """``(tasks, rules, templates)`` weights of the rule templates."""
        if self.template_logits is None:
            return self.start_logits.new_ones(*self.start_logits.shape, 1)
        return self.choice(self.template_logits)

    def _predicate(
        self,
        concepts: torch.Tensor,
        concept_logits: torch.Tensor,
        threshold_logits: torch.Tensor,
    ) -> torch.Tensor:
        weights = self.choice(concept_logits)
        thresholds = torch.sigmoid(threshold_logits)
        return (
            (
                concepts.unsqueeze(1).unsqueeze(1)
                - thresholds.unsqueeze(0).unsqueeze(-2)
            )
            * weights.unsqueeze(0).unsqueeze(-2)
        ).sum(dim=-1)

    def _eventually(
        self, values: torch.Tensor, start: torch.Tensor, end: torch.Tensor
    ) -> torch.Tensor:
        """Soft ``max`` of ``values[t + start : t + end]`` for every ``t``.

        ``values`` is ``(batch, tasks, rules, time)``; returns the same shape.
        The soft interval gate ``g`` blends each value towards a floor below
        every observed value (``g * v + (1 - g) * floor``), so out-of-interval
        values can never win the max. Offsets beyond the end of the trace are
        excluded exactly.
        """
        time = values.shape[-1]
        horizon = self.config.max_horizon
        temperature = self.config.resolved_time_temperature
        offsets = torch.arange(horizon, device=values.device, dtype=values.dtype)
        windows = F.pad(values, (0, horizon - 1)).unfold(-1, horizon, 1)
        positions = torch.arange(time, device=values.device)
        valid = (positions[:, None] + offsets.long()[None, :]) < time
        # Gates are centred half a step outside [start, end], so an integer
        # offset is (softly) inside exactly when it rounds into the interval;
        # this matches the exported discrete interval [round(start), round(end)].
        gates = torch.sigmoid((offsets - start[..., None] + 0.5) / temperature)
        gates = gates * torch.sigmoid((end[..., None] - offsets + 0.5) / temperature)
        gates = gates[None, :, :, None, :]
        floor = values.detach().amin(dim=-1, keepdim=True).unsqueeze(-1) - 1.0
        gated = gates * windows + (1.0 - gates) * floor
        logits = (self.config.smoothness * gated).masked_fill(~valid, float("-inf"))
        weights = torch.softmax(logits, dim=-1)
        return (weights * gated.masked_fill(~valid, 0.0)).sum(dim=-1)

    def _always(self, values: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        """Soft ``min`` over time steps whose future window fits in the trace.

        A softmin-weighted mean is used instead of ``-logsumexp / beta``; the
        latter is biased by ``-log(T) / beta``, which dominates for long traces.
        Time steps with ``t + end > T - 1`` are softly masked out (blended
        towards a ceiling), because the bounded ``Eventually`` is truncated
        there. ``t = 0`` is always kept, so a rule is never vacuously true.
        """
        time = values.shape[-1]
        temperature = self.config.resolved_time_temperature
        positions = torch.arange(time, device=values.device, dtype=values.dtype)
        # The +3 margin keeps the last fully valid step at weight ~0.95.
        margin = (time - 1 - end[..., None] - positions) / temperature + 3.0
        mask = torch.sigmoid(margin)
        mask = torch.cat([torch.ones_like(mask[..., :1]), mask[..., 1:]], dim=-1)
        ceiling = values.detach().amax(dim=-1, keepdim=True) + 1.0
        gated = mask[None] * values + (1.0 - mask[None]) * ceiling
        weights = torch.softmax(-self.config.smoothness * gated, dim=-1)
        return (weights * gated).sum(dim=-1)

    def forward(
        self,
        concepts: AnnotatedTensor,
        rule_selection: torch.Tensor,
    ) -> tuple[AnnotatedTensor, torch.Tensor, torch.Tensor]:
        values = concepts.tensor
        antecedent = self._predicate(
            values, self.antecedent_logits, self.antecedent_threshold_logits
        )
        consequent = self._predicate(
            values, self.consequent_logits, self.consequent_threshold_logits
        )
        start, end = self._intervals()
        beta = self.config.smoothness
        by_template = []
        for template in self.templates:
            if template == "always_not":
                by_template.append(self._always(-antecedent, torch.zeros_like(end)))
                continue
            future = self._eventually(consequent, start, end)
            if template == "always_implies_eventually":
                # Smooth max: with a hard max the consequent gets no gradient
                # wherever the antecedent is false (almost always if sparse).
                by_template.append(self._always(smooth_or(-antecedent, future, beta), end))
            else:  # eventually_and_eventually: outer F over the same steps as G
                by_template.append(-self._always(-smooth_and(antecedent, future, beta), end))
        robustness = (
            torch.stack(by_template, dim=-1) * self.template_weights()[None]
        ).sum(dim=-1)
        satisfaction = torch.sigmoid(self.robustness_scale * robustness)
        if rule_selection.shape != satisfaction.shape:
            raise ValueError(
                f"rule_selection must have shape {tuple(satisfaction.shape)}, "
                f"got {tuple(rule_selection.shape)}."
            )
        tasks = (rule_selection * satisfaction).sum(dim=-1)
        task_annotations = Annotations(
            labels=self.task_names,
            cardinalities=[1] * len(self.task_names),
            types=["binary"] * len(self.task_names),
        )
        return AnnotatedTensor(tasks, task_annotations), robustness, satisfaction

    @staticmethod
    def _spec(name, task, kind, antecedent, consequent, interval) -> dict:
        """A rule in the dataset rulebook format (also readable by STLRuleDecoder).

        ``antecedent``/``consequent``/``interval`` are always present; the
        kind-specific aliases (``first``/``second`` or ``predicate``) are added.
        """
        spec = {
            "name": name, "task": task, "kind": kind,
            "antecedent": antecedent, "consequent": consequent, "interval": interval,
        }
        if kind == "eventually_and_eventually":
            spec.update(first=antecedent, second=consequent)
        elif kind == "always_not":
            spec["predicate"] = antecedent
        return spec

    @torch.no_grad()
    def _choices(self, mode: str, generator=None) -> dict:
        def pick(logits):
            logits = logits.detach().cpu()
            if mode == "argmax":
                return logits.argmax(-1)
            flat = logits.softmax(-1).reshape(-1, logits.shape[-1])
            return torch.multinomial(flat, 1, generator=generator).reshape(logits.shape[:-1])

        chosen = {role: pick(getattr(self, f"{role}_logits")) for role in ("antecedent", "consequent")}
        chosen["template"] = (
            pick(self.template_logits) if self.template_logits is not None
            else torch.zeros(self.start_logits.shape, dtype=torch.long)
        )
        return chosen

    @torch.no_grad()
    def discrete_specs(
        self,
        concepts: str = "argmax",
        generator: Optional[torch.Generator] = None,
        max_end: Optional[int] = None,
    ) -> list[dict]:
        """One discrete rulebook from the bank, in the dataset rulebook format.

        The concept of each antecedent/consequent and each rule's template are
        the most likely ones (``concepts="argmax"``) or drawn from their
        softmax distributions (``"sample"``); thresholds are kept; intervals
        are rounded to integers (the soft gates are centred half a step
        outside ``[start, end]``) and ``end`` is capped at ``max_end``.
        """
        if concepts not in {"argmax", "sample"}:
            raise ValueError("concepts must be 'argmax' or 'sample'.")
        return self._discrete_rules(self._choices(concepts, generator), integer=True, max_end=max_end)

    @torch.no_grad()
    def export_rules(self) -> list[dict]:
        """Return the learned bank as discrete, inspectable rule specs.

        Most likely concepts and templates; continuous interval bounds.
        """
        return self._discrete_rules(self._choices("argmax"), integer=False)

    def _discrete_rules(self, chosen: dict, integer: bool, max_end=None) -> list[dict]:
        start, end = self._intervals()
        thresholds = {
            role: getattr(self, f"{role}_threshold_logits").detach().sigmoid().cpu()
            for role in ("antecedent", "consequent")
        }
        specs = []
        for k, task in enumerate(self.task_names):
            for r in range(self.n_rules):
                if integer:
                    lo = int(round(float(start[k, r])))
                    hi = int(round(float(end[k, r])))
                    if max_end is not None:
                        hi, lo = min(hi, max_end), min(lo, max_end)
                    interval = [lo, max(lo, hi)]
                else:
                    interval = [float(start[k, r]), float(end[k, r])]
                roles = {}
                for role in ("antecedent", "consequent"):
                    concept = int(chosen[role][k, r])
                    roles[role] = {
                        "concept": self.annotations.labels[concept],
                        "threshold": float(thresholds[role][k, r, concept]),
                    }
                kind = self.templates[int(chosen["template"][k, r])]
                specs.append(self._spec(
                    f"learned_{task}_{r}", task, kind, roles["antecedent"],
                    roles["consequent"], interval,
                ))
        return specs


def format_rule(spec: dict) -> str:
    """Readable STL for a rulebook entry, e.g. ``G(a > 0.50 -> F[1,4] b > 0.50)``."""
    def atom(role):
        part = spec.get(role) or spec.get({"antecedent": "first", "consequent": "second"}[role]) \
            or spec.get("predicate")
        return f"{part['concept']} > {part['threshold']:.2f}"

    kind = spec["kind"]
    if kind == "always_not":
        return f"G(not {atom('antecedent')})"
    start, end = spec["interval"]
    window = f"F[{start:.0f},{end:.0f}]"
    if kind == "always_implies_eventually":
        return f"G({atom('antecedent')} -> {window} {atom('consequent')})"
    if kind == "eventually_and_eventually":
        return f"F({atom('antecedent')} and {window} {atom('consequent')})"
    raise ValueError(f"Unknown rule kind {kind!r}.")
