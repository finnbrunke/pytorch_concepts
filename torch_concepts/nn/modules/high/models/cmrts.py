"""Sequence-to-sequence building blocks for temporal CMR models.

The temporal model deliberately keeps STL rule decoding separate from the
sequence representation.  This module provides the reusable parts around that
decoder: sequence encoding, per-time-step concept annotation, and global rule
selection.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from torch import nn

from .....annotations import Annotations
from .....tensor import AnnotatedTensor
from ..rules.operators import smooth_and, smooth_max, smooth_min, smooth_or
from ..rules.snf import SNFConfig, SNFRuleDecoder


class SequenceEncoder(nn.Module):
    """Configurable sequence-to-sequence encoder.

    All supported encoders preserve the time axis:
    ``(batch, time, input_size) -> (batch, time, latent_size)``.
    """

    def __init__(
        self,
        input_size: int,
        latent_size: int,
        kind: str = "gru",
        hidden_size: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        n_heads: int = 4,
    ):
        super().__init__()
        if input_size <= 0 or latent_size <= 0:
            raise ValueError("input_size and latent_size must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the interval [0, 1).")

        kind = kind.lower()
        hidden_size = hidden_size or latent_size
        self.kind = kind
        self.input_size = input_size
        self.latent_size = latent_size

        if kind in {"gru", "lstm"}:
            recurrent = nn.GRU if kind == "gru" else nn.LSTM
            self.input_projection = (
                nn.Linear(input_size, hidden_size)
                if hidden_size != input_size
                else nn.Identity()
            )
            self.encoder = recurrent(
                input_size=hidden_size,
                hidden_size=latent_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
        elif kind == "transformer":
            if latent_size % n_heads != 0:
                raise ValueError(
                    "latent_size must be divisible by n_heads for a transformer."
                )
            self.input_projection = nn.Linear(input_size, latent_size)
            layer = nn.TransformerEncoderLayer(
                d_model=latent_size,
                nhead=n_heads,
                dim_feedforward=hidden_size,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        elif kind == "mlp":
            self.encoder = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, latent_size),
            )
            self.input_projection = nn.Identity()
        else:
            raise ValueError(
                f"Unknown sequence encoder {kind!r}. Choose from "
                "'gru', 'lstm', 'transformer', or 'mlp'."
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(
                "SequenceEncoder expects inputs shaped "
                "(batch, time, features)."
            )
        if inputs.shape[-1] != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} input features, "
                f"got {inputs.shape[-1]}."
            )
        projected = self.input_projection(inputs)
        if self.kind == "mlp":
            return self.encoder(projected)
        encoded = self.encoder(projected)
        return encoded[0] if isinstance(encoded, tuple) else encoded


@dataclass
class CMRTemporalOutput:
    """Outputs of the temporal representation and global STL classifier."""

    latent: torch.Tensor
    concepts: Optional[AnnotatedTensor]
    rule_selection: Optional[torch.Tensor]
    rule_robustness: Optional[torch.Tensor] = None
    rule_satisfaction: Optional[torch.Tensor] = None
    tasks: Optional[AnnotatedTensor] = None
    tasks_with_rec: Optional[AnnotatedTensor] = None

    @property
    def rule_selector(self) -> torch.Tensor:
        """Backward-compatible alias for :attr:`rule_selection`."""
        return self.rule_selection


@dataclass(frozen=True)
class STLRuleSpec:
    """A fixed, named stage-1 STL rule template."""

    name: str
    task: str
    kind: str
    fields: Dict[str, object]

    @classmethod
    def from_dict(cls, spec: dict) -> "STLRuleSpec":
        required = {"name", "task", "kind"}
        missing = required.difference(spec)
        if missing:
            raise ValueError(f"Rule specification is missing {sorted(missing)}.")
        fields = {key: value for key, value in spec.items() if key not in required}
        return cls(spec["name"], spec["task"], spec["kind"], fields)


class STLRuleDecoder(nn.Module):
    """Evaluate a fixed STL rule bank and mix it into global task outputs.

    Robustness is evaluated at the beginning of each trace. ``Always``,
    ``Eventually``, conjunction and implication use smooth min/max with inverse
    temperature ``smoothness`` so every concept value receives a gradient;
    ``smoothness=None`` gives exact hard STL semantics (as used for dataset
    labels). The decoder has no learned formula construction.
    """

    def __init__(
        self,
        annotations: Annotations,
        task_names: List[str],
        rule_specs: List[Union[STLRuleSpec, dict]],
        robustness_scale: float = 5.0,
        smoothness: Optional[float] = 10.0,
    ):
        super().__init__()
        if robustness_scale <= 0:
            raise ValueError("robustness_scale must be positive.")
        if smoothness is not None and smoothness <= 0:
            raise ValueError("smoothness must be positive or None.")
        self.smoothness = smoothness
        self.annotations = annotations
        self.task_names = list(task_names)
        self.rules = [
            spec if isinstance(spec, STLRuleSpec) else STLRuleSpec.from_dict(spec)
            for spec in rule_specs
        ]
        if not self.rules:
            raise ValueError("rule_specs must not be empty.")
        unknown_tasks = {rule.task for rule in self.rules}.difference(self.task_names)
        if unknown_tasks:
            raise ValueError(f"Rules reference unknown tasks: {sorted(unknown_tasks)}.")
        self.rule_names = [rule.name for rule in self.rules]
        self.robustness_scale = robustness_scale
        self._indices = {name: index for index, name in enumerate(annotations.labels)}

    def _atom(self, concepts: torch.Tensor, spec: dict) -> torch.Tensor:
        name = spec["concept"]
        if name not in self._indices:
            raise ValueError(f"Rule references unknown concept {name!r}.")
        return concepts[..., self._indices[name]] - float(spec.get("threshold", 0.0))

    def _eventually(self, values: torch.Tensor, start: int, end: int) -> torch.Tensor:
        """``max(values[t + start : t + end + 1])`` for ``t <= T - 1 - end``.

        Only time steps whose whole window fits in the trace are returned, the
        same bounded semantics as the learned SNF decoder and the MIT-BIH labels.
        """
        if not 0 <= start <= end:
            raise ValueError(f"Invalid interval [{start}, {end}].")
        if end >= values.shape[-1]:
            raise ValueError(
                f"Interval end {end} does not fit a trace of length {values.shape[-1]}."
            )
        windows = values.unfold(-1, end + 1, 1)[..., start:]
        return smooth_max(windows, beta=self.smoothness)

    @staticmethod
    def _interval(fields: dict, time: int) -> tuple[int, int]:
        start, end = fields.get("interval", [0, time - 1])
        return int(start), int(end)

    def _robustness(self, concepts: torch.Tensor, rule: STLRuleSpec) -> torch.Tensor:
        fields = rule.fields
        kind = rule.kind
        time = concepts.shape[-2]
        if kind == "always_implies_eventually":
            antecedent = self._atom(concepts, fields["antecedent"])
            consequent = self._atom(concepts, fields["consequent"])
            future = self._eventually(consequent, *self._interval(fields, time))
            beta = self.smoothness
            implication = smooth_or(-antecedent[..., :future.shape[-1]], future, beta)
            return smooth_min(implication, beta=beta)
        if kind == "eventually_and_eventually":
            first = self._atom(concepts, fields["first"])
            second = self._atom(concepts, fields["second"])
            future = self._eventually(second, *self._interval(fields, time))
            beta = self.smoothness
            both = smooth_and(first[..., :future.shape[-1]], future, beta)
            return smooth_max(both, beta=beta)
        if kind == "always_not":
            return smooth_min(-self._atom(concepts, fields["predicate"]), beta=self.smoothness)
        raise ValueError(f"Unknown STL rule kind {kind!r}.")

    def forward(
        self,
        concepts: AnnotatedTensor,
        rule_selection: torch.Tensor,
    ) -> tuple[AnnotatedTensor, torch.Tensor, torch.Tensor]:
        values = concepts.tensor
        robustness_by_task = []
        satisfaction_by_task = []
        for task in self.task_names:
            indices = [index for index, rule in enumerate(self.rules) if rule.task == task]
            if not indices:
                raise ValueError(f"No rules are defined for task {task!r}.")
            task_robustness = torch.stack(
                [self._robustness(values, self.rules[index]) for index in indices],
                dim=-1,
            )
            robustness_by_task.append(task_robustness)
            satisfaction_by_task.append(
                torch.sigmoid(self.robustness_scale * task_robustness)
            )
        robustness = torch.stack(robustness_by_task, dim=-2)
        satisfaction = torch.stack(satisfaction_by_task, dim=-2)
        if rule_selection.shape != satisfaction.shape:
            raise ValueError(
                "rule_selection must have shape "
                f"{tuple(satisfaction.shape)}, got {tuple(rule_selection.shape)}."
            )
        tasks = (rule_selection * satisfaction).sum(dim=-1)
        task_annotations = Annotations(
            labels=self.task_names,
            cardinalities=[1] * len(self.task_names),
            types=["binary"] * len(self.task_names),
        )
        return AnnotatedTensor(tasks, task_annotations), robustness, satisfaction


class ConceptMemoryReasonerTS(nn.Module):
    """Temporal CMR with fixed or differentiable STL rule decoding.

    Concepts are predicted at every time step. Rule selection is global: the
    sequence encoder is pooled over time and produces one rule distribution per
    task and sequence.     Pass ``rule_specs`` for fixed rules or ``snf_config`` for globally learned
    one-clause separated-normal-form rules.

    Args:
        input_size: Number of features at each time step.
        annotations: Concept annotations. Each concept must occupy one scalar
            column; binary and continuous concepts are supported.
        task_names: Task labels whose global rule distributions are produced.
        n_rules: Number of candidate rules per task.
        latent_size: Width of the sequence representation.
        encoder_kind: ``"gru"``, ``"lstm"``, ``"transformer"``, or ``"mlp"``.
        pooling: Temporal pooling for global rule selection: ``"mean"``,
            ``"max"``, or ``"last"``.
        snf_config: Enables the differentiable SNF rule bank when provided.
        stl_smoothness: Inverse temperature of the fixed rules' smooth min/max;
            ``None`` evaluates them with exact hard semantics.
    """

    def __init__(
        self,
        input_size: int,
        annotations: Annotations,
        task_names: Union[List[str], str],
        n_rules: int = 10,
        latent_size: int = 64,
        encoder_kind: str = "gru",
        encoder_hidden_size: Optional[int] = None,
        encoder_layers: int = 1,
        encoder_dropout: float = 0.0,
        transformer_heads: int = 4,
        pooling: str = "mean",
        rule_specs: Optional[List[Union[STLRuleSpec, dict]]] = None,
        robustness_scale: float = 5.0,
        snf_config: Optional[SNFConfig] = None,
        stl_smoothness: Optional[float] = 10.0,
    ):
        super().__init__()
        if isinstance(task_names, str):
            task_names = [task_names]
        self.annotations = annotations
        self.task_names = list(task_names)
        if n_rules <= 0:
            raise ValueError("n_rules must be positive.")
        if pooling not in {"mean", "max", "last"}:
            raise ValueError("pooling must be 'mean', 'max', or 'last'.")
        if any(annotations.concept(name).cardinality != 1 for name in annotations.labels):
            raise ValueError(
                "ConceptMemoryReasonerTS currently supports scalar concepts only."
            )

        self.n_rules = n_rules
        self._discrete = None  # set by discrete_rules()
        self.latent_size = latent_size
        self.pooling = pooling
        self.rule_specs = rule_specs
        if rule_specs is not None and snf_config is not None:
            raise ValueError("Choose rule_specs or snf_config, not both.")
        self.encoder = SequenceEncoder(
            input_size=input_size,
            latent_size=latent_size,
            kind=encoder_kind,
            hidden_size=encoder_hidden_size,
            num_layers=encoder_layers,
            dropout=encoder_dropout,
            n_heads=transformer_heads,
        )
        self.concept_head = nn.Linear(latent_size, len(annotations.labels))
        self.rule_selector = nn.Linear(
            latent_size,
            len(self.task_names) * n_rules,
        )
        self.rule_decoder = (
            SNFRuleDecoder(
                annotations,
                self.task_names,
                n_rules,
                snf_config,
            )
            if snf_config is not None
            else STLRuleDecoder(
                annotations,
                self.task_names,
                rule_specs,
                robustness_scale=robustness_scale,
                smoothness=stl_smoothness,
            )
            if rule_specs is not None
            else None
        )

    def _concept_probabilities(self, values: torch.Tensor) -> torch.Tensor:
        pieces = []
        for index, name in enumerate(self.annotations.labels):
            concept_type = self.annotations.concept(name).type
            column = values[..., index:index + 1]
            pieces.append(
                torch.sigmoid(column)
                if concept_type == "binary"
                else column
            )
        return torch.cat(pieces, dim=-1)

    def _pool(self, latent: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return latent.mean(dim=1)
        if self.pooling == "max":
            return latent.max(dim=1).values
        return latent[:, -1]

    def forward(
        self,
        inputs: torch.Tensor,
        concept_values: Optional[torch.Tensor] = None,
    ) -> CMRTemporalOutput:
        """Predict concept traces, rule selection and tasks.

        Args:
            inputs: ``(batch, time, features)``.
            concept_values: Optional ``(batch, time, concepts)`` traces (e.g.
                ground truth) that replace the predicted concepts as input to
                the rule decoder. ``output.concepts`` still holds the
                predictions, so a concept loss can be computed; the rule
                selection still comes from ``inputs``. This gives the
                "independent" training mode of concept bottleneck models and
                evaluation on ground-truth concepts.
        """
        latent = self.encoder(inputs)
        concepts = AnnotatedTensor(
            self._concept_probabilities(self.concept_head(latent)),
            self.annotations.to_concept_space(),
            axis=-1,
        )
        selector_logits = self.rule_selector(self._pool(latent))
        selector_logits = selector_logits.reshape(
            inputs.shape[0],
            len(self.task_names),
            self.n_rules,
        )
        selector = torch.softmax(selector_logits, dim=-1)
        decoder_concepts = concepts
        if concept_values is not None:
            if concept_values.shape != concepts.shape:
                raise ValueError(
                    f"concept_values must have shape {tuple(concepts.shape)}, "
                    f"got {tuple(concept_values.shape)}."
                )
            decoder_concepts = AnnotatedTensor(
                concept_values.to(concepts.tensor.dtype),
                self.annotations.to_concept_space(),
                axis=-1,
            )
        tasks = None
        if self._discrete is not None:
            selector, rule_robustness, rule_satisfaction, tasks = self._discrete(
                decoder_concepts, selector
            )
        elif self.rule_decoder is not None:
            tasks, rule_robustness, rule_satisfaction = self.rule_decoder(
                decoder_concepts, selector
            )
        else:
            rule_robustness = None
            rule_satisfaction = None
        return CMRTemporalOutput(
            latent=latent,
            concepts=concepts,
            rule_selection=selector,
            rule_robustness=rule_robustness,
            rule_satisfaction=rule_satisfaction,
            tasks=tasks,
        )


class _DiscreteRules:
    """Exact evaluation of one discrete rulebook with discrete rule selection."""

    def __init__(self, model, specs, selection, generator):
        self.model = model
        self.specs = specs
        self.selection = selection
        self.generator = generator
        self._decoders = {}

    def _decoder(self, time: int, device) -> "STLRuleDecoder":
        if time not in self._decoders:
            specs = []
            for spec in self.specs:
                start, end = spec["interval"]
                end = min(end, time - 1)
                specs.append(dict(spec, interval=[min(start, end), end]))
            self._decoders[time] = STLRuleDecoder(
                self.model.annotations, self.model.task_names, specs, smoothness=None
            ).to(device)
        return self._decoders[time]

    def select(self, selector: torch.Tensor) -> torch.Tensor:
        if self.selection == "soft":
            return selector
        if self.selection == "top":
            choice = selector.argmax(-1)
        else:
            flat = selector.detach().cpu().reshape(-1, selector.shape[-1])
            choice = torch.multinomial(flat, 1, generator=self.generator).reshape(
                selector.shape[:-1]
            ).to(selector.device)
        return torch.nn.functional.one_hot(choice, selector.shape[-1]).to(selector.dtype)

    def __call__(self, concepts: AnnotatedTensor, selector: torch.Tensor):
        selector = self.select(selector)
        decoder = self._decoder(concepts.shape[-2], selector.device)
        _, robustness, _ = decoder(concepts, selector)
        scale = self.model.rule_decoder.robustness_scale.detach()
        satisfaction = torch.sigmoid(scale * robustness)
        tasks = (selector * satisfaction).sum(-1)
        task_annotations = Annotations(
            labels=self.model.task_names,
            cardinalities=[1] * len(self.model.task_names),
            types=["binary"] * len(self.model.task_names),
        )
        return selector, robustness, satisfaction, AnnotatedTensor(tasks, task_annotations)


@contextmanager
def discrete_rules(
    model: "ConceptMemoryReasonerTS",
    concepts: str = "argmax",
    selection: str = "top",
    seed: int = 0,
):
    """Inside the block, ``model`` computes exactly with discrete rules.

    The learned SNF bank is replaced by one discrete rulebook
    (:meth:`SNFRuleDecoder.discrete_specs`; ``concepts="argmax"`` or
    ``"sample"``, drawn once on entry), evaluated with exact STL semantics,
    and the rule selection becomes one rule per sample (``selection="top"``
    or ``"sample"``, drawn at every forward call) or stays ``"soft"``. The
    concept head and the selector are unchanged, so concept and rule
    interventions work as usual inside the block.

    Yields:
        The discrete rulebook (a list of rule specs).
    """
    from ..rules.snf import SNFRuleDecoder

    if not isinstance(model.rule_decoder, SNFRuleDecoder):
        raise TypeError("discrete_rules needs a model with a learned SNF rule bank.")
    if selection not in {"top", "sample", "soft"}:
        raise ValueError("selection must be 'top', 'sample' or 'soft'.")
    generator = torch.Generator().manual_seed(seed)
    specs = model.rule_decoder.discrete_specs(concepts, generator)
    previous = model._discrete
    model._discrete = _DiscreteRules(model, specs, selection, generator)
    try:
        yield specs
    finally:
        model._discrete = previous


class SequenceBlackBox(nn.Module):
    """Concept-free baseline: the same encoder with a direct task head.

    ``(batch, time, features)`` -> encoder -> temporal pooling -> MLP ->
    task probabilities. It is the reference for concept completeness: the
    temporal CMR with the same encoder settings should match it if the
    concepts and rules lose no task-relevant information.

    Returns a :class:`CMRTemporalOutput` with ``concepts=None``, so it trains
    with :class:`TemporalCMRLearner` (task loss only).
    """

    def __init__(
        self,
        input_size: int,
        task_names: Union[List[str], str],
        latent_size: int = 64,
        encoder_kind: str = "gru",
        encoder_hidden_size: Optional[int] = None,
        encoder_layers: int = 1,
        encoder_dropout: float = 0.0,
        transformer_heads: int = 4,
        pooling: str = "mean",
        head_hidden_size: Optional[int] = None,
    ):
        super().__init__()
        if isinstance(task_names, str):
            task_names = [task_names]
        if pooling not in {"mean", "max", "last"}:
            raise ValueError("pooling must be 'mean', 'max', or 'last'.")
        self.task_names = list(task_names)
        self.pooling = pooling
        self.annotations = None
        self.rule_decoder = None
        self.encoder = SequenceEncoder(
            input_size=input_size,
            latent_size=latent_size,
            kind=encoder_kind,
            hidden_size=encoder_hidden_size,
            num_layers=encoder_layers,
            dropout=encoder_dropout,
            n_heads=transformer_heads,
        )
        hidden = head_hidden_size or latent_size
        self.head = nn.Sequential(
            nn.Linear(latent_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, len(self.task_names)),
        )
        self._task_annotations = Annotations(
            labels=self.task_names,
            cardinalities=[1] * len(self.task_names),
            types=["binary"] * len(self.task_names),
        )

    _pool = ConceptMemoryReasonerTS._pool

    def forward(self, inputs: torch.Tensor, concept_values=None) -> CMRTemporalOutput:
        latent = self.encoder(inputs)
        tasks = torch.sigmoid(self.head(self._pool(latent)))
        return CMRTemporalOutput(
            latent=latent,
            concepts=None,
            rule_selection=None,
            tasks=AnnotatedTensor(tasks, self._task_annotations),
        )
