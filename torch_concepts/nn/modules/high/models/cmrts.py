"""Sequence-to-sequence building blocks for temporal CMR models.

The temporal model deliberately keeps STL rule decoding separate from the
sequence representation.  This module provides the reusable parts around that
decoder: sequence encoding, per-time-step concept annotation, and global rule
selection.
"""

from dataclasses import dataclass
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from torch import nn

from .....annotations import Annotations
from .....tensor import AnnotatedTensor


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
    concepts: AnnotatedTensor
    rule_selection: torch.Tensor
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

    Robustness is evaluated at the beginning of each trace.  Bounded
    ``Always`` and ``Eventually`` operators use differentiable ``min`` and
    ``max`` reductions over their temporal windows.  The decoder intentionally
    has no learned formula construction in stage 1.
    """

    def __init__(
        self,
        annotations: Annotations,
        task_names: List[str],
        rule_specs: List[Union[STLRuleSpec, dict]],
        robustness_scale: float = 1.0,
    ):
        super().__init__()
        if robustness_scale <= 0:
            raise ValueError("robustness_scale must be positive.")
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

    @staticmethod
    def _window(values: torch.Tensor, start: int, end: int, reduction: str) -> torch.Tensor:
        length = max(1, end - start + 1)
        values = values[..., start:]
        if values.shape[-1] < length:
            values = torch.nn.functional.pad(values, (0, length - values.shape[-1]), value=0.0)
        windows = values.unfold(-1, length, 1)
        reduced = windows.min(-1).values if reduction == "min" else windows.max(-1).values
        return reduced[..., : max(1, min(end + 1, reduced.shape[-1]))]

    def _robustness(self, concepts: torch.Tensor, rule: STLRuleSpec) -> torch.Tensor:
        fields = rule.fields
        kind = rule.kind
        if kind == "always_implies_eventually":
            antecedent = self._atom(concepts, fields["antecedent"])
            consequent = self._atom(concepts, fields["consequent"])
            end = int(fields.get("interval", [0, concepts.shape[-1] - 1])[1])
            future = self._window(consequent, 0, end, "max")
            implication = torch.maximum(-antecedent[..., :future.shape[-1]], future)
            return implication.min(dim=-1).values
        if kind == "eventually_and_eventually":
            first = self._atom(concepts, fields["first"])
            second = self._atom(concepts, fields["second"])
            end = int(fields.get("interval", [0, concepts.shape[-1] - 1])[1])
            future = self._window(second, 0, end, "max")
            return torch.minimum(first[..., :future.shape[-1]], future).max(dim=-1).values
        if kind == "always_not":
            return (-self._atom(concepts, fields["predicate"])).min(dim=-1).values
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
    """Generic temporal representation for a future STL-based CMR decoder.

    Concepts are predicted at every time step. Rule selection is global: the
    sequence encoder is pooled over time and produces one rule distribution per
    task and sequence. STL rule decoding is intentionally not implemented here;
    future decoders can consume ``concepts`` and ``rule_selection``.

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
        robustness_scale: float = 1.0,
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
        self.latent_size = latent_size
        self.pooling = pooling
        self.rule_specs = rule_specs
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
            STLRuleDecoder(
                annotations,
                self.task_names,
                rule_specs,
                robustness_scale=robustness_scale,
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

    def forward(self, inputs: torch.Tensor) -> CMRTemporalOutput:
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
        tasks = None
        if self.rule_decoder is not None:
            tasks, rule_robustness, rule_satisfaction = self.rule_decoder(
                concepts, selector
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
