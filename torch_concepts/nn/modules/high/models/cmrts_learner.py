"""Lightning training wrapper for :class:`ConceptMemoryReasonerTS`."""

from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics.functional.classification import (
    binary_accuracy,
    binary_auroc,
    binary_f1_score,
)

from .cmrts import CMRTemporalOutput, ConceptMemoryReasonerTS


class TemporalCMRLearner(pl.LightningModule):
    """Train a temporal CMR on batches from a temporal concept dataset.

    Expects batches with ``batch["inputs"]["x"]`` ``(batch, time, features)``,
    ``batch["temporal_concepts"]`` ``(batch, time, concepts)`` and
    ``batch["robustness"]`` ``(batch, tasks)``; a task is positive when its
    robustness is non-negative.

    Loss: task BCE + ``concept_weight`` × concept loss, where binary concepts
    use BCE and continuous concepts MSE.

    Logged per stage (``train``/``val``/``test``), per epoch:
    ``{stage}/loss``, ``{stage}/task_loss``, ``{stage}/concept_loss``, and per
    task ``{stage}/{metric}/{task}`` for accuracy, F1 and AUROC, plus
    ``{stage}/{metric}_macro``. ``{stage}/majority_acc/{task}`` is the
    accuracy of always predicting the most common label, for reference.
    """

    def __init__(
        self,
        model: ConceptMemoryReasonerTS,
        learning_rate: float = 3e-3,
        concept_weight: float = 1.0,
    ):
        super().__init__()
        if model.rule_decoder is None:
            raise ValueError("The model needs rule_specs or snf_config to predict tasks.")
        self.model = model
        self.learning_rate = learning_rate
        self.concept_weight = concept_weight
        types = [model.annotations.concept(name).type for name in model.annotations.labels]
        self.register_buffer(
            "_binary_concepts",
            torch.tensor([kind == "binary" for kind in types]),
            persistent=False,
        )
        self._epoch_outputs: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {}

    @property
    def task_names(self) -> List[str]:
        return self.model.task_names

    def forward(self, inputs: torch.Tensor) -> CMRTemporalOutput:
        return self.model(inputs)

    @staticmethod
    def unpack(batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(inputs, temporal concepts, binary task targets)`` from a batch."""
        concepts = batch["temporal_concepts"]
        concepts = getattr(concepts, "tensor", concepts)
        return (
            batch["inputs"]["x"].float(),
            concepts.float(),
            (batch["robustness"] >= 0).float(),
        )

    def concept_loss(self, predicted: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        binary = self._binary_concepts
        losses = []
        if binary.any():
            losses.append(
                F.binary_cross_entropy(
                    predicted[..., binary].clamp(1e-6, 1 - 1e-6), true[..., binary]
                )
            )
        if (~binary).any():
            losses.append(F.mse_loss(predicted[..., ~binary], true[..., ~binary]))
        return torch.stack(losses).mean()

    def _step(self, batch, stage: str) -> torch.Tensor:
        inputs, concepts, targets = self.unpack(batch)
        output = self(inputs)
        probabilities = output.tasks.tensor
        task_loss = F.binary_cross_entropy(probabilities.clamp(1e-6, 1 - 1e-6), targets)
        concept_loss = self.concept_loss(output.concepts.tensor, concepts)
        loss = task_loss + self.concept_weight * concept_loss
        size = inputs.shape[0]
        options = dict(on_step=False, on_epoch=True, batch_size=size)
        self.log(f"{stage}/loss", loss, prog_bar=stage != "test", **options)
        self.log(f"{stage}/task_loss", task_loss, **options)
        self.log(f"{stage}/concept_loss", concept_loss, **options)
        self._epoch_outputs.setdefault(stage, []).append(
            (probabilities.detach().cpu(), targets.detach().cpu())
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    @staticmethod
    def task_metrics(
        probabilities: torch.Tensor, targets: torch.Tensor, task_names: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Per-task accuracy, F1, AUROC and majority-class accuracy."""
        metrics: Dict[str, Dict[str, float]] = {}
        labels = targets.long()
        for index, task in enumerate(task_names):
            p, y = probabilities[:, index], labels[:, index]
            positive_rate = y.float().mean().item()
            metrics[task] = {
                "acc": binary_accuracy(p, y).item(),
                "f1": binary_f1_score(p, y).item(),
                "majority_acc": max(positive_rate, 1.0 - positive_rate),
                "positive_rate": positive_rate,
            }
            # AUROC is undefined when only one class is present.
            if 0.0 < positive_rate < 1.0:
                metrics[task]["auroc"] = binary_auroc(p, y).item()
        return metrics

    def _log_epoch_metrics(self, stage: str) -> Optional[Dict[str, Dict[str, float]]]:
        outputs = self._epoch_outputs.pop(stage, [])
        if not outputs or (self.trainer is not None and self.trainer.sanity_checking):
            return None
        probabilities = torch.cat([p for p, _ in outputs])
        targets = torch.cat([t for _, t in outputs])
        metrics = self.task_metrics(probabilities, targets, self.task_names)
        logged = {}
        for task, values in metrics.items():
            for name, value in values.items():
                if name != "positive_rate":
                    logged[f"{stage}/{name}/{task}"] = value
        for name in ("acc", "f1", "auroc"):
            values = [m[name] for m in metrics.values() if name in m]
            if values:
                logged[f"{stage}/{name}_macro"] = sum(values) / len(values)
        self.log_dict(logged)
        return metrics

    def on_train_epoch_end(self):
        self._log_epoch_metrics("train")

    def on_validation_epoch_end(self):
        self._log_epoch_metrics("val")

    def on_test_epoch_end(self):
        self.test_metrics = self._log_epoch_metrics("test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
