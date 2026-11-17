"""Small end-to-end training example for the temporal CMR model.

Run from the repository root:

    python -m examples.high_level.12_cmr_ts_training

The example generates a small predator-prey dataset, trains a compact GRU
with the fixed named STL rule bank, and applies it to a held-out sequence.
"""

import torch
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, Subset

from torch_concepts import seed_everything
from torch_concepts.data.datasets.predator_prey import PredatorPreyDataset
from torch_concepts.nn import ConceptMemoryReasonerTS


class TemporalCMRLightning(LightningModule):
    """Lightning training wrapper for the plain CMR-TS module."""

    def __init__(self, model: ConceptMemoryReasonerTS, learning_rate: float = 3e-3):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

    def forward(self, inputs: torch.Tensor):
        return self.model(inputs)

    @staticmethod
    def _targets(batch) -> torch.Tensor:
        return (batch["robustness"] >= 0).float()

    def training_step(self, batch, batch_idx):
        output = self(batch["inputs"]["x"].float())
        loss = torch.nn.functional.binary_cross_entropy(
            output.tasks.tensor,
            self._targets(batch),
        )
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        output = self(batch["inputs"]["x"].float())
        target = self._targets(batch).bool()
        prediction = output.tasks.tensor.ge(0.5)
        self.log("test_accuracy", (prediction == target).float().mean())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def main() -> None:
    seed_everything(7)

    dataset = PredatorPreyDataset(num_samples=96, time_steps=40, dt=0.1)
    split = int(0.8 * len(dataset))
    trainset = Subset(dataset, range(split))
    testset = Subset(dataset, range(split, len(dataset)))
    train_loader = DataLoader(
        trainset,
        batch_size=16,
        shuffle=True,
        collate_fn=dataset.collate,
    )
    test_loader = DataLoader(
        testset,
        batch_size=len(testset),
        collate_fn=dataset.collate,
    )

    model = ConceptMemoryReasonerTS(
        input_size=2,
        annotations=dataset.concept_annotations,
        task_names=["prey_recovery", "predator_crash", "system_stability"],
        n_rules=1,
        latent_size=16,
        encoder_kind="gru",
        rule_specs=dataset.rulebook,
    )
    lightning_model = TemporalCMRLightning(model)
    trainer = Trainer(
        max_epochs=8,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    trainer.fit(lightning_model, train_dataloaders=train_loader)
    trainer.test(lightning_model, dataloaders=test_loader, verbose=False)

    lightning_model.eval()
    with torch.no_grad():
        batch = next(iter(test_loader))
        output = lightning_model(batch["inputs"]["x"].float())
        predicted = output.tasks.tensor >= 0.5
        target = batch["robustness"] >= 0
        accuracy = (predicted == target).float().mean().item()

    print("\nheld-out example")
    print("concept names:", batch["concept_names"])
    print("concept trace shape:", tuple(output.concepts.shape))
    print("rule robustness:", output.rule_robustness[0].tolist())
    print("rule satisfaction:", output.rule_satisfaction[0].tolist())
    print("rule names:", [rule["name"] for rule in dataset.rulebook])
    print("task probabilities:", output.tasks.tensor[0].tolist())
    print("target / prediction:", target[0].tolist(), predicted[0].tolist())
    print(f"held-out accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    main()
