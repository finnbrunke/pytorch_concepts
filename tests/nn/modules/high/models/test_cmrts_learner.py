from pathlib import Path

import pytest
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers.logger import Logger
from torch.utils.data import DataLoader, Subset

from torch_concepts.data.datasets.predator_prey import PredatorPreyDataset
from torch_concepts.nn import (
    ConceptMemoryReasonerTS,
    SNFConfig,
    TemporalCMRLearner,
    TemporalCMRMonitor,
)

TASKS = ["prey_recovery", "predator_crash", "system_stability"]


class ImageLogger(Logger):
    """Minimal logger with the ``log_image`` API of ``WandbLogger``."""

    def __init__(self):
        super().__init__()
        self.images, self.metrics = [], {}

    @property
    def name(self):
        return "images"

    @property
    def version(self):
        return 0

    def log_hyperparams(self, params, *args, **kwargs):
        pass

    def log_metrics(self, metrics, step=None):
        self.metrics.update(metrics)

    def log_image(self, key, images, step=None, **kwargs):
        self.images.extend((key, image, step) for image in images)


@pytest.fixture(scope="module")
def loaders():
    seed_everything(0)
    dataset = PredatorPreyDataset(num_samples=24, time_steps=20, dt=0.1)
    train = DataLoader(Subset(dataset, range(16)), batch_size=8, collate_fn=dataset.collate)
    val = DataLoader(Subset(dataset, range(16, 24)), batch_size=8, collate_fn=dataset.collate)
    return dataset, train, val


def learner(dataset, **model_options):
    options = dict(snf_config=SNFConfig(max_horizon=20))
    options.update(model_options)
    model = ConceptMemoryReasonerTS(
        input_size=2,
        annotations=dataset.concept_annotations,
        task_names=TASKS,
        n_rules=2,
        latent_size=8,
        **options,
    )
    return TemporalCMRLearner(model)


def trainer(tmp_path, callbacks, logger, epochs=2):
    return Trainer(
        max_epochs=epochs,
        default_root_dir=tmp_path,
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )


def test_learner_logs_losses_and_per_task_metrics(loaders, tmp_path):
    dataset, train, val = loaders
    module = learner(dataset)
    fit = trainer(tmp_path, [], CSVLogger(tmp_path))
    fit.fit(module, train, val)
    fit_metrics = dict(fit.callback_metrics)
    fit.test(module, val, verbose=False)
    test_metrics = dict(fit.callback_metrics)

    for stage, metrics in (("train", fit_metrics), ("val", fit_metrics), ("test", test_metrics)):
        for name in ("loss", "task_loss", "concept_loss", "acc_macro", "f1_macro"):
            assert f"{stage}/{name}" in metrics
        for task in TASKS:
            assert f"{stage}/acc/{task}" in metrics
            assert f"{stage}/majority_acc/{task}" in metrics
    assert set(module.test_metrics) == set(TASKS)


def test_learner_uses_bce_for_binary_and_mse_for_continuous_concepts(loaders):
    dataset, _, _ = loaders
    module = learner(dataset)
    predicted = torch.full((2, 5, 4), 0.5)

    assert not module._binary_concepts.any()  # predator-prey concepts are continuous
    assert module.concept_loss(predicted, torch.zeros(2, 5, 4)).item() == pytest.approx(0.25)


def test_learner_requires_a_rule_decoder(loaders):
    dataset, _, _ = loaders
    model = ConceptMemoryReasonerTS(
        input_size=2, annotations=dataset.concept_annotations, task_names=TASKS,
    )
    with pytest.raises(ValueError, match="rule_specs or snf_config"):
        TemporalCMRLearner(model)


def test_task_metrics_skip_auroc_for_a_single_class():
    probabilities = torch.tensor([[0.9, 0.2], [0.8, 0.7], [0.6, 0.1], [0.3, 0.9]])
    targets = torch.tensor([[1.0, 0.0], [1.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    metrics = TemporalCMRLearner.task_metrics(probabilities, targets, ["a", "b"])

    assert "auroc" not in metrics["a"]
    assert metrics["a"]["majority_acc"] == 1.0 and metrics["a"]["acc"] == 0.75
    assert metrics["b"]["auroc"] == 1.0 and metrics["b"]["f1"] == 1.0


def test_monitor_sends_scalars_and_figures_to_an_image_logger(loaders, tmp_path):
    dataset, train, val = loaders
    logger = ImageLogger()
    fit = trainer(tmp_path, [TemporalCMRMonitor(n_examples=2)], logger)
    fit.fit(learner(dataset), train, val)

    keys = [key for key, _, _ in logger.images]
    assert keys.count("plots/concept_traces") == 2
    assert keys.count("plots/concept_traces_rare") == 2
    assert keys.count("plots/rule_bank") == 2
    assert keys.count("plots/rule_history") == 1
    assert all(image.ndim == 3 and image.shape[-1] == 3 for _, image, _ in logger.images)
    assert "rules/robustness_scale" in logger.metrics
    for task in TASKS:
        for suffix in ("start", "end", "selection"):
            assert f"rules/{task}/r1/{suffix}" in logger.metrics
    assert not (tmp_path / "plots").exists()


def test_monitor_saves_pngs_when_no_logger_takes_images(loaders, tmp_path):
    dataset, train, val = loaders
    callback = TemporalCMRMonitor(every_n_epochs=2)
    fit = trainer(tmp_path, [callback], CSVLogger(tmp_path), epochs=3)
    fit.fit(learner(dataset), train, val)
    plots = Path(fit.log_dir) / "plots"
    assert plots.parent != tmp_path  # inside the logger's run directory
    files = sorted(path.name for path in plots.iterdir())

    # every_n_epochs=2 -> epochs 0 and 2 (the last epoch is always plotted).
    assert files == [
        "concept_traces_epoch000.png", "concept_traces_epoch002.png",
        "concept_traces_rare_epoch000.png", "concept_traces_rare_epoch002.png",
        "rule_bank_epoch000.png", "rule_bank_epoch002.png",
        "rule_history_epoch002.png",
    ]
    assert len(callback.history["epoch"]) == 3


def test_monitor_handles_fixed_rules_without_a_rule_bank(loaders, tmp_path):
    dataset, train, val = loaders
    rules = [
        {"name": f"r_{task}", "task": task, "kind": "always_not",
         "predicate": {"concept": "system_crash", "threshold": 0.5}}
        for task in TASKS
    ]
    module = learner(dataset, snf_config=None, rule_specs=rules)
    module.model.n_rules = 1
    module.model.rule_selector = torch.nn.Linear(8, len(TASKS))
    logger = ImageLogger()
    fit = trainer(tmp_path, [TemporalCMRMonitor()], logger, epochs=1)
    fit.fit(module, train, val)

    assert {key for key, _, _ in logger.images} <= {
        "plots/concept_traces", "plots/concept_traces_rare"
    }
    assert "plots/rule_bank" not in {key for key, _, _ in logger.images}


def test_select_rare_picks_one_minority_sequence_per_task():
    targets = torch.tensor([
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],  # the only positive for task a
        [0.0, 0.0, 1.0],  # the only negative for task b
        [1.0, 1.0, 1.0],
    ])
    chosen = TemporalCMRMonitor.select_rare(targets, ["a", "b", "c"])

    assert [(index, task, label) for index, task, label, _ in chosen] == [
        (2, "a", 1), (3, "b", 0)
    ]  # task c has a single class and is skipped
    assert chosen[0][3] == pytest.approx(0.4)


def test_select_rare_uses_distinct_sequences():
    targets = torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    chosen = TemporalCMRMonitor.select_rare(targets, ["a", "b"])

    assert [index for index, _, _, _ in chosen] == [0, 3]
