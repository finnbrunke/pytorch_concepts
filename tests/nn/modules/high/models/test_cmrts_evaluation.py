import numpy as np
import pytest
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from torch_concepts import Annotations
from torch_concepts.data.datasets.mit_bih import evaluate_rule
from torch_concepts.nn import (
    ConceptMemoryReasonerTS,
    SNFConfig,
    SequenceBlackBox,
    TemporalCMRLearner,
    collect_predictions,
    concept_completeness,
    rule_language_completeness,
)
from torch_concepts.nn.modules.high.models.cmrts_evaluation import Predictions

ANNOTATIONS = Annotations(labels=["a", "b"], cardinalities=[1, 1], types=["binary"] * 2)
RULE = {
    "name": "a_then_b", "task": "task", "kind": "always_implies_eventually",
    "antecedent": {"concept": "a", "threshold": 0.5},
    "consequent": {"concept": "b", "threshold": 0.5},
    "interval": [1, 4],
}


def rule_dataset(n=96, time=20, seed=0):
    """Binary traces labelled by ``RULE``, as batch dicts."""
    rng = np.random.default_rng(seed)
    concepts = (rng.random((n, time, 2)) < 0.12).astype(np.float32)
    robustness = np.array(
        [[evaluate_rule(trace, RULE, ["a", "b"])] for trace in concepts], dtype=np.float32
    )
    inputs = concepts + 0.1 * rng.standard_normal(concepts.shape).astype(np.float32)
    batches = []
    for start in range(0, n, 32):
        part = slice(start, start + 32)
        batches.append({
            "inputs": {"x": torch.from_numpy(inputs[part])},
            "temporal_concepts": torch.from_numpy(concepts[part]),
            "robustness": torch.from_numpy(robustness[part]),
        })
    return batches


def cmr(**options):
    defaults = dict(n_rules=1, latent_size=8, snf_config=SNFConfig(max_horizon=10))
    defaults.update(options)
    return ConceptMemoryReasonerTS(2, ANNOTATIONS, ["task"], **defaults)


def test_concept_values_replace_the_decoder_input_but_not_the_predictions():
    model = cmr().eval()
    x = torch.randn(3, 20, 2)
    values = torch.ones(3, 20, 2)
    with torch.no_grad():
        plain, replaced = model(x), model(x, concept_values=values)

    assert torch.equal(plain.concepts.tensor, replaced.concepts.tensor)
    assert not torch.allclose(plain.tasks.tensor, replaced.tasks.tensor)
    with pytest.raises(ValueError, match="concept_values must have shape"):
        model(x, concept_values=torch.ones(3, 19, 2))


def test_black_box_has_the_same_interface_and_trains_on_the_task_loss(tmp_path):
    model = SequenceBlackBox(2, ["task"], latent_size=8)
    output = model(torch.randn(4, 20, 2))

    assert output.tasks.tensor.shape == (4, 1) and output.concepts is None
    batches = rule_dataset(n=64)
    trainer = Trainer(max_epochs=1, logger=False, enable_checkpointing=False,
                      enable_progress_bar=False, enable_model_summary=False,
                      default_root_dir=tmp_path)
    loader = DataLoader(batches, batch_size=None)  # already batched
    trainer.fit(TemporalCMRLearner(model), loader, loader)
    assert trainer.callback_metrics["train/concept_loss"] == 0


def test_independent_training_decodes_ground_truth_concepts(monkeypatch):
    learner = TemporalCMRLearner(cmr(), concept_source="ground_truth")
    seen = []
    original = learner.model.forward

    def spy(inputs, concept_values=None):
        seen.append(concept_values)
        return original(inputs, concept_values=concept_values)

    monkeypatch.setattr(learner.model, "forward", spy)
    monkeypatch.setattr(learner, "log", lambda *a, **k: None)
    batch = rule_dataset(n=32)[0]
    learner._step(batch, "train")
    learner._step(batch, "val")

    assert torch.equal(seen[0], batch["temporal_concepts"])
    assert seen[1] is None


def test_independent_training_needs_concepts():
    with pytest.raises(ValueError, match="needs a model with concepts"):
        TemporalCMRLearner(SequenceBlackBox(2, ["task"]), concept_source="ground_truth")


def test_concept_completeness_ratio_and_normalised_form():
    targets = torch.tensor([[0.0, 1.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    black = Predictions(torch.tensor([[0.1, .9], [0.2, .9], [0.8, .9], [0.9, .9]]), targets)
    white = Predictions(torch.tensor([[0.1, .9], [0.6, .9], [0.5, .9], [0.9, .9]]), targets)
    table = concept_completeness(white, black, ["y", "constant"])

    assert set(table) == {"y", "macro"}  # single-class task skipped
    assert table["y"]["reference"] == 1.0 and table["y"]["whitebox"] == 0.75
    assert table["y"]["completeness"] == pytest.approx(0.75)
    assert table["y"]["normalized_completeness"] == pytest.approx(0.5)
    accuracy = concept_completeness(white, black, ["y", "constant"], scorer="accuracy")
    assert accuracy["y"]["random"] == 0.5 and accuracy["y"]["whitebox"] == 0.75


def test_concept_completeness_requires_matching_samples():
    a = Predictions(torch.rand(4, 1), torch.tensor([[0.0], [1.0], [0.0], [1.0]]))
    b = Predictions(torch.rand(4, 1), torch.tensor([[1.0], [1.0], [0.0], [1.0]]))
    with pytest.raises(ValueError, match="share their targets"):
        concept_completeness(a, b, ["y"])


def test_collect_predictions_uses_ground_truth_concepts_when_asked():
    batches = rule_dataset()
    exact = cmr(snf_config=None, rule_specs=[RULE], stl_smoothness=None)
    predicted = collect_predictions(exact, batches)
    truth = collect_predictions(exact, batches, "ground_truth", keep_concepts=True)

    assert truth.probabilities.shape == (96, 1) and truth.true_concepts.shape == (96, 20, 2)
    labels = truth.targets[:, 0].bool()
    assert torch.equal(truth.probabilities[:, 0] >= 0.5, labels)
    assert not torch.equal(predicted.probabilities, truth.probabilities)


def test_rule_language_completeness_is_one_for_the_generating_rules():
    batches = rule_dataset()
    exact = cmr(snf_config=None, rule_specs=[RULE], stl_smoothness=None)
    table = rule_language_completeness(exact, batches, reference_rules=[RULE])

    assert table["task"]["reference"] == 1.0
    assert table["task"]["completeness"] == pytest.approx(1.0)
    default = rule_language_completeness(exact, batches)
    assert default["task"]["completeness"] == pytest.approx(1.0)


def test_rule_language_completeness_drops_for_rules_that_differ():
    batches = rule_dataset()
    wrong = dict(RULE, antecedent={"concept": "b", "threshold": 0.5},
                 consequent={"concept": "a", "threshold": 0.5})
    model = cmr(snf_config=None, rule_specs=[wrong], stl_smoothness=None)
    table = rule_language_completeness(model, batches, reference_rules=[RULE])

    assert table["task"]["completeness"] < 0.8


def test_event_scores_count_runs_with_a_tolerance():
    from torch_concepts.nn import event_scores

    true = torch.zeros(1, 30)
    true[0, 5:8] = 1   # event 1
    true[0, 20:22] = 1  # event 2 (missed)
    predicted = torch.zeros(1, 30)
    predicted[0, 8] = 0.9    # 1 step after event 1 (steps 5-7)
    predicted[0, 14] = 0.9   # false positive
    exact = event_scores(predicted, true, tolerance=0)
    loose = event_scores(predicted, true, tolerance=1)

    assert exact["event_recall"] == 0.0 and exact["true_events"] == 2
    assert loose["event_recall"] == 0.5 and loose["event_precision"] == 0.5
    assert loose["event_f1"] == pytest.approx(0.5)


def test_concept_scores_cover_binary_and_continuous_concepts():
    from torch_concepts.nn import concept_scores

    annotations = Annotations(labels=["a", "b"], cardinalities=[1, 1], types=["binary", "continuous"])
    model = ConceptMemoryReasonerTS(2, annotations, ["task"], n_rules=1, latent_size=8,
                                    snf_config=SNFConfig(max_horizon=5))
    true = torch.zeros(4, 10, 2)
    true[:, 3, 0] = 1
    true[..., 1] = torch.linspace(0, 1, 10)
    predicted = true.clone()
    predicted[..., 1] += 0.1
    table = concept_scores(model, Predictions(torch.zeros(4, 1), torch.zeros(4, 1), predicted, true))

    assert table["a"]["auroc"] == 1.0 and table["a"]["event_f1"] == 1.0
    assert table["b"]["mse"] == pytest.approx(0.01) and table["b"]["r2"] < 1


def test_rule_compactness_detects_duplicates_and_crisp_rules():
    from torch_concepts.nn import edit_rule, rule_compactness

    model = cmr(n_rules=2).eval()
    with edit_rule(model, "task", 0, RULE), edit_rule(model, "task", 1, RULE):
        table = rule_compactness(model, rule_dataset())

    row = table["task"]
    assert row["duplicate_rules"] == 1.0
    assert row["concept_entropy"] < 0.01
    assert 0.0 <= row["selection_entropy"] <= 1.0 and row["rules_used"] in (1.0, 2.0)
