import pytest
import torch

from torch_concepts.nn import (
    SNFConfig,
    edit_rule,
    exported_rule_specs,
    force_rule,
    remove_rule,
    rule_fidelity,
    rule_intervention_scores,
)

from test_cmrts_evaluation import RULE, cmr, rule_dataset

CRISP = SNFConfig(max_horizon=10, smoothness=50.0, time_temperature=0.05,
                  learn_robustness_scale=False, robustness_scale=20.0)


def two_tasks(n_rules=3):
    from torch_concepts import Annotations
    from torch_concepts.nn import ConceptMemoryReasonerTS

    annotations = Annotations(labels=["a", "b"], cardinalities=[1, 1], types=["binary"] * 2)
    return ConceptMemoryReasonerTS(
        2, annotations, ["t1", "t2"], n_rules=n_rules, latent_size=8,
        snf_config=SNFConfig(max_horizon=10),
    ).eval()


def test_force_rule_makes_the_selection_one_hot_for_that_task_only():
    model, x = two_tasks(), torch.randn(4, 20, 2)
    with torch.no_grad():
        base = model(x)
        with force_rule(model, "t1", 2):
            forced = model(x)

    assert torch.allclose(forced.rule_selection[:, 0], torch.tensor([0.0, 0.0, 1.0]).expand(4, 3), atol=1e-6)
    assert torch.allclose(forced.rule_selection[:, 1], base.rule_selection[:, 1])


def test_remove_rule_renormalises_over_the_remaining_rules():
    model, x = two_tasks(), torch.randn(4, 20, 2)
    with torch.no_grad():
        base = model(x)
        with remove_rule(model, "t2", 0):
            removed = model(x)

    assert torch.allclose(removed.rule_selection[:, 1, 0], torch.zeros(4), atol=1e-6)
    assert torch.allclose(removed.rule_selection.sum(-1), torch.ones(4, 2))
    expected = base.rule_selection[:, 1, 1:] / base.rule_selection[:, 1, 1:].sum(-1, keepdim=True)
    assert torch.allclose(removed.rule_selection[:, 1, 1:], expected, atol=1e-5)
    with pytest.raises(ValueError, match="only rule"):
        with remove_rule(two_tasks(n_rules=1), "t1", 0):
            pass


def test_edit_rule_writes_the_expert_rule_and_restores_the_learned_one():
    model = two_tasks()
    before = model.rule_decoder.export_rules()
    with edit_rule(model, "t2", 1, RULE):
        rule = model.rule_decoder.export_rules()[1 * 3 + 1]
        assert (rule["antecedent"]["concept"], rule["consequent"]["concept"]) == ("a", "b")
        assert rule["antecedent"]["threshold"] == pytest.approx(0.5, abs=1e-4)
        assert rule["interval"] == pytest.approx([1.0, 4.0], abs=1e-3)
    assert model.rule_decoder.export_rules() == before


def test_edit_rule_rejects_what_the_bank_cannot_represent():
    model = two_tasks()
    with pytest.raises(ValueError, match="only represents"):
        with edit_rule(model, "t1", 0, dict(RULE, kind="always_not")):
            pass
    with pytest.raises(ValueError, match="interval must satisfy"):
        with edit_rule(model, "t1", 0, dict(RULE, interval=[5, 5.5])):
            pass
    with pytest.raises(TypeError):
        with edit_rule(cmr(snf_config=None, rule_specs=[RULE]), "task", 0, RULE):
            pass


def test_exported_specs_use_integer_intervals_inside_the_soft_window():
    model = two_tasks()
    for spec, rule in zip(exported_rule_specs(model), model.rule_decoder.export_rules()):
        start, end = spec["interval"]
        assert isinstance(start, int) and isinstance(end, int) and start <= end
        assert (start, end) == (round(rule["interval"][0]), max(start, round(rule["interval"][1])))


def test_rule_fidelity_is_perfect_for_a_crisp_rule_on_true_concepts():
    batches = rule_dataset()
    model = cmr(n_rules=2, snf_config=CRISP).eval()
    with edit_rule(model, "task", 0, RULE), edit_rule(model, "task", 1, RULE):
        table = rule_fidelity(model, batches, concept_source="ground_truth")

    row = table["task"]
    assert row["top_rule_score"] == pytest.approx(1.0)
    assert row["soft_selection_score"] == pytest.approx(1.0)
    assert row["top_rule_agreement"] > 0.95 and row["model_score"] > 0.95
    assert set(table["macro"]) == set(row)


def test_rule_intervention_scores_report_each_rule_alone():
    batches = rule_dataset()
    model = cmr(n_rules=2, snf_config=CRISP).eval()
    with edit_rule(model, "task", 1, RULE):
        table = rule_intervention_scores(model, batches)

    row = table["task"]
    assert len(row["forced"]) == 2 and len(row["removed"]) == 2
    assert row["gain"] == pytest.approx(max(row["forced"]) - row["selected"])
    assert row["best_rule"] in (0, 1)
