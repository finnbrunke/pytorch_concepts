import pytest
import torch

from torch_concepts import AnnotatedTensor
from torch_concepts.nn import STLRuleDecoder, discrete_rules, edit_rule

from test_cmrts_evaluation import RULE, cmr, rule_dataset
from test_cmrts_rules import CRISP


def batch():
    item = rule_dataset(n=32)[0]
    return item["inputs"]["x"], item["temporal_concepts"]


def test_argmax_mode_is_exactly_the_exported_rulebook_with_top_selection():
    model = cmr(n_rules=2).eval()
    x, _ = batch()
    with torch.no_grad():
        soft = model(x)
        with discrete_rules(model, "argmax", "top") as rulebook:
            hard = model(x)
    reference = STLRuleDecoder(model.annotations, ["task"], rulebook, smoothness=None)
    _, robustness, _ = reference(hard.concepts, hard.rule_selection)
    scale = model.rule_decoder.robustness_scale
    expected = (hard.rule_selection * torch.sigmoid(scale * robustness)).sum(-1)

    assert torch.allclose(hard.tasks.tensor, expected)
    assert set(hard.rule_selection.flatten().tolist()) <= {0.0, 1.0}
    assert torch.equal(hard.rule_selection.argmax(-1), soft.rule_selection.argmax(-1))
    assert torch.equal(hard.concepts.tensor, soft.concepts.tensor)
    with torch.no_grad():
        assert torch.equal(model(x).tasks.tensor, soft.tasks.tensor)  # restored


def test_discrete_and_soft_agree_for_a_crisp_rule():
    model = cmr(n_rules=1, snf_config=CRISP).eval()
    x, true = batch()
    with edit_rule(model, "task", 0, RULE), torch.no_grad():
        soft = model(x, concept_values=true).tasks.tensor
        with discrete_rules(model):
            hard = model(x, concept_values=true).tasks.tensor

    assert torch.equal(soft >= 0.5, hard >= 0.5)


def test_sampled_concepts_follow_the_soft_concept_distribution():
    model = cmr(n_rules=1)
    with torch.no_grad():
        model.rule_decoder.antecedent_logits.copy_(
            torch.log(torch.tensor([[[0.8, 0.2]]]))
        )
    picks = [
        model.rule_decoder.discrete_specs("sample", torch.Generator().manual_seed(s))[0]
        ["antecedent"]["concept"]
        for s in range(400)
    ]

    assert picks.count("a") / 400 == pytest.approx(0.8, abs=0.06)
    argmax = {model.rule_decoder.discrete_specs("argmax")[0]["antecedent"]["concept"]
              for _ in range(5)}
    assert argmax == {"a"}


def test_sampled_selection_follows_the_selector_distribution():
    model = cmr(n_rules=2).eval()
    x, _ = batch()
    with torch.no_grad():
        model.rule_selector.weight.zero_()
        model.rule_selector.bias.copy_(torch.log(torch.tensor([0.7, 0.3])))
        with discrete_rules(model, "argmax", "sample", seed=0):
            choices = torch.cat([model(x).rule_selection[:, 0].argmax(-1) for _ in range(20)])

    assert (choices == 0).float().mean().item() == pytest.approx(0.7, abs=0.06)


def test_intervals_are_capped_to_the_trace():
    model = cmr(n_rules=1)
    specs = model.rule_decoder.discrete_specs(max_end=3)
    assert specs[0]["interval"][1] <= 3 and specs[0]["interval"][0] <= specs[0]["interval"][1]
    with torch.no_grad():
        model.rule_decoder.width_logits.fill_(30.0)  # end -> max_horizon (10)
        model.rule_decoder.start_logits.fill_(-30.0)
        with discrete_rules(model):
            out = model(torch.randn(2, 10, 2))  # trace of 10 steps: end capped at 9
    assert out.tasks.tensor.shape == (2, 1)


def test_discrete_rules_rejects_fixed_rules_and_bad_modes():
    with pytest.raises(TypeError):
        with discrete_rules(cmr(snf_config=None, rule_specs=[RULE])):
            pass
    with pytest.raises(ValueError):
        with discrete_rules(cmr(), selection="mean"):
            pass
    with pytest.raises(ValueError):
        with discrete_rules(cmr(), concepts="max"):
            pass
