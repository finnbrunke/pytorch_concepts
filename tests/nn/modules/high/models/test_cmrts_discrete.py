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


def test_straight_through_selection_picks_one_rule_and_trains_the_selector():
    model = cmr(n_rules=3, selection="straight_through")
    x, _ = batch()
    out = model(x)  # training mode: Gumbel sample
    assert set(out.rule_selection.flatten().tolist()) <= {0.0, 1.0}
    assert torch.equal(out.rule_selection.sum(-1), torch.ones(32, 1))
    out.tasks.tensor.sum().backward()
    assert model.rule_selector.weight.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        logits = model.rule_selector(model._pool(model.encoder(x))).reshape(32, 1, 3)
        assert torch.equal(model(x).rule_selection.argmax(-1), logits.argmax(-1))
    with pytest.raises(ValueError, match="selection"):
        cmr(selection="gumbel")


def test_bidirectional_encoder_keeps_the_latent_size():
    from torch_concepts.nn import SequenceEncoder

    encoder = SequenceEncoder(2, 16, kind="gru", bidirectional=True)
    assert encoder(torch.randn(3, 7, 2)).shape == (3, 7, 16)
    # A bidirectional step depends on later inputs.
    x = torch.randn(1, 7, 2, requires_grad=True)
    encoder(x)[0, 0].sum().backward()
    assert x.grad[0, -1].abs().sum() > 0
    with pytest.raises(ValueError, match="even"):
        SequenceEncoder(2, 15, kind="lstm", bidirectional=True)
    with pytest.raises(ValueError, match="only available"):
        SequenceEncoder(2, 16, kind="transformer", bidirectional=True)
    assert cmr(encoder_bidirectional=True)(torch.randn(2, 20, 2)).tasks.tensor.shape == (2, 1)


def test_concept_pos_weight_upweights_rare_positive_steps():
    from torch_concepts.nn import TemporalCMRLearner, concept_pos_weight

    loader = rule_dataset()
    weight = concept_pos_weight(loader)
    true = torch.cat([b["temporal_concepts"] for b in loader])
    rate = true.mean((0, 1))
    assert torch.allclose(weight, ((1 - rate) / rate).clamp(1, 50), atol=1e-4)

    plain = TemporalCMRLearner(cmr())
    weighted = TemporalCMRLearner(cmr(), concept_pos_weight=torch.tensor([9.0, 1.0]))
    predicted = torch.full((1, 4, 2), 0.1)
    target = torch.tensor([[[1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])
    assert weighted.concept_loss(predicted, target) > plain.concept_loss(predicted, target)
    with pytest.raises(ValueError, match="shape"):
        TemporalCMRLearner(cmr(), concept_pos_weight=torch.ones(3))
