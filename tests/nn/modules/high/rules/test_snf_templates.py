import numpy as np
import pytest
import torch

from torch_concepts import AnnotatedTensor, Annotations
from torch_concepts.data.datasets.mit_bih import evaluate_rule
from torch_concepts.nn import SNFConfig, SNFRuleDecoder, STLRuleDecoder, format_rule
from torch_concepts.nn.modules.high.rules.snf import TEMPLATES

ANN = Annotations(labels=["a", "b"], cardinalities=[1, 1], types=["binary"] * 2)
ALL = SNFConfig(max_horizon=10, smoothness=50.0, time_temperature=0.05,
                learn_robustness_scale=False, robustness_scale=20.0, templates=TEMPLATES)


def spec(kind, start=2, end=5):
    a, b = {"concept": "a", "threshold": 0.5}, {"concept": "b", "threshold": 0.5}
    return SNFRuleDecoder._spec("r", "task", kind, a, b, [start, end])


def crisp(kind):
    from torch_concepts.nn import ConceptMemoryReasonerTS, edit_rule

    model = ConceptMemoryReasonerTS(2, ANN, ["task"], n_rules=1, latent_size=8, snf_config=ALL)
    return model, edit_rule(model, "task", 0, spec(kind))


def test_config_validates_templates_and_discrete_mode():
    with pytest.raises(ValueError, match="templates"):
        SNFConfig(templates=("always_implies_eventually", "nonsense"))
    with pytest.raises(ValueError, match="templates"):
        SNFConfig(templates=())
    with pytest.raises(ValueError, match="discrete"):
        SNFConfig(discrete="gumbel")
    assert SNFConfig(templates=["always_not"]).templates == ("always_not",)


@pytest.mark.parametrize("kind", TEMPLATES)
def test_each_template_matches_exact_stl_on_crisp_traces(kind):
    rng = np.random.default_rng(1)
    traces = torch.from_numpy((rng.random((64, 20, 2)) < 0.1).astype(np.float32))
    model, edit = crisp(kind)
    with edit, torch.no_grad():
        _, robustness, _ = model.rule_decoder(
            AnnotatedTensor(traces, ANN.to_concept_space(), axis=-1), torch.ones(64, 1, 1)
        )
    hard = torch.tensor([evaluate_rule(t.numpy(), spec(kind), ["a", "b"]) for t in traces])

    assert 0 < (hard >= 0).float().mean() < 1
    assert torch.equal(robustness[:, 0, 0] >= 0, hard >= 0)


def test_straight_through_uses_one_hot_samples_with_soft_gradients():
    config = SNFConfig(max_horizon=5, discrete="straight_through", templates=TEMPLATES)
    decoder = SNFRuleDecoder(ANN, ["task"], n_rules=3, config=config)
    trace = AnnotatedTensor(torch.rand(4, 12, 2), ANN.to_concept_space(), axis=-1)

    weights = decoder.choice(decoder.antecedent_logits)
    assert set(weights.flatten().tolist()) <= {0.0, 1.0} and torch.equal(weights.sum(-1), torch.ones(1, 3))
    tasks, _, _ = decoder(trace, torch.full((4, 1, 3), 1 / 3))
    tasks.tensor.sum().backward()
    for name in ("antecedent_logits", "consequent_logits", "template_logits"):
        assert getattr(decoder, name).grad.abs().sum() > 0
    decoder.eval()
    expected = torch.nn.functional.one_hot(decoder.antecedent_logits.argmax(-1), 2).float()
    assert torch.equal(decoder.choice(decoder.antecedent_logits), expected)


def test_discrete_specs_and_export_carry_the_template():
    decoder = SNFRuleDecoder(ANN, ["task"], n_rules=3, config=SNFConfig(max_horizon=8, templates=TEMPLATES))
    with torch.no_grad():
        decoder.template_logits.copy_(torch.eye(3)[None] * 10)
    kinds = [rule["kind"] for rule in decoder.discrete_specs()]

    assert kinds == list(TEMPLATES)
    assert [rule["kind"] for rule in decoder.export_rules()] == kinds
    specs = decoder.discrete_specs()
    assert specs[1]["first"] == specs[1]["antecedent"] and specs[2]["predicate"] == specs[2]["antecedent"]
    STLRuleDecoder(ANN, ["task"], specs, smoothness=None)  # readable by the exact decoder


def test_format_rule_prints_every_template():
    assert format_rule(spec("always_implies_eventually")) == "G(a > 0.50 -> F[2,5] b > 0.50)"
    assert format_rule(spec("eventually_and_eventually")) == "F(a > 0.50 and F[2,5] b > 0.50)"
    assert format_rule(spec("always_not")) == "G(not a > 0.50)"


def test_edit_rule_rejects_kinds_outside_the_bank():
    from torch_concepts.nn import ConceptMemoryReasonerTS, edit_rule

    model = ConceptMemoryReasonerTS(2, ANN, ["task"], n_rules=1, latent_size=8,
                                    snf_config=SNFConfig(max_horizon=10))
    with pytest.raises(ValueError, match="only represents"):
        with edit_rule(model, "task", 0, spec("eventually_and_eventually")):
            pass


def test_template_bank_learns_an_eventually_and_eventually_rule():
    """Labels from F(a and F[2,5] b), which the old single template cannot express."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    target = spec("eventually_and_eventually")
    traces = (rng.random((384, 24, 2)) < 0.08).astype(np.float32)
    labels = torch.tensor([float(evaluate_rule(t, target, ["a", "b"]) >= 0) for t in traces])
    traces = AnnotatedTensor(torch.from_numpy(traces), ANN.to_concept_space(), axis=-1)
    config = SNFConfig(max_horizon=10, templates=TEMPLATES, discrete="straight_through")
    decoder = SNFRuleDecoder(ANN, ["task"], n_rules=1, config=config)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=0.05)
    selection = torch.ones(len(labels), 1, 1)
    for _ in range(300):
        optimizer.zero_grad()
        tasks, _, _ = decoder(traces, selection)
        torch.nn.functional.binary_cross_entropy(
            tasks.tensor[:, 0].clamp(1e-6, 1 - 1e-6), labels
        ).backward()
        optimizer.step()
    decoder.eval()
    with torch.no_grad():
        tasks, _, _ = decoder(traces, selection)
    accuracy = ((tasks.tensor[:, 0] >= 0.5).float() == labels).float().mean()
    rule = decoder.export_rules()[0]

    assert 0.2 < labels.mean() < 0.8
    assert accuracy > 0.9, (accuracy, format_rule(rule))
    assert rule["kind"] == "eventually_and_eventually", format_rule(rule)
