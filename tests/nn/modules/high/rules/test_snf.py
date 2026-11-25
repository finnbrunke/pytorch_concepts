import math

import pytest
import torch

from torch_concepts import Annotations, AnnotatedTensor
from torch_concepts.nn import ConceptMemoryReasonerTS, SNFConfig, SNFRuleDecoder


def annotations():
    return Annotations(labels=["a", "b"], cardinalities=[1, 1])


def concepts(values):
    return AnnotatedTensor(
        torch.tensor(values, dtype=torch.float32),
        annotations().to_concept_space(),
        axis=-1,
    )


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


@torch.no_grad()
def set_rule(decoder, antecedent, consequent, start, end, rule=0, task=0):
    """Pin one rule to a crisp ``G(antecedent -> F_[start,end] consequent)``."""
    labels = decoder.annotations.labels
    horizon = decoder.config.max_horizon
    min_width = decoder.config.min_interval
    decoder.antecedent_logits[task, rule] = -20.0
    decoder.antecedent_logits[task, rule, labels.index(antecedent)] = 20.0
    decoder.consequent_logits[task, rule] = -20.0
    decoder.consequent_logits[task, rule, labels.index(consequent)] = 20.0
    decoder.antecedent_threshold_logits[task, rule] = 0.0
    decoder.consequent_threshold_logits[task, rule] = 0.0
    decoder.start_logits[task, rule] = _logit(start / (horizon - min_width))
    decoder.width_logits[task, rule] = _logit(
        (end - start - min_width) / (horizon - start - min_width)
    )


def crisp_decoder(max_horizon, n_rules=1):
    config = SNFConfig(
        max_horizon=max_horizon,
        smoothness=50.0,
        time_temperature=0.05,
        learn_robustness_scale=False,
    )
    return SNFRuleDecoder(annotations(), ["task"], n_rules=n_rules, config=config)


def trace_with(length, a_steps=(), b_steps=(), a_all=False, b_all=False):
    values = torch.zeros(1, length, 2)
    values[0, list(a_steps), 0] = 1.0
    values[0, list(b_steps), 1] = 1.0
    if a_all:
        values[0, :, 0] = 1.0
    if b_all:
        values[0, :, 1] = 1.0
    return AnnotatedTensor(values, annotations().to_concept_space(), axis=-1)


def robustness_of(decoder, trace):
    selection = torch.full((1, 1, decoder.n_rules), 1.0 / decoder.n_rules)
    return decoder(trace, selection)[1][0, 0, 0].item()


def test_snf_decoder_returns_batched_rule_outputs_and_gradients():
    decoder = SNFRuleDecoder(
        annotations(), ["task"], n_rules=2, config=SNFConfig(max_horizon=4)
    )
    trace = concepts([[[1, 0], [0, 1], [0, 0], [0, 1]]]).requires_grad_()
    selection = torch.full((1, 1, 2), 0.5)
    tasks, robustness, satisfaction = decoder(trace, selection)

    assert tasks.tensor.shape == (1, 1)
    assert robustness.shape == (1, 1, 2)
    assert satisfaction.shape == (1, 1, 2)
    (tasks.tensor + satisfaction).sum().backward()
    assert trace.tensor.grad is not None
    assert all(parameter.grad is not None for parameter in decoder.parameters())


def test_snf_export_contains_one_valid_clause_per_rule():
    decoder = SNFRuleDecoder(
        annotations(), ["task"], n_rules=2, config=SNFConfig(max_horizon=8)
    )
    rules = decoder.export_rules()

    assert len(rules) == 2
    assert all(rule["kind"] == "always_implies_eventually" for rule in rules)
    assert all(rule["interval"][0] < rule["interval"][1] <= 8 for rule in rules)


def test_temporal_cmr_can_use_learned_snf_rules():
    model = ConceptMemoryReasonerTS(
        input_size=2,
        annotations=annotations(),
        task_names=["task"],
        n_rules=2,
        latent_size=4,
        snf_config=SNFConfig(max_horizon=4),
    )
    output = model(torch.randn(3, 5, 2))

    assert output.tasks.tensor.shape == (3, 1)
    assert output.rule_robustness.shape == (3, 1, 2)
    assert output.rule_satisfaction.shape == (3, 1, 2)


def test_config_rejects_invalid_time_temperature_and_init_std():
    with pytest.raises(ValueError):
        SNFConfig(time_temperature=0.0)
    with pytest.raises(ValueError):
        SNFConfig(init_std=-1.0)


def test_rules_of_one_task_start_from_different_parameters():
    torch.manual_seed(0)
    decoder = SNFRuleDecoder(
        annotations(), ["task"], n_rules=3, config=SNFConfig(max_horizon=40)
    )
    rules = decoder.export_rules()

    assert len({tuple(rule["interval"]) for rule in rules}) == 3
    for name in ("antecedent_logits", "consequent_logits", "start_logits"):
        parameter = getattr(decoder, name)[0]
        assert not torch.allclose(parameter[0], parameter[1])


@pytest.mark.parametrize("horizon", [40, 360])
def test_interval_parameters_receive_non_negligible_gradients(horizon):
    torch.manual_seed(0)
    decoder = SNFRuleDecoder(
        annotations(), ["task"], n_rules=2, config=SNFConfig(max_horizon=horizon)
    )
    trace = torch.rand(4, horizon, 2)
    tasks, _, _ = decoder(
        AnnotatedTensor(trace, annotations().to_concept_space(), axis=-1),
        torch.full((4, 1, 2), 0.5),
    )
    tasks.tensor.sum().backward()

    assert decoder.start_logits.grad.abs().max() > 1e-3
    assert decoder.width_logits.grad.abs().max() > 1e-3


def test_soft_always_is_unbiased_for_long_constant_traces():
    decoder = SNFRuleDecoder(
        annotations(), ["task"], n_rules=1, config=SNFConfig(max_horizon=360)
    )
    values = torch.full((1, 1, 1, 360), 0.3)
    end = torch.zeros(1, 1)

    # The old -logsumexp/beta soft-min was biased by -log(360)/10 ~ -0.59 here.
    assert decoder._always(values, end).item() == pytest.approx(0.3, abs=1e-2)


def test_soft_always_tracks_the_minimum():
    decoder = SNFRuleDecoder(
        annotations(), ["task"], n_rules=1, config=SNFConfig(max_horizon=360)
    )
    values = torch.full((1, 1, 1, 360), 0.5)
    values[..., 100] = -0.5

    assert decoder._always(values, torch.zeros(1, 1)).item() < -0.4


def test_satisfaction_can_be_confident():
    decoder = SNFRuleDecoder(
        annotations(), ["task"], n_rules=1, config=SNFConfig(max_horizon=10)
    )
    robustness = torch.tensor(0.5)

    assert decoder.log_robustness_scale.requires_grad
    assert torch.sigmoid(decoder.robustness_scale * robustness) > 0.9


def test_eventually_respects_the_interval():
    decoder = crisp_decoder(max_horizon=10)
    trace = trace_with(12, a_steps=[2], b_steps=[6])

    set_rule(decoder, "a", "b", start=3, end=5)
    assert robustness_of(decoder, trace) > 0.4
    set_rule(decoder, "a", "b", start=0, end=2)
    assert robustness_of(decoder, trace) < -0.4


def test_truncated_windows_at_the_end_of_the_trace_are_not_violations():
    decoder = crisp_decoder(max_horizon=10)
    set_rule(decoder, "a", "b", start=5, end=8)
    trace = trace_with(20, a_all=True, b_all=True)

    assert robustness_of(decoder, trace) > 0.4


def test_learned_rule_recovers_a_generating_interval():
    """Trained on ground-truth concepts, the bank should fit a known rule."""
    torch.manual_seed(0)
    length, delay = 30, 6
    traces, labels = [], []
    for _ in range(256):
        trace = torch.zeros(length, 2)
        trigger = torch.randint(0, length - 12, ()).item()
        trace[trigger, 0] = 1.0
        response = trigger + torch.randint(1, 11, ()).item()
        trace[response, 1] = 1.0
        traces.append(trace)
        labels.append(float(response - trigger <= delay))
    traces = AnnotatedTensor(
        torch.stack(traces), annotations().to_concept_space(), axis=-1
    )
    labels = torch.tensor(labels)

    decoder = SNFRuleDecoder(
        annotations(), ["task"], n_rules=1, config=SNFConfig(max_horizon=12)
    )
    with torch.no_grad():
        set_rule(decoder, "a", "b", start=1, end=11)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=0.05)
    selection = torch.ones(len(labels), 1, 1)
    for _ in range(300):
        optimizer.zero_grad()
        tasks, _, _ = decoder(traces, selection)
        loss = torch.nn.functional.binary_cross_entropy(
            tasks.tensor[:, 0].clamp(1e-6, 1 - 1e-6), labels
        )
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        tasks, _, _ = decoder(traces, selection)
    accuracy = ((tasks.tensor[:, 0] >= 0.5).float() == labels).float().mean()
    rule = decoder.export_rules()[0]

    assert accuracy > 0.9
    assert rule["interval"][1] == pytest.approx(delay, abs=1.5)


def test_consequent_gets_gradient_when_the_antecedent_never_fires():
    """With a hard max in the implication this gradient was exactly zero."""
    decoder = crisp_decoder(max_horizon=10)
    set_rule(decoder, "a", "b", start=1, end=5)
    decoder.consequent_threshold_logits.requires_grad_(True)
    trace = trace_with(20, b_steps=[3])
    tasks, _, _ = decoder(trace, torch.ones(1, 1, 1))
    tasks.tensor.sum().backward()

    assert decoder.consequent_threshold_logits.grad.abs().sum() > 0
    assert decoder.consequent_logits.grad.abs().sum() > 0


def test_intervals_stay_valid_for_extreme_logits():
    decoder = SNFRuleDecoder(
        annotations(), ["task"], n_rules=4, config=SNFConfig(max_horizon=20)
    )
    with torch.no_grad():
        decoder.start_logits.copy_(torch.tensor([[-30.0, 30.0, 30.0, 0.0]]))
        decoder.width_logits.copy_(torch.tensor([[30.0, 30.0, -30.0, 0.0]]))
    start, end = decoder._intervals()

    assert (start >= 0).all() and (end <= 20).all()
    assert (end - start >= decoder.config.min_interval - 1e-5).all()
