import pytest
import torch

from torch_concepts import Annotations
from torch_concepts.nn import (
    ConceptMemoryReasonerTS,
    SNFConfig,
    UncertaintyInterventionPolicy,
    concept_intervention_curve,
    intervene_on_concepts,
    single_concept_effects,
    time_intervention_curve,
)
from torch_concepts.nn.modules.high.models.cmrts_interventions import concept_mask, time_mask

from test_cmrts_evaluation import RULE, cmr, rule_dataset


def model_and_batch(types=("binary", "binary")):
    annotations = Annotations(labels=["a", "b"], cardinalities=[1, 1], types=list(types))
    model = ConceptMemoryReasonerTS(
        2, annotations, ["task"], n_rules=1, latent_size=8,
        snf_config=SNFConfig(max_horizon=5),
    ).eval()
    torch.manual_seed(0)
    x = torch.randn(3, 12, 2)
    true = (torch.rand(3, 12, 2) > 0.5).float()
    return model, x, true


def test_whole_trace_intervention_replaces_only_the_chosen_concept():
    model, x, true = model_and_batch()
    with torch.no_grad():
        base = model(x)
        with intervene_on_concepts(model, true, ["b"]):
            out = model(x)

    assert torch.allclose(out.concepts.tensor[..., 1], true[..., 1], atol=1e-3)
    assert torch.equal(out.concepts.tensor[..., 0], base.concepts.tensor[..., 0])
    assert not torch.allclose(out.tasks.tensor, base.tasks.tensor)
    assert torch.equal(out.rule_selection, base.rule_selection)


def test_continuous_concepts_are_set_to_the_raw_value():
    model, x, _ = model_and_batch(types=("continuous", "binary"))
    value = torch.full((3, 12, 2), 0.37)
    with torch.no_grad(), intervene_on_concepts(model, value, ["a"]):
        out = model(x)

    assert torch.allclose(out.concepts.tensor[..., 0], value[..., 0])


def test_time_mask_intervention_replaces_only_masked_steps():
    model, x, true = model_and_batch()
    mask = torch.zeros(3, 12, dtype=torch.bool)
    mask[:, :4] = True
    with torch.no_grad():
        base = model(x)
        with intervene_on_concepts(model, true, mask=mask):
            out = model(x)

    assert torch.allclose(out.concepts.tensor[:, :4], true[:, :4], atol=1e-3)
    assert torch.equal(out.concepts.tensor[:, 4:], base.concepts.tensor[:, 4:])
    with pytest.raises(ValueError, match="leave policy and quantile unset"):
        with intervene_on_concepts(model, true, mask=mask, quantile=0.5):
            pass


def test_library_policies_choose_concepts_per_time_step():
    model, x, true = model_and_batch()
    with torch.no_grad():
        base = model(x)
        policy = UncertaintyInterventionPolicy(max_uncertainty_point=0.0)
        with intervene_on_concepts(model, true, policy=policy, quantile=0.5):
            out = model(x)
    changed = (out.concepts.tensor - base.concepts.tensor).abs() > 1e-6

    assert (changed.sum(-1) <= 1).all() and changed.any()


def test_time_mask_selects_the_requested_fraction_most_uncertain_first():
    scores = torch.rand(2, 10, 3)
    mask = time_mask(scores, 0.3, order="uncertainty")

    assert (mask.sum(1) == 3).all()
    # The selected steps are exactly the three most uncertain ones.
    assert (scores.topk(3, dim=1).values.amin(1) == scores.masked_fill(~mask, 2).amin(1)).all()
    assert (time_mask(scores, 0.0) == False).all() and time_mask(scores, 1.0).all()
    with pytest.raises(ValueError):
        time_mask(scores, 1.5)


def test_concept_mask_selects_whole_traces():
    scores = torch.rand(4, 6, 3)
    mask = concept_mask(scores, 2)

    assert (mask.all(1).sum(-1) == 2).all()  # two concepts, all time steps
    assert torch.equal(mask, mask[:, :1].expand_as(mask))


def test_intervention_curves_reach_the_rule_ceiling_with_all_ground_truth():
    batches = rule_dataset()
    exact = cmr(snf_config=None, rule_specs=[RULE], stl_smoothness=None)
    concepts = concept_intervention_curve(exact, batches, n_repeats=2)
    times = time_intervention_curve(exact, batches, fractions=(0.0, 0.5, 1.0), order="uncertainty")

    assert concepts["x"] == [0, 1, 2] and len(concepts["score"]) == 3
    assert concepts["score"][-1] == pytest.approx(1.0)
    assert times["score"][-1] == pytest.approx(1.0)
    assert concepts["gain"] == pytest.approx(concepts["score"][-1] - concepts["score"][0])
    assert concepts["auc"] == pytest.approx(sum(concepts["score"]) / 3)
    assert set(concepts["per_task"]) == {"task"}


def test_single_concept_effects_name_every_concept():
    batches = rule_dataset()
    exact = cmr(snf_config=None, rule_specs=[RULE], stl_smoothness=None)
    effects = single_concept_effects(exact, batches)

    assert set(effects) == {"a", "b"}
    assert all(isinstance(value, float) for value in effects.values())
