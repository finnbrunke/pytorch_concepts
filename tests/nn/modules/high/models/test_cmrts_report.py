import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from torch_concepts.nn import (
    SequenceBlackBox,
    TemporalCMRLearner,
    TemporalInterpretabilityReport,
    flatten_report,
    interpretability_report,
)

from test_cmrts_evaluation import RULE, cmr, rule_dataset
from test_cmrts_learner import ImageLogger


def test_report_has_every_section_for_a_learned_rule_bank():
    batches = rule_dataset()
    model = cmr(n_rules=2).eval()
    report = interpretability_report(
        model, batches, blackbox=SequenceBlackBox(2, ["task"], latent_size=8),
        reference_rules=[RULE], n_repeats=1, time_fractions=(0.0, 1.0),
    )

    assert set(report) == {
        "tasks", "concepts", "completeness", "rule_language_completeness",
        "interventions", "rules",
    }
    assert set(report["interventions"]["curves"]) == {
        "concepts_random", "concepts_uncertainty", "time_random", "time_uncertainty"
    }
    assert set(report["rules"]) == {
        "interventions", "fidelity_predicted", "fidelity_ground_truth", "compactness"
    }


def test_report_skips_rule_sections_for_fixed_rules():
    model = cmr(snf_config=None, rule_specs=[RULE], stl_smoothness=None)
    report = interpretability_report(model, rule_dataset(), n_repeats=1, time_fractions=(0.0, 1.0))

    assert "rules" not in report and "completeness" not in report


def test_flatten_report_keeps_scalars_only():
    report = {"a": {"x": [1, 2], "auc": 0.7, "nan": float("nan"), "b": {"c": 3}}, "flag": True}
    assert flatten_report(report) == {"interp/a/auc": 0.7, "interp/a/b/c": 3.0}


def test_callback_logs_the_report_and_the_curve_figure(tmp_path):
    loader = DataLoader(rule_dataset(), batch_size=None)
    learner = TemporalCMRLearner(cmr(n_rules=2))
    logger = ImageLogger()
    callback = TemporalInterpretabilityReport(
        rule_modes=("soft", "argmax"), n_repeats=1, time_fractions=(0.0, 1.0)
    )
    trainer = Trainer(max_epochs=1, logger=logger, callbacks=[callback],
                      enable_checkpointing=False, enable_progress_bar=False,
                      enable_model_summary=False, default_root_dir=tmp_path)
    trainer.fit(learner, loader)
    trainer.test(learner, loader, verbose=False)

    assert set(callback.report) == {"soft", "argmax"}
    assert "interp/soft/interventions/curves/concepts_random/auc" in logger.metrics
    assert "interp/soft/rules/fidelity_predicted/macro/top_rule_agreement" in logger.metrics
    assert "interp/argmax/rule_language_completeness/task/completeness" in logger.metrics
    assert [key for key, _, _ in logger.images] == [
        "plots/intervention_curves_soft", "plots/intervention_curves_argmax"
    ]


def test_discrete_report_modes():
    batches = rule_dataset()
    model = cmr(n_rules=2).eval()
    options = dict(n_repeats=1, time_fractions=(0.0, 1.0))
    argmax = interpretability_report(model, batches, rule_mode="argmax", **options)
    sample = interpretability_report(model, batches, rule_mode="sample", n_samples=3, **options)

    assert len(argmax["rulebook"]) == 2 and "fidelity_predicted" not in argmax["rules"]
    assert len(sample["rulebooks"]) == 3
    assert "tasks/task/score" in sample["sample_std"]
    assert all(value >= 0 for value in sample["sample_std"].values())
    assert not any(key.startswith("interp/rulebook") for key in flatten_report(argmax))
    try:
        interpretability_report(cmr(snf_config=None, rule_specs=[RULE]), batches, rule_mode="argmax")
    except TypeError:
        pass
    else:
        raise AssertionError("discrete modes need an SNF bank")


def test_average_reports_is_elementwise():
    from torch_concepts.nn.modules.high.models.cmrts_report import average_reports

    reports = [{"a": 1.0, "c": [1.0, 3.0], "x": [0, 1], "name": "r"},
               {"a": 3.0, "c": [3.0, 5.0], "x": [0, 1], "name": "r"}]
    assert average_reports(reports) == {"a": 2.0, "c": [2.0, 4.0], "x": [0.0, 1.0], "name": "r"}
