"""Offline tests for the MIT-BIH concept, rule and window construction."""

import sys
import types

import numpy as np
import pytest
import torch

from torch_concepts import AnnotatedTensor
from torch_concepts.data.datasets.mit_bih import (
    MITBIHArrhythmiaDataset as MITBIH,
    evaluate_rule,
)
from torch_concepts.nn import STLRuleDecoder

FS = 100.0
NORMAL, VENTRICULAR, TACHY, BRADY = range(4)


def beats_every(seconds, count, symbol="N", start=0.5):
    samples = [int(round((start + i * seconds) * FS)) for i in range(count)]
    return samples, [symbol] * count


def concepts_for(samples, symbols, n_samples=None):
    n_samples = n_samples or max(samples) + 200
    return MITBIH.record_concepts(n_samples, FS, samples, symbols, beat_width_seconds=0.1)


class TestRecordConcepts:
    def test_beats_become_pulses_of_the_requested_width(self):
        values = concepts_for([100, 300], ["N", "V"])

        assert values[95:106, NORMAL].tolist() == [1.0] * 11
        assert values[94, NORMAL] == 0 and values[106, NORMAL] == 0
        assert values[295:306, VENTRICULAR].sum() == 11
        assert values[:, NORMAL].sum() == 11 and values[:, VENTRICULAR].sum() == 11

    def test_paced_and_non_beat_symbols_are_not_beat_concepts(self):
        values = concepts_for([100, 200, 300], ["/", "+", "~"])

        assert values[:, NORMAL].sum() == 0
        assert values[:, VENTRICULAR].sum() == 0

    def test_fast_rhythm_is_tachycardia_and_slow_rhythm_is_bradycardia(self):
        fast = concepts_for(*beats_every(0.5, 10))  # 120 bpm
        slow = concepts_for(*beats_every(1.25, 10))  # 48 bpm
        normal = concepts_for(*beats_every(0.8, 10))  # 75 bpm
        inside = slice(int(1.0 * FS), int(4.0 * FS))

        assert fast[inside, TACHY].all() and not fast[:, BRADY].any()
        assert slow[inside, BRADY].all() and not slow[:, TACHY].any()
        assert not normal[:, TACHY].any() and not normal[:, BRADY].any()

    def test_rate_is_defined_across_window_boundaries(self):
        """Rate comes from the whole record, so 1 s of a slow rhythm still has it."""
        values = concepts_for(*beats_every(1.25, 10))

        assert values[int(5.0 * FS):int(6.0 * FS), BRADY].all()

    def test_single_premature_beat_does_not_trigger_tachycardia(self):
        samples, symbols = beats_every(0.8, 12)
        samples[6] = samples[5] + int(0.35 * FS)  # one early ventricular beat
        symbols[6] = "V"
        values = concepts_for(samples, symbols)

        assert not values[:, TACHY].any()

    def test_long_gaps_do_not_count_as_bradycardia(self):
        values = concepts_for([100, 700], ["N", "N"])  # 6 s gap, e.g. noise

        assert not values[:, BRADY].any()


class TestWindows:
    def test_downsampling_keeps_short_pulses(self):
        signal = np.arange(12, dtype=np.float32)
        concepts = np.zeros((12, 4), dtype=np.float32)
        concepts[5, NORMAL] = 1.0

        signal_ds, concepts_ds = MITBIH._downsample(signal, concepts, 4)

        assert signal_ds.tolist() == [1.5, 5.5, 9.5]
        assert concepts_ds[:, NORMAL].tolist() == [0.0, 1.0, 0.0]

    def test_extract_windows_shapes_and_alignment(self):
        signal = np.arange(1000, dtype=np.float32)
        concepts = np.tile(signal[:, None], (1, 4))

        signals, windows = MITBIH._extract_windows(signal, concepts, FS, 2.0, 1.0)

        assert signals.shape == (9, 200)
        assert windows.shape == (9, 200, 4)
        assert signals[3, 0] == 300 and windows[3, 0, 2] == 300

    def test_records_shorter_than_one_window_give_no_windows(self):
        signals, windows = MITBIH._extract_windows(
            np.zeros(50, dtype=np.float32), np.zeros((50, 4), dtype=np.float32), FS, 1.0, 1.0
        )

        assert signals.shape == (0, 100) and windows.shape == (0, 100, 4)


def trace(length, normal=(), ventricular=(), tachy=(), brady=()):
    values = np.zeros((length, 4), dtype=np.float32)
    for column, steps in enumerate((normal, ventricular, tachy, brady)):
        values[list(steps), column] = 1.0
    return values


class TestRules:
    rules = {rule["task"]: rule for rule in MITBIH.build_rulebook(FS)}
    labels = MITBIH.concept_labels

    def holds(self, task, values):
        return evaluate_rule(values, self.rules[task], self.labels) >= 0

    def test_rulebook_intervals_are_in_time_steps(self):
        assert self.rules["ventricular_arrhythmia"]["interval"] == [20, 200]
        assert MITBIH.build_rulebook(90.0)[0]["interval"] == [18, 180]

    def test_repeated_ventricular_beats_need_two_distinct_beats(self):
        task = "ventricular_arrhythmia"
        single = trace(400, ventricular=range(100, 111))
        couplet = trace(400, ventricular=[*range(100, 111), *range(160, 171)])
        far_apart = trace(500, ventricular=[*range(50, 61), *range(300, 311)])

        assert not self.holds(task, single)
        assert self.holds(task, couplet)
        assert not self.holds(task, far_apart)

    def test_normal_rhythm_is_broken_by_a_missing_normal_beat(self):
        task = "sustained_normal_rhythm"
        regular = trace(400, normal=[s for b in range(0, 400, 80) for s in range(b, b + 5)])
        pause = trace(400, normal=[*range(0, 5), *range(250, 255)])

        assert self.holds(task, regular)
        assert not self.holds(task, pause)

    def test_bradycardia_anywhere_violates_stability(self):
        assert self.holds("rhythm_stability", trace(300))
        assert not self.holds("rhythm_stability", trace(300, brady=[299]))

    def test_interval_longer_than_the_trace_is_rejected(self):
        with pytest.raises(ValueError, match="does not fit"):
            evaluate_rule(trace(100), self.rules["ventricular_arrhythmia"], self.labels)

    def test_fixed_stl_decoder_matches_the_dataset_labels(self):
        rng = np.random.default_rng(0)
        traces = (rng.random((64, 300, 4)) < 0.03).astype(np.float32)
        dataset_labels = np.array(
            [[evaluate_rule(t, self.rules[task], self.labels) for task in MITBIH.task_names]
             for t in traces]
        )
        annotations = MITBIH_annotations()
        decoder = STLRuleDecoder(
            annotations, MITBIH.task_names, MITBIH.build_rulebook(FS), smoothness=None
        )
        _, robustness, _ = decoder(
            AnnotatedTensor(torch.from_numpy(traces), annotations, axis=-1),
            torch.ones(64, 3, 1),
        )

        assert 0 < (dataset_labels >= 0).mean() < 1
        np.testing.assert_allclose(robustness[..., 0].numpy(), dataset_labels, atol=1e-6)


def MITBIH_annotations():
    from torch_concepts import Annotations

    return Annotations(
        labels=MITBIH.concept_labels,
        cardinalities=[1] * 4,
        types=["binary"] * 4,
    )


@pytest.fixture
def fake_wfdb(monkeypatch):
    """Stand-in for ``wfdb``: every record is 60 s of regular 75 bpm rhythm."""
    loaded = []

    def rdrecord(path):
        loaded.append(path.rsplit("/", 1)[-1])
        signal = np.sin(np.linspace(0, 100, int(60 * 360)))[:, None]
        return types.SimpleNamespace(fs=360, p_signal=np.hstack([signal, signal]))

    def rdann(path, extension):
        samples = np.arange(180, 60 * 360, int(0.8 * 360))
        return types.SimpleNamespace(sample=samples, symbol=["N"] * len(samples))

    module = types.SimpleNamespace(
        rdrecord=rdrecord, rdann=rdann, dl_database=lambda *a, **k: None
    )
    monkeypatch.setitem(sys.modules, "wfdb", module)
    return loaded


class TestDataset:
    def test_builds_windows_labels_and_skips_paced_records(self, fake_wfdb, tmp_path):
        dataset = MITBIH(
            tmp_path, records=["100", "102", "106"], window_seconds=4.0,
            downsample=4, download=False,
        )

        assert dataset.records == ["100", "106"]
        assert fake_wfdb == ["100", "106"]
        assert dataset.sample_rate == 90.0
        assert len(dataset) == 30
        assert dataset.temporal_concepts.shape == (30, 360, 4)
        assert dataset.robustness.shape == (30, 3)
        # Regular 75 bpm: no couplets, unbroken normal rhythm, no bradycardia.
        assert (dataset.robustness >= 0).tolist() == [[False, True, True]] * 30
        sample = dataset[0]
        assert sample["robustness"].shape == (3,)
        assert sample["temporal_concepts"].shape == (360, 4)

    def test_window_too_short_for_the_rulebook_is_rejected(self, fake_wfdb, tmp_path):
        with pytest.raises(ValueError, match="too short for the rulebook"):
            MITBIH(tmp_path, records=["100"], window_seconds=1.0, download=False)

    def test_only_paced_records_is_rejected(self, fake_wfdb, tmp_path):
        with pytest.raises(ValueError, match="non-paced"):
            MITBIH(tmp_path, records=["102", "104"], download=False)
