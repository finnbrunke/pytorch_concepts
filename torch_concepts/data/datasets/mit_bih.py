"""MIT-BIH Arrhythmia temporal dataset."""

import warnings
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view

from ..base.temporal_dataset import TemporalConceptDatasetWithRules
from ...annotations import Annotations
from ...tensor import AnnotatedTensor


def _bounded_eventually(values: np.ndarray, start: int, end: int) -> np.ndarray:
    """``max(values[t + start : t + end + 1])`` for every ``t <= T - 1 - end``.

    Only time steps whose whole window lies inside the trace are returned, so
    the result has length ``T - end``.
    """
    if not 0 <= start <= end:
        raise ValueError(f"Invalid interval [{start}, {end}].")
    if end >= values.shape[-1]:
        raise ValueError(
            f"Interval end {end} does not fit a trace of length {values.shape[-1]}."
        )
    return sliding_window_view(values, end + 1, axis=-1)[..., start:].max(axis=-1)


def evaluate_rule(concepts: np.ndarray, rule: dict, labels: Sequence[str]) -> float:
    """Hard STL robustness of one rulebook entry on a ``(time, concepts)`` trace.

    Supported kinds, with atoms ``x = concept - threshold``:

    * ``always_implies_eventually``: ``G(antecedent -> F_[a,b] consequent)``
    * ``eventually_and_eventually``: ``F(first and F_[a,b] second)``
    * ``always_not``: ``G(not predicate)``

    ``G`` and the outer ``F`` range over the time steps whose bounded window
    fits in the trace, matching the learned SNF decoder.
    """
    indices = {name: index for index, name in enumerate(labels)}

    def atom(spec):
        return concepts[:, indices[spec["concept"]]] - float(spec.get("threshold", 0.0))

    kind = rule["kind"]
    if kind == "always_not":
        return float((-atom(rule["predicate"])).min())
    start, end = (int(value) for value in rule["interval"])
    if kind == "always_implies_eventually":
        future = _bounded_eventually(atom(rule["consequent"]), start, end)
        antecedent = atom(rule["antecedent"])[: future.shape[-1]]
        return float(np.maximum(-antecedent, future).min())
    if kind == "eventually_and_eventually":
        future = _bounded_eventually(atom(rule["second"]), start, end)
        first = atom(rule["first"])[: future.shape[-1]]
        return float(np.minimum(first, future).max())
    raise ValueError(f"Unknown rule kind {kind!r}.")


class MITBIHArrhythmiaDataset(TemporalConceptDatasetWithRules):
    """Windows of MIT-BIH ECG with per-time-step beat and rhythm concepts.

    Concepts:

    * ``normal_beat`` / ``ventricular_beat``: pulses of ``beat_width_seconds``
      centred on annotated normal (N, L, R, e, j) or ventricular (V, E) beats.
    * ``tachycardia`` / ``bradycardia``: heart rate >= 100 or < 60 bpm, from a
      running median of RR intervals over the whole record. The median
      prevents single premature beats from triggering a rate concept, and
      computing it per record makes it meaningful for short windows.

    Paced records (102, 104, 107, 217) are skipped by default, following the
    AAMI convention; their paced beats fit neither beat concept.
    """

    concept_labels = [
        "normal_beat",
        "ventricular_beat",
        "tachycardia",
        "bradycardia",
    ]
    task_names = [
        "ventricular_arrhythmia",
        "sustained_normal_rhythm",
        "rhythm_stability",
    ]
    normal_symbols = frozenset({"N", "L", "R", "e", "j"})
    ventricular_symbols = frozenset({"V", "E"})
    beat_symbols = frozenset(
        {"N", "L", "R", "B", "A", "a", "J", "S", "V", "r", "F", "e", "j",
         "n", "E", "/", "f", "Q", "?"}
    )
    paced_records = frozenset({"102", "104", "107", "217"})
    tachycardia_bpm = 100.0
    bradycardia_bpm = 60.0
    rr_median_beats = 5
    max_rr_seconds = 3.0
    rulebook = None  # built per instance from the sample rate

    def __init__(
        self,
        root: str | Path,
        records: Sequence[str] | None = None,
        window_seconds: float = 4.0,
        stride_seconds: float | None = None,
        channel: int = 0,
        download: bool = True,
        downsample: int = 1,
        beat_width_seconds: float = 0.1,
        exclude_paced: bool = True,
        normalize: bool = True,
    ):
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive.")
        if stride_seconds is None:
            stride_seconds = window_seconds
        if stride_seconds <= 0:
            raise ValueError("stride_seconds must be positive.")
        if channel < 0:
            raise ValueError("channel must be non-negative.")
        if downsample < 1:
            raise ValueError("downsample must be a positive integer.")
        if beat_width_seconds <= 0:
            raise ValueError("beat_width_seconds must be positive.")

        try:
            import wfdb
        except ImportError as exc:
            raise ImportError(
                "MITBIHArrhythmiaDataset requires the optional 'wfdb' package."
            ) from exc

        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        records = [str(record) for record in (records or self._default_records())]
        if exclude_paced:
            records = [record for record in records if record not in self.paced_records]
        if not records:
            raise ValueError("records must contain at least one non-paced record.")

        if download:
            wfdb.dl_database("mitdb", dl_dir=str(root), records=records, keep_subdirs=False)

        signals, concepts, robustness, window_records = [], [], [], []
        sample_rate = None
        for record_name in records:
            record = wfdb.rdrecord(str(root / record_name))
            annotation = wfdb.rdann(str(root / record_name), "atr")
            if record.fs <= 0 or channel >= record.p_signal.shape[1]:
                raise ValueError(f"Invalid channel {channel} for record {record_name}.")
            signal = record.p_signal[:, channel].astype(np.float32)
            if normalize:
                signal = (signal - np.median(signal)) / (signal.std() + 1e-6)
            record_concepts = self.record_concepts(
                len(signal),
                record.fs,
                annotation.sample,
                annotation.symbol,
                beat_width_seconds,
            )
            signal, record_concepts = self._downsample(signal, record_concepts, downsample)
            sample_rate = record.fs / downsample
            if self.rulebook is None:
                self.rulebook = self.build_rulebook(sample_rate)
                window_steps = int(round(window_seconds * sample_rate))
                longest = max(rule.get("interval", [0, 0])[1] for rule in self.rulebook)
                if longest >= window_steps:
                    raise ValueError(
                        f"window_seconds={window_seconds} is too short for the "
                        f"rulebook; intervals reach {longest / sample_rate:.2f} s."
                    )
            record_signals, record_windows = self._extract_windows(
                signal, record_concepts, sample_rate, window_seconds, stride_seconds
            )
            signals.append(record_signals)
            concepts.append(record_windows)
            robustness.extend(self._compute_robustness(item) for item in record_windows)
            window_records.extend([record_name] * len(record_windows))

        if not window_records:
            raise ValueError("No complete windows were found in the selected records.")

        self.concept_annotations = Annotations(
            labels=self.concept_labels,
            cardinalities=[1] * len(self.concept_labels),
            types=["binary"] * len(self.concept_labels),
        )
        self.records = records
        self.window_records = window_records
        self.sample_rate = float(sample_rate)
        self.robustness = np.asarray(robustness, dtype=np.float32)
        super().__init__(
            input_data=np.concatenate(signals)[..., None],
            concepts=AnnotatedTensor(
                torch.from_numpy(np.concatenate(concepts)),
                self.concept_annotations,
                axis=-1,
            ),
            rulebook=self.rulebook,
            precision=32,
            name="MIT-BIH Arrhythmia",
        )

    @classmethod
    def build_rulebook(cls, sample_rate: float) -> list[dict]:
        """Rulebook with intervals converted from seconds to time steps."""

        def steps(seconds: float) -> int:
            return int(round(seconds * sample_rate))

        return [
            {
                # Repeated ventricular ectopy: two ventricular beats 0.2-2.0 s
                # apart (couplets, bigeminy, runs). The 0.2 s lower bound skips
                # the pulse of the first beat.
                "name": "repeated_ventricular_beats",
                "task": "ventricular_arrhythmia",
                "kind": "eventually_and_eventually",
                "first": {"concept": "ventricular_beat", "threshold": 0.5},
                "second": {"concept": "ventricular_beat", "threshold": 0.5},
                "interval": [steps(0.2), steps(2.0)],
            },
            {
                # Every normal beat is followed by another normal beat within
                # 0.3-1.5 s; violated by ectopic beats or pauses. (A rule with
                # a rare antecedent, e.g. tachycardia, is vacuously true in
                # >99% of windows.)
                "name": "normal_beat_continuity",
                "task": "sustained_normal_rhythm",
                "kind": "always_implies_eventually",
                "antecedent": {"concept": "normal_beat", "threshold": 0.5},
                "consequent": {"concept": "normal_beat", "threshold": 0.5},
                "interval": [steps(0.3), steps(1.5)],
            },
            {
                "name": "no_bradycardia",
                "task": "rhythm_stability",
                "kind": "always_not",
                "predicate": {"concept": "bradycardia", "threshold": 0.5},
            },
        ]

    @staticmethod
    def _default_records() -> list[str]:
        return [
            "100", "101", "102", "103", "104", "105", "106", "107",
            "108", "109", "111", "112", "113", "114", "115", "116",
            "117", "118", "119", "121", "122", "123", "124", "200",
            "201", "202", "203", "205", "207", "208", "209", "210",
            "212", "213", "214", "215", "217", "219", "220", "221",
            "222", "223", "228", "230", "231", "232", "233", "234",
        ]

    @classmethod
    def record_concepts(
        cls,
        n_samples: int,
        sample_rate: float,
        samples: Iterable[int],
        symbols: Iterable[str],
        beat_width_seconds: float = 0.1,
    ) -> np.ndarray:
        """Per-sample concept trace ``(n_samples, 4)`` for a whole record."""
        samples = np.asarray(list(samples), dtype=np.int64)
        symbols = np.asarray(list(symbols), dtype=object)
        values = np.zeros((n_samples, len(cls.concept_labels)), dtype=np.float32)

        half_width = max(0, int(round(beat_width_seconds * sample_rate / 2)))
        for column, group in ((0, cls.normal_symbols), (1, cls.ventricular_symbols)):
            positions = samples[np.isin(symbols, list(group))]
            marks = np.zeros(n_samples + 1, dtype=np.int64)
            np.add.at(marks, np.clip(positions - half_width, 0, n_samples), 1)
            np.add.at(marks, np.clip(positions + half_width + 1, 0, n_samples), -1)
            values[:, column] = (np.cumsum(marks[:-1]) > 0).astype(np.float32)

        beats = np.sort(samples[np.isin(symbols, list(cls.beat_symbols))])
        if beats.size > 1:
            rr = np.diff(beats).astype(np.float64) / sample_rate
            rr[rr > cls.max_rr_seconds] = np.nan
            k = cls.rr_median_beats
            padded = np.pad(rr, (k // 2, k - 1 - k // 2), constant_values=np.nan)
            with warnings.catch_warnings():
                # All-NaN windows (gaps) are expected and stay NaN.
                warnings.simplefilter("ignore", RuntimeWarning)
                median_rr = np.nanmedian(sliding_window_view(padded, k), axis=-1)
            median_rr[np.isnan(rr)] = np.nan
            rate = 60.0 / median_rr
            # Interval i spans [beats[i], beats[i + 1]).
            interval = np.searchsorted(beats, np.arange(n_samples), side="right") - 1
            inside = (interval >= 0) & (interval < rr.size)
            sample_rate_bpm = np.full(n_samples, np.nan)
            sample_rate_bpm[inside] = rate[interval[inside]]
            with np.errstate(invalid="ignore"):
                values[:, 2] = (sample_rate_bpm >= cls.tachycardia_bpm).astype(np.float32)
                values[:, 3] = (sample_rate_bpm < cls.bradycardia_bpm).astype(np.float32)
        return values

    @staticmethod
    def _downsample(
        signal: np.ndarray, concepts: np.ndarray, factor: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mean-pool the signal and max-pool concepts so short pulses survive."""
        if factor == 1:
            return signal, concepts
        length = (len(signal) // factor) * factor
        signal = signal[:length].reshape(-1, factor).mean(axis=1)
        concepts = concepts[:length].reshape(-1, factor, concepts.shape[-1]).max(axis=1)
        return signal.astype(np.float32), concepts

    @staticmethod
    def _extract_windows(
        signal: np.ndarray,
        concepts: np.ndarray,
        sample_rate: float,
        window_seconds: float,
        stride_seconds: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        window = int(round(window_seconds * sample_rate))
        stride = int(round(stride_seconds * sample_rate))
        if window <= 0 or stride <= 0:
            raise ValueError("Window and stride must contain at least one sample.")
        if len(signal) < window:
            return (
                np.empty((0, window), dtype=np.float32),
                np.empty((0, window, concepts.shape[-1]), dtype=np.float32),
            )
        signal_windows = sliding_window_view(signal, window)[::stride]
        concept_windows = sliding_window_view(concepts, window, axis=0)[::stride]
        return (
            np.ascontiguousarray(signal_windows, dtype=np.float32),
            np.ascontiguousarray(concept_windows.transpose(0, 2, 1), dtype=np.float32),
        )

    def _compute_robustness(self, concepts) -> np.ndarray:
        concepts = np.asarray(concepts)
        by_task = {rule["task"]: rule for rule in self.rulebook}
        return np.asarray(
            [
                evaluate_rule(concepts, by_task[task], self.concept_labels)
                for task in self.task_names
            ],
            dtype=np.float32,
        )

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample["robustness"] = self.robustness[index]
        return sample
