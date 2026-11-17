"""MIT-BIH Arrhythmia temporal dataset."""

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from ..base.temporal_dataset import TemporalConceptDatasetWithRules
from ...annotations import Annotations
from ...tensor import AnnotatedTensor


class MITBIHArrhythmiaDataset(TemporalConceptDatasetWithRules):
    concept_labels = [
        "normal_beat",
        "ventricular_beat",
        "tachycardia",
        "bradycardia",
    ]
    normal_symbols = frozenset({"N", "L", "R", "e", "j"})
    ventricular_symbols = frozenset({"V", "E"})

    rulebook = [
        {
            "name": "ventricular_activity",
            "task": "ventricular_arrhythmia",
            "kind": "eventually_and_eventually",
            "first": {"concept": "ventricular_beat", "threshold": 0.5},
            "second": {"concept": "ventricular_beat", "threshold": 0.5},
            "interval": [0, 3599],
        },
        {
            "name": "tachycardia_recovery",
            "task": "rhythm_recovery",
            "kind": "always_implies_eventually",
            "antecedent": {"concept": "tachycardia", "threshold": 0.5},
            "consequent": {"concept": "normal_beat", "threshold": 0.5},
            "interval": [0, 3599],
        },
        {
            "name": "no_bradycardia",
            "task": "rhythm_stability",
            "kind": "always_not",
            "predicate": {"concept": "bradycardia", "threshold": 0.5},
        },
    ]

    def __init__(
        self,
        root: str | Path,
        records: Sequence[str] | None = None,
        window_seconds: float = 10.0,
        stride_seconds: float | None = None,
        channel: int = 0,
        download: bool = True,
    ):
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive.")
        if stride_seconds is None:
            stride_seconds = window_seconds
        if stride_seconds <= 0:
            raise ValueError("stride_seconds must be positive.")
        if channel < 0:
            raise ValueError("channel must be non-negative.")

        try:
            import wfdb
        except ImportError as exc:
            raise ImportError(
                "MITBIHArrhythmiaDataset requires the optional 'wfdb' package."
            ) from exc

        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        records = [str(record) for record in (records or self._default_records())]
        if not records:
            raise ValueError("records must contain at least one record.")

        if download:
            wfdb.dl_database("mitdb", dl_dir=str(root), records=records, keep_subdirs=False)

        signals, concepts, robustness = [], [], []
        sample_rate = None
        for record_name in records:
            record = wfdb.rdrecord(str(root / record_name))
            annotation = wfdb.rdann(str(root / record_name), "atr")
            if record.fs <= 0 or channel >= record.p_signal.shape[1]:
                raise ValueError(f"Invalid channel {channel} for record {record_name}.")
            sample_rate = record.fs
            record_signals, record_concepts = self._extract_windows(
                record.p_signal[:, channel],
                sample_rate,
                annotation.sample,
                annotation.symbol,
                window_seconds,
                stride_seconds,
            )
            signals.extend(record_signals)
            concepts.extend(record_concepts)
            robustness.extend(self._compute_robustness(item) for item in record_concepts)

        if not signals:
            raise ValueError("No complete windows were found in the selected records.")

        self.concept_annotations = Annotations(
            labels=self.concept_labels,
            cardinalities=[1] * len(self.concept_labels),
            types=["binary"] * len(self.concept_labels),
        )
        self.records = records
        self.sample_rate = float(sample_rate)
        self.rulebook = list(self.rulebook)
        self.robustness = np.asarray(robustness, dtype=np.float32)
        super().__init__(
            input_data=np.asarray(signals, dtype=np.float32)[..., None],
            concepts=AnnotatedTensor(
                torch.stack(concepts),
                self.concept_annotations,
                axis=-1,
            ),
            rulebook=self.rulebook,
            precision=32,
            name="MIT-BIH Arrhythmia",
        )

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
    def _extract_windows(
        cls,
        signal: np.ndarray,
        sample_rate: float,
        samples: Iterable[int],
        symbols: Iterable[str],
        window_seconds: float,
        stride_seconds: float,
    ) -> tuple[list[np.ndarray], list[torch.Tensor]]:
        window = int(round(window_seconds * sample_rate))
        stride = int(round(stride_seconds * sample_rate))
        if window <= 0 or stride <= 0:
            raise ValueError("Window and stride must contain at least one sample.")

        sample_positions = np.asarray(list(samples), dtype=np.int64)
        symbol_list = list(symbols)
        windows, concept_windows = [], []

        for start in range(0, len(signal) - window + 1, stride):
            stop = start + window
            values = np.zeros((window, len(cls.concept_labels)), dtype=np.float32)
            for sample, symbol in zip(sample_positions, symbol_list):
                if start <= sample < stop:
                    idx = sample - start
                    if symbol in cls.normal_symbols:
                        values[idx, 0] = 1.0
                    if symbol in cls.ventricular_symbols:
                        values[idx, 1] = 1.0

            beat_positions = np.flatnonzero(values[:, 0] + values[:, 1])
            if beat_positions.size > 1:
                rr = np.diff(beat_positions) / sample_rate
                rate = 60.0 / max(float(np.median(rr)), 1e-6)
                values[:, 2] = float(rate >= 100.0)
                values[:, 3] = float(rate < 60.0)

            windows.append(signal[start:stop].astype(np.float32))
            concept_windows.append(torch.from_numpy(values))

        return windows, concept_windows

    @staticmethod
    def _compute_robustness(concepts: torch.Tensor) -> np.ndarray:
        ventricular = concepts[:, 1] - 0.5
        normal = concepts[:, 0] - 0.5
        tachycardia = concepts[:, 2] - 0.5
        bradycardia = concepts[:, 3] - 0.5
        window = min(3600, ventricular.shape[0])
        future_ventricular = torch.stack(
            [
                ventricular[idx : min(idx + window, ventricular.shape[0])].max()
                for idx in range(window)
            ]
        )
        future_normal = torch.stack(
            [
                normal[idx : min(idx + window, normal.shape[0])].max()
                for idx in range(window)
            ]
        )
        return np.asarray(
            [
                torch.minimum(ventricular[:window], future_ventricular).max().item(),
                torch.maximum(-tachycardia[:window], future_normal).min().item(),
                (-bradycardia).min().item(),
            ],
            dtype=np.float32,
        )

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample["robustness"] = self.robustness[index]
        return sample
