"""
This module implements the temporal concept dataset class.
"""

from abc import abstractmethod

from torch.utils.data import default_collate

from .dataset import ConceptDataset
from ...tensor import AnnotatedTensor


class TemporalConceptDataset(ConceptDataset):
    ## TODO extend
    def __init__(
        self,
        input_data,
        concepts,
        annotations=None,
        precision=32,
        name=None,
        temporal_reduction="mean",
    ):
        self.temporal_concepts = None
        self.temporal_concept_names = None
        if isinstance(concepts, AnnotatedTensor) and concepts.dim() > 2:
            if concepts.axis != -1:
                raise ValueError(
                    "Temporal concept tensors must annotate their last axis."
                )
            if temporal_reduction != "mean":
                raise ValueError(
                    "Only mean temporal concept reduction is currently supported."
                )
            self.temporal_concepts = concepts
            self.temporal_concept_names = list(concepts.annotations.labels)
            concepts = AnnotatedTensor(
                concepts.tensor.mean(dim=1),
                concepts.annotations,
                axis=1,
            )

        super().__init__(
            input_data=input_data,
            concepts=concepts,
            annotations=annotations,
            precision=precision,
            name=name
        )

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        if self.temporal_concepts is not None:
            sample["temporal_concepts"] = self.temporal_concepts[idx]
            sample["concept_names"] = self.temporal_concept_names
        return sample

    def collate(self, samples):
        batch = super().collate(samples)
        if self.temporal_concepts is not None:
            temporal = default_collate(
                [
                    (
                        sample["temporal_concepts"].tensor
                        if isinstance(sample["temporal_concepts"], AnnotatedTensor)
                        else sample["temporal_concepts"]
                    )
                    for sample in samples
                ]
            )
            batch["temporal_concepts"] = AnnotatedTensor(
                temporal,
                self.temporal_concepts.annotations,
                axis=-1,
            )
            batch["concept_names"] = self.temporal_concept_names
        if "robustness" in samples[0]:
            batch["robustness"] = default_collate(
                [sample["robustness"] for sample in samples]
            )
        return batch

class TemporalConceptDatasetWithRules(TemporalConceptDataset):
    ##TODO extend
    """
    Minimal abstract base class for time-series concept datasets 
    integrated with Signal Temporal Logic (STL) rulebooks.
    """
    def __init__(
        self,
        input_data,
        concepts,
        rulebook=None,
        annotations=None,
        precision=32,
        name=None,
        **kwargs
    ):
        self.rulebook = rulebook
        
        super().__init__(
            input_data=input_data,
            concepts=concepts,
            annotations=annotations,
            precision=precision,
            name=name,
            **kwargs
        )

    @abstractmethod
    def _compute_robustness(self, concepts):
        """
        Subclasses must implement this method to evaluate the dataset's 
        rulebook against a temporal concept sequence.
        """
        pass