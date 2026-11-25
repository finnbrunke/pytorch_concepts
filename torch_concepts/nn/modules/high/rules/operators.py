"""Smooth min/max used by the temporal rule decoders.

Both are softmax-weighted means, so the result always lies between the hard
min and max (no ``log(n) / beta`` offset as with ``logsumexp``) and every
input receives a non-zero gradient. ``beta=None`` gives the exact hard
operator.
"""

from typing import Optional

import torch


def smooth_max(values: torch.Tensor, dim: int = -1, beta: Optional[float] = None) -> torch.Tensor:
    if beta is None:
        return values.amax(dim=dim)
    weights = torch.softmax(beta * values, dim=dim)
    return (weights * values).sum(dim=dim)


def smooth_min(values: torch.Tensor, dim: int = -1, beta: Optional[float] = None) -> torch.Tensor:
    return -smooth_max(-values, dim=dim, beta=beta)


def smooth_or(left: torch.Tensor, right: torch.Tensor, beta: Optional[float] = None) -> torch.Tensor:
    """Smooth ``max(left, right)``: STL disjunction, and implication as ``max(-a, b)``."""
    return smooth_max(torch.stack([left, right], dim=-1), dim=-1, beta=beta)


def smooth_and(left: torch.Tensor, right: torch.Tensor, beta: Optional[float] = None) -> torch.Tensor:
    return smooth_min(torch.stack([left, right], dim=-1), dim=-1, beta=beta)
