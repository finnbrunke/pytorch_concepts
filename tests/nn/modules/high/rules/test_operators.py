import pytest
import torch

from torch_concepts.nn.modules.high.rules.operators import (
    smooth_and,
    smooth_max,
    smooth_min,
    smooth_or,
)


def test_hard_operators_without_beta():
    values = torch.tensor([[0.2, -0.5, 0.9]])

    assert smooth_max(values).item() == pytest.approx(0.9)
    assert smooth_min(values).item() == pytest.approx(-0.5)


@pytest.mark.parametrize("beta", [1.0, 10.0, 100.0])
def test_smooth_operators_lie_between_min_and_max(beta):
    values = torch.randn(32, 50)

    for result in (smooth_max(values, beta=beta), smooth_min(values, beta=beta)):
        assert (result <= values.amax(-1) + 1e-6).all()
        assert (result >= values.amin(-1) - 1e-6).all()
    assert torch.allclose(smooth_max(values, beta=1e4), values.amax(-1), atol=1e-3)


def test_smooth_operators_give_every_input_a_gradient():
    left = torch.tensor([0.5], requires_grad=True)
    right = torch.tensor([-0.5], requires_grad=True)
    (smooth_or(left, right, beta=10.0) + smooth_and(left, right, beta=10.0)).sum().backward()

    assert left.grad.item() != 0 and right.grad.item() != 0
