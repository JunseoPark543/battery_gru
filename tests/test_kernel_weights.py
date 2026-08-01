from __future__ import annotations

import numpy as np
import torch

from battery_weighted_maml.meta.kernel_weights import compute_target_aware_weights


def test_alpha_nonnegative_and_sums_to_one():
    sources = [torch.tensor([[0.0], [0.1]]), torch.tensor([[1.0], [1.1]])]
    target = torch.tensor([[0.05], [0.15]])
    result = compute_target_aware_weights(sources, target, sigma=0.5)
    assert torch.all(result.alpha >= 0)
    assert torch.isclose(result.alpha.sum(), torch.tensor(1.0), atol=1e-5)


def test_identical_sources_receive_symmetric_weights():
    point_set = torch.tensor([[0.0], [0.2], [0.4]])
    result = compute_target_aware_weights([point_set, point_set.clone()], point_set, sigma=0.3)
    np.testing.assert_allclose(result.alpha.numpy(), [0.5, 0.5], atol=2e-4)


def test_source_closest_to_target_receives_larger_weight():
    close = torch.tensor([[0.00], [0.10], [0.20]])
    far = torch.tensor([[3.00], [3.10], [3.20]])
    target = torch.tensor([[0.02], [0.12], [0.22]])
    result = compute_target_aware_weights([close, far], target, sigma=0.4)
    assert result.alpha[0] > result.alpha[1]

