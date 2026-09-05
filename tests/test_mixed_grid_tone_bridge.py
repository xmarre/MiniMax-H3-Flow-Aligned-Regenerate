from __future__ import annotations

import pytest
import torch

from h3_flow_regenerate.tone_bridge import (
    apply_suffix_dc_bridge,
    disabled_suffix_dc_bridge_metrics,
    map_clean_bridge_to_conditional_state,
)


def _fixture(dtype=torch.float32):
    learned = torch.zeros(1, 3, 5, 2, 2, dtype=dtype)
    learned[:, 0, 1] = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
    learned[:, 1, 1] = 10.0
    learned[:, 2, 1] = -4.0
    learned[:, 0, 2] = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=dtype)
    learned[:, 1, 2] = 12.0
    learned[:, 2, 2] = 1.0
    learned[:, :, 3] = 20.0
    learned[:, :, 4] = 30.0
    exact = learned[:, :, :2].clone()
    exact[:, 0, 1] += 3.0
    exact[:, 1, 1] -= 2.0
    exact[:, 2, 1] += 5.0
    return learned, exact


def test_weight_one_restores_native_per_channel_dc_boundary_exactly():
    learned, exact = _fixture()
    corrected, metrics = apply_suffix_dc_bridge(learned, exact)
    learned_native = learned[:, :, 2].float().mean((-2, -1)) - learned[:, :, 1].float().mean((-2, -1))
    corrected_exact = corrected[:, :, 2].float().mean((-2, -1)) - exact[:, :, 1].float().mean((-2, -1))
    assert torch.allclose(corrected_exact, learned_native, rtol=0.0, atol=1e-6)
    assert metrics["suffix_dc_bridge_corrected_tokens"] == 1
    assert metrics["suffix_dc_bridge_first_weight"] == 1.0


def test_bridge_is_per_channel_not_one_global_scalar():
    learned, exact = _fixture()
    corrected, _ = apply_suffix_dc_bridge(learned, exact)
    shifts = (corrected[:, :, 2] - learned[:, :, 2]).float().mean((-2, -1))
    assert shifts.shape == (1, 3)
    assert torch.allclose(shifts, torch.tensor([[3.0, -2.0, 5.0]]))


def test_bridge_preserves_spatial_detail_residual_and_later_suffix_tokens():
    learned, exact = _fixture()
    original = learned.clone()
    corrected, _ = apply_suffix_dc_bridge(learned, exact)
    before = learned[:, :, 2].float()
    after = corrected[:, :, 2].float()
    before_residual = before - before.mean((-2, -1), keepdim=True)
    after_residual = after - after.mean((-2, -1), keepdim=True)
    assert torch.allclose(after_residual, before_residual, rtol=0.0, atol=1e-6)
    assert torch.equal(corrected[:, :, :2], original[:, :, :2])
    assert torch.equal(corrected[:, :, 3:], original[:, :, 3:])
    assert torch.equal(learned, original)


def test_single_suffix_token_is_supported():
    learned, exact = _fixture()
    learned = learned[:, :, :3].clone()
    exact = exact.clone()
    corrected, metrics = apply_suffix_dc_bridge(learned, exact, weights=(1.0, 0.5))
    assert corrected.shape == learned.shape
    assert metrics["suffix_dc_bridge_corrected_tokens"] == 1
    assert metrics["suffix_dc_bridge_last_weight"] == 1.0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_bridge_preserves_dtype_and_device(dtype):
    learned, exact = _fixture(dtype=dtype)
    corrected, _ = apply_suffix_dc_bridge(learned, exact)
    assert corrected.dtype == dtype
    assert corrected.device == learned.device


def test_conditional_state_mapping_matches_pre_renoise_bridge():
    learned, exact = _fixture()
    corrected, metrics = apply_suffix_dc_bridge(learned, exact)
    sigma = 0.73
    torch.manual_seed(8)
    noise = torch.randn_like(learned)
    state = (1.0 - sigma) * learned + sigma * noise
    mapped = map_clean_bridge_to_conditional_state(
        state,
        learned,
        corrected,
        sigma=sigma,
        prefix_t=2,
        corrected_tokens=int(metrics["suffix_dc_bridge_corrected_tokens"]),
    )
    direct = (1.0 - sigma) * corrected + sigma * noise
    assert torch.allclose(mapped, direct, rtol=1e-6, atol=1e-6)
    assert torch.equal(mapped[:, :, :2], state[:, :, :2])
    assert torch.equal(mapped[:, :, 3:], state[:, :, 3:])


def test_disabled_metrics_are_explicit_and_zero_cost_semantics():
    metrics = disabled_suffix_dc_bridge_metrics(prefix_t=12)
    assert metrics["suffix_dc_bridge_enabled"] is False
    assert metrics["suffix_dc_bridge_corrected_tokens"] == 0
    assert metrics["suffix_dc_bridge_delta_rms"] == 0.0


def test_bridge_rejects_nonfinite_and_malformed_geometry():
    learned, exact = _fixture()
    bad = learned.clone()
    bad[:, :, 2, 0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="NaN or Inf"):
        apply_suffix_dc_bridge(bad, exact)
    with pytest.raises(ValueError, match="spatial geometry"):
        apply_suffix_dc_bridge(learned, exact[..., :1, :])
    with pytest.raises(ValueError, match="prefix shorter"):
        apply_suffix_dc_bridge(learned, learned.clone())


def test_bridge_rejects_invalid_weights_and_state_mapping_ranges():
    learned, exact = _fixture()
    with pytest.raises(ValueError, match="weights"):
        apply_suffix_dc_bridge(learned, exact, weights=())
    with pytest.raises(ValueError, match="weights"):
        apply_suffix_dc_bridge(learned, exact, weights=(1.1,))
    corrected, _ = apply_suffix_dc_bridge(learned, exact)
    with pytest.raises(ValueError, match="corrected-token range"):
        map_clean_bridge_to_conditional_state(
            learned,
            learned,
            corrected,
            sigma=0.5,
            prefix_t=2,
            corrected_tokens=99,
        )
