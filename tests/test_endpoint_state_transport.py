from __future__ import annotations

import pytest
import torch

from h3_flow_regenerate.geometry import resize_spatial_5d
from h3_flow_regenerate.state_transport import (
    HANDOFF_STATE_POLICY_ENDPOINT_RESIDUAL_BICUBIC_V1,
    HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1,
    measure_state_transport_comparison_v1,
    resolve_handoff_state_policy,
    transport_endpoint_residual_bicubic_v1,
    transport_velocity_bicubic_v1,
)


def test_endpoint_policy_resolves_without_changing_legacy_default():
    assert (
        resolve_handoff_state_policy(HANDOFF_STATE_POLICY_ENDPOINT_RESIDUAL_BICUBIC_V1)
        == HANDOFF_STATE_POLICY_ENDPOINT_RESIDUAL_BICUBIC_V1
    )


def test_endpoint_and_velocity_are_distinct_for_nonlinear_clean_reanchor():
    sigma = 0.8
    prefix_t = 1
    source_clean = torch.full((1, 24, 3, 4, 4), 0.5, dtype=torch.float32)
    source_state = source_clean + 0.75
    target_clean = torch.full((1, 24, 3, 6, 8), 2.0, dtype=torch.float32)

    velocity, _ = transport_velocity_bicubic_v1(source_state, source_clean, target_clean, prefix_t=prefix_t)
    endpoint, metrics = transport_endpoint_residual_bicubic_v1(
        source_state,
        source_clean,
        target_clean,
        prefix_t=prefix_t,
        sigma=sigma,
    )
    lifted_source_clean = resize_spatial_5d(source_clean, 6, 8, mode="bicubic")
    expected_delta = sigma * (target_clean[:, :, prefix_t:] - lifted_source_clean[:, :, prefix_t:])
    torch.testing.assert_close(
        velocity[:, :, prefix_t:] - endpoint[:, :, prefix_t:],
        expected_delta,
        rtol=2e-5,
        atol=2e-5,
    )
    assert metrics["state_transport_clean_coefficient"] == pytest.approx(1.0 - sigma)
    assert metrics["state_transport_residual_kind"] == "effective_endpoint"


def test_endpoint_linear_clean_transfer_reduces_to_direct_state_lift():
    sigma = 0.8780487775802612
    prefix_t = 1
    source_clean = torch.linspace(-0.5, 0.7, steps=1 * 24 * 4 * 4 * 6).reshape(1, 24, 4, 4, 6)
    source_state = source_clean + torch.linspace(-0.2, 0.3, steps=source_clean.numel()).reshape_as(source_clean)
    target_clean = resize_spatial_5d(source_clean, 8, 10, mode="bicubic")
    endpoint, _ = transport_endpoint_residual_bicubic_v1(
        source_state,
        source_clean,
        target_clean,
        prefix_t=prefix_t,
        sigma=sigma,
    )
    direct = resize_spatial_5d(source_state, 8, 10, mode="bicubic")
    torch.testing.assert_close(endpoint[:, :, prefix_t:], direct[:, :, prefix_t:], rtol=2e-5, atol=2e-5)


def test_endpoint_rejects_sigma_endpoints_and_comparison_records_policy_gap():
    source_clean = torch.zeros((1, 24, 3, 4, 4), dtype=torch.float32)
    source_state = torch.ones_like(source_clean)
    target_clean = torch.full((1, 24, 3, 6, 8), 2.0)
    for sigma in (0.0, 1.0, float("nan")):
        with pytest.raises(ValueError, match="sigma inside"):
            transport_endpoint_residual_bicubic_v1(
                source_state,
                source_clean,
                target_clean,
                prefix_t=1,
                sigma=sigma,
            )

    metrics = measure_state_transport_comparison_v1(
        source_state,
        source_clean,
        target_clean,
        prefix_t=1,
        sigma=0.8,
    )
    assert metrics["state_transport_comparison_version"] == 1
    assert metrics["state_transport_clean_reanchor_delta_rms"] == pytest.approx(2.0, rel=1e-6)
    assert metrics["state_transport_velocity_endpoint_expected_delta_rms"] == pytest.approx(1.6, rel=1e-6)
    for key, value in metrics.items():
        if key.endswith("_rms"):
            assert torch.isfinite(torch.tensor(value))
