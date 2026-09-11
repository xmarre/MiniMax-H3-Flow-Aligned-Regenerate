from __future__ import annotations

import pytest
import torch
from test_handoff import FakeLearnedProvider

from h3_flow_regenerate.handoff import ProgressiveTargetInputConfig
from h3_flow_regenerate.state_transport import (
    HANDOFF_STATE_POLICY_ENDPOINT_RESIDUAL_BICUBIC_V1,
    HANDOFF_STATE_POLICY_LEGACY,
    HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1,
    resolve_handoff_state_policy,
)
from h3_flow_regenerate.target_sparse_node import H3ProgressiveMixedGridHandoff, H3ProgressiveTargetSparseHandoff


def test_omitted_policy_keeps_serialized_mixed_grid_legacy_behavior():
    provider = FakeLearnedProvider()
    config = ProgressiveTargetInputConfig(
        source_scale=0.7,
        exact_prefix_mode="mixed_grid_low_suffix",
        transfer_mode="learned_3d",
        learned_upscaler=provider,
    )
    assert config.handoff_state_policy is None
    assert resolve_handoff_state_policy(config.handoff_state_policy) == HANDOFF_STATE_POLICY_LEGACY


def test_velocity_policy_is_restricted_to_mixed_grid_learned_transfer():
    provider = FakeLearnedProvider()
    config = ProgressiveTargetInputConfig(
        source_scale=0.7,
        exact_prefix_mode="mixed_grid_low_suffix",
        transfer_mode="learned_3d",
        learned_upscaler=provider,
        handoff_state_policy=HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1,
    )
    assert config.handoff_state_policy == HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1

    with pytest.raises(ValueError, match="only supported by mixed-grid Continuum"):
        ProgressiveTargetInputConfig(
            source_scale=0.7,
            handoff_state_policy=HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1,
        )

    with pytest.raises(ValueError, match="unsupported handoff_state_policy"):
        ProgressiveTargetInputConfig(
            source_scale=0.7,
            exact_prefix_mode="mixed_grid_low_suffix",
            transfer_mode="learned_3d",
            learned_upscaler=provider,
            handoff_state_policy="endpoint_residual_v1",
        )


def test_mixed_grid_ui_exposes_legacy_default_without_changing_target_sparse_ui():
    mixed = H3ProgressiveMixedGridHandoff.INPUT_TYPES()
    state_policy = mixed["optional"]["handoff_state_policy"]
    assert state_policy[0] == [
        HANDOFF_STATE_POLICY_LEGACY,
        HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1,
        HANDOFF_STATE_POLICY_ENDPOINT_RESIDUAL_BICUBIC_V1,
    ]
    assert state_policy[1]["default"] == HANDOFF_STATE_POLICY_LEGACY

    sparse = H3ProgressiveTargetSparseHandoff.INPUT_TYPES()
    assert "handoff_state_policy" not in sparse.get("required", {})
    assert "handoff_state_policy" not in sparse.get("optional", {})


def test_velocity_policy_does_not_require_or_create_random_state_at_configuration_time():
    provider = FakeLearnedProvider(output=torch.zeros(1, 24, 2, 8, 8))
    before = torch.random.get_rng_state().clone()
    ProgressiveTargetInputConfig(
        source_latent_h=4,
        source_latent_w=4,
        exact_prefix_mode="mixed_grid_low_suffix",
        transfer_mode="learned_3d",
        learned_upscaler=provider,
        handoff_state_policy=HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1,
    )
    assert torch.equal(torch.random.get_rng_state(), before)
