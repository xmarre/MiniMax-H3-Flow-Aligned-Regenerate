from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate.contracts import H3FlowTrajectory, TrajectorySample
from h3_flow_regenerate.metrics import H3FlowMetrics
from h3_flow_regenerate.partitioned_diagnostics import (
    PARTITIONED_AV_HANDOFF_SOURCE_MAIN,
    PARTITIONED_AV_HANDOFF_SOURCE_SHADOW,
    PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_MAIN,
    PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_SHADOW,
)
from h3_flow_regenerate.partitioned_scheduler import (
    PartitionedPreflightUnsupported,
    _isolated_shadow_trajectory_capture,
    _validate_guidance_trajectory_shadow_configuration,
)
from h3_flow_regenerate.runtime import FlowBinding


def _sampler():
    def sample_test(*args, **kwargs):
        del args, kwargs

    return SimpleNamespace(sampler_function=sample_test, extra_options={})


def test_guidance_trajectory_shadow_requires_av_shadow_handoff():
    _validate_guidance_trajectory_shadow_configuration(
        PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_MAIN,
        av_handoff_source=PARTITIONED_AV_HANDOFF_SOURCE_MAIN,
    )
    _validate_guidance_trajectory_shadow_configuration(
        PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_SHADOW,
        av_handoff_source=PARTITIONED_AV_HANDOFF_SOURCE_SHADOW,
    )
    with pytest.raises(PartitionedPreflightUnsupported, match="av_handoff_source"):
        _validate_guidance_trajectory_shadow_configuration(
            PARTITIONED_GUIDANCE_TRAJECTORY_SOURCE_SHADOW,
            av_handoff_source=PARTITIONED_AV_HANDOFF_SOURCE_MAIN,
        )


def test_isolated_shadow_trajectory_capture_restores_shared_handle_and_main_identity():
    shared = H3FlowTrajectory(storage="system_ram", max_runs=4)
    metrics = H3FlowMetrics()
    binding = FlowBinding(
        trajectory=shared,
        metrics=metrics,
        capture_enabled=True,
        captured_run_id="main-run",
    )
    guider = SimpleNamespace(model_options={}, original_conds={})
    shapes = [(1, 24, 4, 2, 3), (1, 32, 2, 8)]
    sigmas = torch.tensor([1.0, 0.5], dtype=torch.float32)

    with _isolated_shadow_trajectory_capture(
        binding,
        guider,
        _sampler(),
        sigmas,
        shapes,
        enabled=True,
    ) as holder:
        assert holder == {}
        assert binding.trajectory is not shared
        active = binding.active_capture
        assert active is not None
        binding.trajectory.append(
            active.run_id,
            TrajectorySample(
                coordinate=1.0,
                video_sigma=1.0,
                audio_sigma=1.0,
                outer_step=0,
                call_index=0,
                phase="single",
                provenance="actual",
                video_x0=torch.zeros(shapes[0]),
            ),
        )

    run = holder["run"]
    assert run.complete is True
    assert len(run.exact_samples()) == 1
    assert binding.trajectory is shared
    assert binding.captured_run_id == "main-run"
    assert binding.active_capture is None
    assert shared.runs == ()


def test_isolated_shadow_trajectory_capture_aborts_and_restores_on_error():
    shared = H3FlowTrajectory(storage="system_ram", max_runs=4)
    binding = FlowBinding(
        trajectory=shared,
        metrics=H3FlowMetrics(),
        capture_enabled=True,
        captured_run_id="main-run",
    )
    guider = SimpleNamespace(model_options={}, original_conds={})
    shapes = [(1, 24, 4, 2, 3), (1, 32, 2, 8)]

    with pytest.raises(RuntimeError, match="sentinel"), _isolated_shadow_trajectory_capture(
        binding,
        guider,
        _sampler(),
        torch.tensor([1.0, 0.5], dtype=torch.float32),
        shapes,
        enabled=True,
    ):
        raise RuntimeError("sentinel")

    assert binding.trajectory is shared
    assert binding.captured_run_id == "main-run"
    assert binding.active_capture is None
    assert shared.runs == ()
