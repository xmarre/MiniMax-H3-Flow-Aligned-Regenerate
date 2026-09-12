from __future__ import annotations

import torch

from h3_flow_regenerate.high_stage_diagnostics import (
    make_high_stage_diagnostic_contract,
    record_video_boundary,
)
from h3_flow_regenerate.metrics import H3FlowMetrics


def test_denoised_boundary_registration_is_recorded_without_mutation():
    torch.manual_seed(1234)
    reference = torch.randn((1, 24, 1, 24, 24), dtype=torch.float32)
    moving = torch.roll(reference, shifts=1, dims=-1)
    video = torch.cat((reference, moving), dim=2)
    before = video.clone()

    contract = make_high_stage_diagnostic_contract(
        prefix_t=1,
        shapes=[tuple(video.shape), (1, 32, 2, 1)],
        phases=((0, "single"),),
        sampler="sample_res_multistep",
    )
    metrics = H3FlowMetrics()

    record_video_boundary(
        metrics,
        "mixed_grid_high_guided_boundary",
        video,
        contract,
        {
            "logical_step": 0,
            "sigma": 0.5,
            "coordinate": 0.375,
            "actual": True,
            "provenance": "actual",
            "solver_phase": "single",
            "solver_outer_step": 0,
            "spectrum_step_id": 0,
            "prefix_t": 1,
            "sampler": "sample_res_multistep",
        },
    )

    assert torch.equal(video, before)
    fields = metrics.events[-1].fields
    assert fields["registration_available"] is True
    assert fields["registration_reason"] == "measured"
    assert fields["registration_transform"] is not None
    assert fields["registration_identity_error"] is not None
    assert fields["registration_aligned_error"] is not None
    assert fields["registration_aligned_error"] < fields["registration_identity_error"]
    assert fields["registration_improvement"] > 0.0
