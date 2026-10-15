import pytest
import torch

from h3_flow_regenerate.geometry import pack_streams
from h3_flow_regenerate.low_cfg_branch_dc import (
    capture_low_cfg_branch_dc,
    low_cfg_branch_dc_receipt,
    measure_cfg_branch_dc,
)

PREFIX_T, TEMPORAL, C = 12, 16, 24
SHAPES = [(1, C, TEMPORAL, 4, 6), (1, 32, 2, 8)]


def _packed(dc_prefix, dc_suffix, seed):
    generator = torch.Generator().manual_seed(seed)
    video = torch.randn(1, C, TEMPORAL, 4, 6, generator=generator)
    video = video - video.mean(dim=(-2, -1), keepdim=True)
    video[:, :, :PREFIX_T] += dc_prefix.view(1, C, 1, 1, 1)
    video[:, :, PREFIX_T:] += dc_suffix.view(1, C, 1, 1, 1)
    return pack_streams((video, torch.randn(1, 32, 2, 8, generator=generator)))[0]


def _args(cond, uncond, scale=4.0, sigma=0.95):
    guided = uncond + scale * (cond - uncond) if uncond is not None else cond
    return {
        "input": _packed(torch.zeros(C), torch.zeros(C), 9),
        "cond_denoised": cond,
        "uncond_denoised": uncond,
        "denoised": guided,
        "cond_scale": scale,
        "sigma": torch.tensor([sigma]),
    }


def test_branches_separate_a_prompt_selected_tone_from_an_unfollowed_context():
    base = torch.randn(C, generator=torch.Generator().manual_seed(1))
    dark = -0.4 * torch.ones(C)
    factors = torch.full((C, 3), 0.1)
    row = measure_cfg_branch_dc(
        _args(_packed(base, base + dark, 2), _packed(base, base, 3)), SHAPES, PREFIX_T, preview_factors=factors
    )
    assert row["tokens"] == [PREFIX_T - 1, PREFIX_T, PREFIX_T + 1]
    assert row["sigma"] == pytest.approx(0.95) and row["cond_scale"] == pytest.approx(4.0)
    torch.testing.assert_close(torch.tensor(row["cond"]["boundary_step"]), dark, atol=1e-5, rtol=0)
    assert row["uncond"]["boundary_step_rms"] < 1e-5
    assert row["guided"]["boundary_step_rms"] == pytest.approx(4 * 0.4, rel=1e-4)
    assert row["cond"]["preview_luma_boundary_step"] == pytest.approx(-0.4 * C * 0.1, rel=1e-4)


def test_capture_observes_without_changing_the_prediction_and_restores_options():
    existing = []

    def other(args):
        existing.append(True)
        return args["denoised"]

    options = {"sampler_post_cfg_function": [other]}
    rows = []
    args = _args(_packed(torch.zeros(C), torch.ones(C), 2), None)
    with capture_low_cfg_branch_dc(options, SHAPES, PREFIX_T, stage="low", rows=rows):
        hooks = options["sampler_post_cfg_function"]
        assert hooks[0] is other and len(hooks) == 2
        result = args["denoised"]
        for hook in hooks:
            result = hook({**args, "denoised": result})
    assert result is args["denoised"]
    assert options["sampler_post_cfg_function"] == [other] and existing == [True]
    receipt = low_cfg_branch_dc_receipt(rows)
    assert receipt["status"] == "measured" and receipt["evaluations"] == 1
    assert receipt["rows"][0]["uncond"] is None and receipt["summary"]["uncond"] is None
    assert receipt["summary"]["cond"]["first"]["stage"] == "low"

    bare = {}
    with capture_low_cfg_branch_dc(bare, SHAPES, PREFIX_T, stage="probe", rows=[]):
        assert len(bare["sampler_post_cfg_function"]) == 1
    assert "sampler_post_cfg_function" not in bare


def test_a_failed_evaluation_is_recorded_and_the_prediction_still_returned():
    rows = []
    options = {}
    args = _args(_packed(torch.zeros(C), torch.zeros(C), 2), None)
    with capture_low_cfg_branch_dc(options, [(1, C, 4, 4, 6), (1, 32, 2, 8)], PREFIX_T, stage="low", rows=rows):
        assert options["sampler_post_cfg_function"][0](args) is args["denoised"]
    receipt = low_cfg_branch_dc_receipt(rows)
    assert receipt["status"] == "error" and receipt["failed_evaluations"] == 1
    assert low_cfg_branch_dc_receipt([])["status"] == "not_observed"
