from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from h3_flow_regenerate.transfer_lattice import H3PatchLatticeTransferProvider, measure_paired_prefix_affine


def test_physical_transfer_uses_explicit_provider_capability_once():
    calls = []
    provider = SimpleNamespace(
        h3_patch_lattice_api=2,
        model_name="checkpoint",
        upscale_clean_video=lambda *_a, **_k: pytest.fail("half-pixel path executed"),
        upscale_clean_video_h3_patch_lattice=lambda value, **kw: calls.append((value, kw)) or value,
    )
    selected = H3PatchLatticeTransferProvider(provider)
    video = torch.randn(1, 24, 5, 4, 6)
    before = video.clone()
    assert selected.upscale_clean_video(video, target_h=8, target_w=12) is video
    assert selected.model_name == "checkpoint"
    assert selected.calls == len(calls) == 1
    assert calls[0][0] is video
    assert calls[0][1] == {"target_h": 8, "target_w": 12}
    assert torch.equal(video, before)


@pytest.mark.parametrize("api", [None, 1])
def test_legacy_provider_is_rejected_before_a_model_call(api):
    with pytest.raises(RuntimeError, match="h3_patch_lattice_api=2"):
        H3PatchLatticeTransferProvider(
            SimpleNamespace(h3_patch_lattice_api=api, upscale_clean_video_h3_patch_lattice=lambda: None)
        )


def test_same_frame_affine_diagnostic_observes_scale_and_does_not_modify_output():
    generator = torch.Generator().manual_seed(814)
    exact = F.avg_pool2d(torch.randn(3, 12, 40, 48, generator=generator), 5, stride=1, padding=2)
    y, x = torch.meshgrid(torch.arange(40), torch.arange(48), indexing="ij")
    dx, dy = 0.4 + 0.01 * (x - 23.5), -0.25 - 0.007 * (y - 19.5)
    grid = torch.stack((2 * (x - dx) / 47 - 1, 2 * (y - dy) / 39 - 1), -1)[None].expand(3, -1, -1, -1)
    learned = F.grid_sample(exact, grid, align_corners=True, padding_mode="border")
    exact, learned = exact.permute(1, 0, 2, 3)[None], learned.permute(1, 0, 2, 3)[None]
    before = learned.clone()
    with torch.inference_mode():
        receipt = measure_paired_prefix_affine(learned, exact, prefix_t=3)
    assert receipt["status"] == "measured"
    assert receipt["output_mutated"] is False
    assert receipt["extra_provider_calls"] == receipt["extra_h3_nfe"] == 0
    for frame in receipt["frames"]:
        assert frame["affine_huber"] < frame["zero_huber"] * 0.8
        assert frame["center_dx_dy_cells"][0] == pytest.approx(-0.4, abs=0.2)
        assert frame["center_dx_dy_cells"][1] == pytest.approx(0.25, abs=0.2)
        assert abs(frame["affine_displacement_gradients"][0][0]) > 0.003
        assert abs(frame["affine_displacement_gradients"][1][1]) > 0.003
    assert torch.equal(learned, before)
