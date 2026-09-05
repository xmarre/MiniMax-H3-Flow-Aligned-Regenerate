from __future__ import annotations

import ast
import copy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate.decode_context import H3ContinuumDecodeContext, prepare_decode_context


def frames(t):
    return (t - 2) // 5 * 17 + 5


def sequence(prefix=12, count=4):
    generator = torch.Generator().manual_seed(82)
    timeline = torch.randn(1, 24, 22 + 15 * (count - 1), 2, 3, generator=generator)
    videos = [timeline[:, :, :22].clone()]
    groups = [{"total_frames": frames(22), "trim_frames": 0, "net_frames": frames(22), "expected_video_latent_t": 22}]
    for index in range(1, count):
        stop = 22 + 15 * index
        video = timeline[:, :, stop - 15 - prefix : stop].clone()
        videos.append(video)
        groups.append(
            {
                "total_frames": frames(prefix + 15),
                "trim_frames": frames(prefix),
                "net_frames": 51,
                "expected_video_latent_t": prefix + 15,
            }
        )
    return (
        timeline,
        [{"samples": v} for v in videos],
        {
            "magic": "H3_CONTINUUM_ASSEMBLY_PLAN",
            "schema_version": 1,
            "fps": 24,
            "chunks": groups,
        },
    )


@pytest.mark.parametrize("prefix", [2, 7, 12])
def test_exact_context_is_bounded_and_does_not_mutate_accepted_latents(prefix):
    _, latents, plan = sequence(prefix)
    before = [x["samples"].clone() for x in latents]
    old_plan = copy.deepcopy(plan)
    result, report = H3ContinuumDecodeContext().prepare(latents, [plan])
    assert "3/3 exact boundaries" in report
    for index in range(3):
        assert result[index]["samples"].shape[2] == before[index].shape[2] + 5
        assert torch.equal(result[index]["samples"][:, :, :-5], before[index])
        assert torch.equal(result[index]["samples"][:, :, -5:], before[index + 1][:, :, prefix : prefix + 5])
        assert result[index]["samples"].data_ptr() != latents[index]["samples"].data_ptr()
    assert result[-1] is latents[-1]
    assert plan == old_plan
    assert all(torch.equal(x["samples"], v) for x, v in zip(latents, before, strict=True))


def test_nonexact_or_guide_boundary_is_not_silently_replaced():
    _, latents, plan = sequence()
    latents[1]["samples"][:, :, 0] += 0.1
    result, report = prepare_decode_context(latents, plan)
    assert result[0] is latents[0]
    assert "protected overlap is not exact" in report
    assert "2/3 exact boundaries" in report


def test_physical_decode_groups_take_precedence_over_logical_chunks():
    _, latents, plan = sequence(count=2)
    plan["decode_groups"] = plan["chunks"]
    plan["chunks"] = [{}] * 3
    result, report = prepare_decode_context(latents, plan)
    assert len(result) == 2
    assert "1/1 exact boundaries" in report


def test_single_chunk_is_identity():
    _, latents, plan = sequence(count=1)
    result, report = prepare_decode_context(latents, plan)
    assert result[0] is latents[0]
    assert "0/0 exact boundaries" in report


@pytest.mark.parametrize("corruption", ["count", "duration", "length", "schema"])
def test_stale_plan_cannot_silently_shift_video(corruption):
    _, latents, plan = sequence()
    if corruption == "count":
        latents.pop()
    elif corruption == "duration":
        plan["chunks"][1]["total_frames"] += 1
    elif corruption == "length":
        plan["chunks"][1]["expected_video_latent_t"] += 5
    else:
        plan["schema_version"] += 1
    with pytest.raises(ValueError):
        prepare_decode_context(latents, plan)


def native_temporal_decoder():
    """Execute pinned native temporal decode code with a context-sensitive pixel oracle.

    The learned decoder is replaced; the window selection, padding, blending,
    frame trimming and writes execute verbatim from ComfyUI. This proves temporal
    equivalence, not perceptual quality of an actual checkpoint.
    """
    root = os.environ.get("COMFYUI_ROOT")
    if not root:
        pytest.skip("native VAE temporal oracle runs in source-contract CI with COMFYUI_ROOT")
    source = Path(root, "comfy/ldm/minimax/vae.py").read_text()
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MiniMaxH3VideoVAE")
    names = {
        "blend",
        "decode_output_shape",
        "_decode_temporal_pad_frames",
        "_decode_temporal_frame_plan",
        "_decode_temporal_chunks",
        "decode_temporal",
    }
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in methods} == names
    extracted = ast.ClassDef(name="NativeTemporal", bases=[], keywords=[], body=methods, decorator_list=[])
    namespace = {"torch": torch}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[extracted], type_ignores=[])), "native_vae", "exec"),
        namespace,
    )
    decoder = namespace["NativeTemporal"]()
    decoder.tokens_chunk_size = 5
    decoder.token_overlap = 2
    decoder.token_drop = 3
    decoder.vae_ratio_t = 4
    decoder.frame_pre_padding = 3
    decoder.frame_overlap = 5
    decoder.clip_length = 17
    decoder.vae_ratio = 1
    decoder.decoder = SimpleNamespace(out_channels=3)
    decoder._finalize_pixels = lambda x: x
    # Each output depends on the entire seven-token window, like native ViT3D
    # attention. This exposes discontinuities invisible to a framewise fake VAE.
    decoder._adaptive_decode = lambda z: (z[:, :3] + z[:, :3].mean(dim=2, keepdim=True)).repeat_interleave(4, dim=2)

    def decode(z):
        return decoder.decode_temporal(z, output_buffer=torch.empty(decoder.decode_output_shape(z.shape)))

    return decode


@pytest.mark.parametrize("prefix", [2, 7, 12])
def test_native_window_oracle_matches_continuous_decode_and_reproduces_old_seam(prefix):
    decode = native_temporal_decoder()
    timeline, latents, plan = sequence(prefix)
    extended, _ = prepare_decode_context(latents, plan)

    def assemble(inputs):
        return torch.cat(
            [
                decode(x["samples"])[:, :, p["trim_frames"] : p["total_frames"]]
                for x, p in zip(inputs, plan["chunks"], strict=True)
            ],
            dim=2,
        )

    expected = decode(timeline)
    corrected = assemble(extended)
    old = assemble(latents)
    assert torch.equal(corrected, expected)
    assert not torch.equal(old, expected)
    # Failure is localized to the five-frame decode overlap before each join.
    cursor = 0
    bad_frames = torch.zeros(expected.shape[2], dtype=torch.bool)
    for chunk in plan["chunks"][:-1]:
        cursor += chunk["net_frames"]
        bad_frames[cursor - 5 : cursor] = True
    assert torch.equal(old[:, :, ~bad_frames], expected[:, :, ~bad_frames])
