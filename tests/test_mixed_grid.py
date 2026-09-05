from __future__ import annotations

import ast
import math
import os
from pathlib import Path
from types import ModuleType

import pytest
import torch

from h3_flow_regenerate.geometry import pack_streams
from h3_flow_regenerate.mixed_grid import (
    MixedGridPlan,
    build_mixed_grid_plan,
    carrier_layout,
    mixed_mod_segments,
    mixed_positions,
)


def inputs(t=7, prefix=2, h=8, w=12):
    video = torch.arange(24 * t * h * w, dtype=torch.float32).reshape(1, 24, t, h, w)
    audio = torch.randn(1, 32, 2, 11)
    packed, shapes = pack_streams((video, audio))
    mask = torch.ones_like(video)
    mask[:, :, :prefix] = 0
    return packed, shapes, pack_streams((mask, torch.ones_like(audio)))[0]


def test_observed_row_count_and_prefix_snapshot():
    packed, shapes, mask = inputs(t=62, prefix=12, h=48, w=64)
    noise = torch.randn_like(packed)
    plan = build_mixed_grid_plan(mask, shapes, packed, noise, source_h=34, source_w=44)
    assert plan.prefix_rows == 9216
    assert plan.mixed_rows == 27916
    before = plan.prefix.clone()
    before_noise = plan.prefix_noise.clone()
    packed.zero_()
    noise.zero_()
    assert torch.equal(plan.prefix, before)
    assert torch.equal(plan.prefix_noise, before_noise)


@pytest.mark.parametrize("kind", ["partial", "discontiguous", "soft", "nan", "none", "all"])
def test_reject_unsupported_masks(kind):
    packed, shapes, mask = inputs()
    noise = torch.randn_like(packed)
    if kind == "partial":
        mask[..., 0] = 1
    elif kind == "discontiguous":
        video = mask[..., : math.prod(shapes[0][1:])].reshape(shapes[0])
        video[:, :, [1, 3]] = video[:, :, [3, 1]]
    elif kind == "soft":
        mask[..., 1000] = 0.5
    elif kind == "nan":
        mask[..., 0] = float("nan")
    elif kind == "none":
        mask.fill_(1)
    else:
        mask.zero_()
    with pytest.raises(ValueError):
        build_mixed_grid_plan(mask, shapes, packed, noise, source_h=4, source_w=6)


def test_reject_invalid_sampler_noise():
    packed, shapes, mask = inputs()
    bad = torch.randn_like(packed)
    bad[..., 0] = float("nan")
    with pytest.raises(ValueError, match="sampler noise must be finite"):
        build_mixed_grid_plan(mask, shapes, packed, bad, source_h=4, source_w=6)


@pytest.fixture
def native():
    root = Path(os.environ.get("COMFYUI_ROOT", Path(__file__).resolve().parents[2] / "comfy"))
    path = root / "comfy/ldm/minimax/model.py"
    if not path.is_file():
        if os.environ.get("COMFYUI_ROOT"):
            raise FileNotFoundError(path)
        pytest.skip("native source oracle runs in source-contract CI with COMFYUI_ROOT")
    tree = ast.parse(path.read_text())
    names = {
        "PackedLayout",
        "patchify_video",
        "unpatchify_video",
        "_frame_grid",
        "_axis_from_sqrt_area",
        "_video_t_spans",
        "_video_t_grid",
        "_video_grid",
        "_audio_grid",
        "_ref_t_span",
    }
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    module = ModuleType("native_h3_oracle")
    module.__dict__.update(torch=torch, math=math, FRAME_PER_TOKEN=(1, 4, 4, 4, 4), FRAME_RESCALE=5 / 3)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("prefix_t", [1, 2, 4, 5, 6, 12])
def test_native_spatial_temporal_and_nonvideo_contract(native, prefix_t):
    plan = MixedGridPlan(torch.randn(1, 24, prefix_t, 8, 12), 15, 4, 6)
    payload = {"refs": [{"kind": "image", "latent_h": 4, "latent_w": 8}]}
    layout = carrier_layout(native, plan, 3, 11, payload)
    target = native.PackedLayout(3, 15, 8, 12, 11, refs=payload["refs"])
    low = native.PackedLayout(3, 15, 4, 6, 11, refs=payload["refs"])
    va = layout.segments[-1][0]
    positions = mixed_positions(native, plan, layout)
    assert torch.equal(layout.position_ids[:va], target.position_ids[:va])
    assert torch.equal(positions[va : va + plan.prefix_rows], target.position_ids[va : va + plan.prefix_rows])
    assert torch.equal(positions[va + plan.prefix_rows :], low.position_ids[va + prefix_t * plan.source_rows :])
    assert len(positions) == va + plan.mixed_rows
    # Actual low-grid rows round-trip, independently of target prefix density.
    suffix = torch.randn(1, 24, 15 - prefix_t, 4, 6)
    prefix_rows = native.patchify_video(plan.prefix)
    mixed = torch.cat((prefix_rows, native.patchify_video(suffix)))
    assert torch.equal(native.unpatchify_video(mixed[plan.prefix_rows :], 15 - prefix_t, 2, 3), suffix)


def test_per_row_timestep_expansion_preserves_suffix_and_nonvideo():
    plan = MixedGridPlan(torch.randn(1, 24, 2, 8, 12), 7, 4, 6)
    row = torch.cat((torch.full((12,), 9), torch.full((30,), 3)))
    result = mixed_mod_segments([(0, 5, 4), (5, 47, row)], plan, 5, 47)
    assert result[0] == (0, 5, 4)
    assert result[1][:2] == (5, 83)
    assert torch.equal(result[1][2][:48], torch.full((48,), 9))
    assert torch.equal(result[1][2][48:], row[12:])


def test_keyframes_keep_target_geometry(native):
    plan = MixedGridPlan(torch.randn(1, 24, 2, 8, 12), 7, 4, 6)
    payload = {"keyframes": [{"resolved_frame_index": 0, "latent": torch.randn(1, 24, 1, 8, 12)}]}
    layout = carrier_layout(native, plan, 3, 11, payload)
    assert next(b - a for a, b, kind in layout.segments if kind == "cond") == 24
    assert layout.signature == (3, 7, 4, 6, 11)


def test_mixed_node_requires_learned_transfer():
    from h3_flow_regenerate.handoff import ProgressiveTargetInputConfig
    from h3_flow_regenerate.target_sparse_node import H3ProgressiveMixedGridHandoff

    with pytest.raises(ValueError, match="requires learned_3d"):
        ProgressiveTargetInputConfig(source_scale=0.7, exact_prefix_mode="mixed_grid_low_suffix")
    inputs = H3ProgressiveMixedGridHandoff.INPUT_TYPES()
    transfer = next(group["handoff_transfer"] for group in inputs.values() if "handoff_transfer" in group)
    assert transfer[0] == ["learned_3d"]


@pytest.mark.parametrize("fail_stage", [None, "low", "probe", "high"])
def test_stage_lifetimes_learned_context_and_original_prefix(monkeypatch, fail_stage):
    import sys
    from types import SimpleNamespace

    from test_handoff import FakeLearnedProvider

    from h3_flow_regenerate.geometry import resize_spatial_5d, unpack_streams
    from h3_flow_regenerate.handoff import ProgressiveTargetInputConfig
    from h3_flow_regenerate.mixed_grid import MIXED_GRID_KEY
    from h3_flow_regenerate.runtime import FlowBinding, _run_progressive

    fake = ModuleType("comfy")
    fake.samplers = ModuleType("comfy.samplers")
    fake.samplers.KSAMPLER = lambda function, **kw: SimpleNamespace(sampler_function=function, extra_options={})
    monkeypatch.setitem(sys.modules, "comfy", fake)
    monkeypatch.setitem(sys.modules, "comfy.samplers", fake.samplers)
    packed, shapes, mask = inputs()
    base = SimpleNamespace(process_latent_in=lambda value: value, diffusion_model=SimpleNamespace(blocks=[]))
    guider = SimpleNamespace(
        model_options={"transformer_options": {}}, model_patcher=SimpleNamespace(model=base), conds={"positive": []}
    )
    binding = FlowBinding()
    provider = FakeLearnedProvider()
    config = ProgressiveTargetInputConfig(
        source_latent_h=4,
        source_latent_w=6,
        exact_prefix_mode="mixed_grid_low_suffix",
        transfer_mode="learned_3d",
        learned_upscaler=provider,
    )
    calls = []

    def execute(noise, latent, sampler, sigmas, call_mask, *args, latent_shapes):
        stage = guider.model_options["transformer_options"]["h3_flow_stage"]
        calls.append(stage)
        contract = guider.model_options["transformer_options"].get(MIXED_GRID_KEY)
        assert (contract is not None) == (stage != "high")
        if stage == fail_stage:
            raise RuntimeError("test stage failure")
        if stage == "high":
            assert torch.equal(call_mask, mask)
            assert latent_shapes == shapes
            binding.metrics.event("model_call", actual=True)
            return packed.clone()
        video_mask, _ = unpack_streams(call_mask, latent_shapes)
        assert torch.count_nonzero(video_mask[:, :, :2]) == 0
        if stage == "probe":
            clean = latent.clone()
            clean_video, _ = unpack_streams(clean, latent_shapes)
            clean_video[:, :, :2] = -999  # Must be replaced by original context before learned transfer.
            return clean
        return latent / (1 - sigmas[-1])

    sampler = SimpleNamespace(sampler_function=lambda: None, extra_options={})
    args = (
        execute,
        guider,
        binding,
        config,
        torch.randn_like(packed),
        packed,
        sampler,
        torch.tensor([1.0, 0.9, 0.7, 0.4, 0.0]),
        mask,
        None,
        True,
        7,
        list(shapes),
    )
    if fail_stage:
        with pytest.raises(RuntimeError, match="test stage failure"):
            _run_progressive(*args)
    else:
        assert torch.equal(_run_progressive(*args), packed)
        assert calls == ["low", "probe", "high"]
        original_video, _ = unpack_streams(packed, shapes)
        # Provider records its real input; authoritative context replaces the poisoned probe carrier.
        context = resize_spatial_5d(original_video[:, :, :2], 4, 6, mode="bicubic")
        assert torch.equal(provider.calls[0][0][:, :, :2], context)
    assert MIXED_GRID_KEY not in guider.model_options["transformer_options"]


def test_native_forward_uses_authoritative_prefix_and_real_suffix(monkeypatch, native):

    from h3_flow_regenerate.metrics import H3FlowMetrics
    from h3_flow_regenerate.mixed_grid import MIXED_GRID_KEY, mixed_diffusion_wrapper

    root = Path(os.environ.get("COMFYUI_ROOT", Path(__file__).resolve().parents[2] / "comfy"))
    monkeypatch.syspath_prepend(str(root))
    # Load the real native forward path with CPU operations, not a copied model.
    import comfy.cli_args

    comfy.cli_args.args.cpu = True
    import comfy.ldm.minimax.model as model_module
    import comfy.ops

    torch.manual_seed(71)
    model = model_module.MiniMaxH3Model(
        hidden_size=96,
        num_layers=2,
        token_refiner_num_layers=0,
        num_attention_heads=1,
        attention_head_dim=96,
        ffn_hidden_size=96,
        text_dim=96,
        timestep_input_dim=8,
        time_embed_hidden_size=16,
        time_embed_dim=8,
        rope_inv_freq_len=16,
        dtype=torch.float32,
        operations=comfy.ops.disable_weight_init,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.03)
        model.rope.inv_freq.fill_(0.01)
    seen = []

    class MixingBlock(torch.nn.Module):
        def forward(self, h, t_emb, segments, rope, transformer_options):
            seen.append((h.clone(), segments, rope.shape[1]))
            # Deliberate global dependence proves which prefix conditions the suffix.
            return h + h.mean(0)

    model.blocks = torch.nn.ModuleList([MixingBlock(), MixingBlock()])
    prefix = torch.randn(1, 24, 2, 8, 12)
    prefix_noise = torch.randn_like(prefix)
    plan = MixedGridPlan(prefix, 7, 4, 6, prefix_noise)
    x = [torch.randn(1, 24, 7, 4, 6), torch.randn(1, 32, 2, 11)]
    mask = torch.ones(1, 1, 7, 4, 6)
    mask[:, :, :2] = 0
    context = torch.randn(1, 3, 96)
    metrics = H3FlowMetrics()
    from h3_flow_regenerate.attention import make_layout_block_wrapper

    options = {
        MIXED_GRID_KEY: {"plan": plan, "metrics": metrics},
        "patches_replace": {"dit": {("double_block", 0): make_layout_block_wrapper(0, metrics)}},
    }
    observed = []

    class Executor:
        class_obj = model

        def __call__(self, x, timestep, context, options, **kwargs):
            # Simulate Spectrum's actual last-block observation outside Flow.
            existing = options["patches_replace"]["dit"][("double_block", 1)]

            def observe(args, extra):
                output = existing(args, extra)
                observed.append(output["img"].shape[0])
                return output

            options["patches_replace"]["dit"][("double_block", 1)] = observe
            return model._forward(x, timestep, context, options, **kwargs)

    with torch.no_grad():
        first = mixed_diffusion_wrapper(Executor(), x, torch.tensor([700.0]), context, options, denoise_mask=mask)
        altered = [x[0].clone(), x[1]]
        altered[0][:, :, :2] += 1000
        second = mixed_diffusion_wrapper(
            Executor(), altered, torch.tensor([700.0]), context, options, denoise_mask=mask
        )
        changed_plan = MixedGridPlan(prefix + 1, 7, 4, 6, prefix_noise)
        changed = {MIXED_GRID_KEY: {"plan": changed_plan, "metrics": metrics}}
        third = mixed_diffusion_wrapper(Executor(), x, torch.tensor([700.0]), context, changed, denoise_mask=mask)
        changed_noise_plan = MixedGridPlan(prefix, 7, 4, 6, prefix_noise + 1)
        changed_noise = {MIXED_GRID_KEY: {"plan": changed_noise_plan, "metrics": metrics}}
        fourth = mixed_diffusion_wrapper(
            Executor(), x, torch.tensor([700.0]), context, changed_noise, denoise_mask=mask
        )
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert not torch.equal(first[0][:, :, 2:], third[0][:, :, 2:])
    assert not torch.equal(first[0][:, :, 2:], fourth[0][:, :, 2:])
    assert torch.count_nonzero(first[0][:, :, :2]) == 0
    assert first[0].shape == x[0].shape
    assert all(count == 3 + 22 + 7 * 6 for count in observed)
    assert all(rows == 3 + 22 + plan.mixed_rows for _, _, rows in seen)
    native_prefix = (
        model_module.VISUAL_COND_TIMESTEP * prefix + (1.0 - model_module.VISUAL_COND_TIMESTEP) * prefix_noise
    )
    expected = model.video_patch_proj(model_module.patchify_video(native_prefix))
    assert torch.equal(seen[0][0][25 : 25 + plan.prefix_rows], expected)
    expected_suffix = model.video_patch_proj(model_module.patchify_video(x[0][:, :, 2:]))
    assert torch.equal(seen[0][0][25 + plan.prefix_rows :], expected_suffix)
