"""Exercise the actual Core block loop and Flow's per-forward label lifetime."""

from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate import partitioned_transformer as transform
from h3_flow_regenerate.metrics import H3FlowMetrics
from h3_flow_regenerate.partitioned_stage import PARTITIONED_STAGE_KEY, PartitionedStagePlan, PartitionedStageRuntime


@pytest.mark.parametrize("source_hw", [(4, 4), (8, 8)])
def test_native_modulation_validates_once_and_isolates_replacement_writes(monkeypatch, source_hw):
    cli = pytest.importorskip("comfy.cli_args")
    cli.args.cpu = True
    import comfy.ops
    from comfy.ldm.minimax.model import MiniMaxH3Model
    from comfy.patcher_extension import WrapperExecutor

    dm = MiniMaxH3Model(
        hidden_size=8,
        num_layers=50,
        token_refiner_num_layers=0,
        num_attention_heads=1,
        attention_head_dim=128,
        ffn_hidden_size=16,
        text_dim=8,
        time_embed_dim=2,
        adaln_curve_grid=4,
        rope_inv_freq_len=2,
        dtype=torch.float32,
        device="cpu",
        operations=comfy.ops.disable_weight_init,
    )
    with torch.no_grad():
        for param in dm.parameters():
            param.fill_(0.02)
        dm.adaln_t_table.fill_(0.1)
        dm.rope.inv_freq.fill_(0.2)
    for block in dm.blocks:
        block.forward = lambda img, *_args, **_kw: img.clone()

    prefix = torch.randn(1, 24, 2, 8, 8)
    plan = PartitionedStagePlan(
        prefix=prefix,
        prefix_noise=torch.randn_like(prefix),
        temporal=5,
        source_h=source_hw[0],
        source_w=source_hw[1],
    )
    metrics = H3FlowMetrics()
    runtime = PartitionedStageRuntime(plan=plan, metrics=metrics, audio_position_domain="source_carrier")
    labels = []

    def previous(args, extra):
        row = args["mod_segments"][-1][2]
        labels.append(row.clone())
        result = extra["original_block"](args)
        # A downstream replacement can write its forwarded index tensor without
        # corrupting either native Core labels or another block's expansion.
        row.fill_(777)
        return result

    options = {
        PARTITIONED_STAGE_KEY: runtime,
        "patches_replace": {"dit": {("double_block", i): previous for i in range(50)}},
    }
    video = torch.randn(1, 24, 5, *source_hw)
    audio = torch.randn(1, 32, 2, 5)
    mask = torch.ones(1, 1, 5, *source_hw)
    mask[:, :, :2] = 0
    context = torch.randn(1, 3, 8)
    calls = []
    original = transform.partitioned_mod_segments

    def observed(*args):
        calls.append(args[0])
        return original(*args)

    monkeypatch.setattr(transform, "partitioned_mod_segments", observed)
    with torch.no_grad():
        first = WrapperExecutor.new_class_executor(
            dm._forward,
            dm,
            [transform.partitioned_diffusion_wrapper],
        ).execute(
            [video, audio],
            torch.tensor([750.0]),
            context,
            transformer_options=options,
            denoise_mask=mask,
        )
    assert len(calls) == 1
    assert len(labels) == 50
    assert all(torch.equal(row, labels[0]) for row in labels)
    assert labels[0].numel() == plan.partitioned_rows
    assert labels[0][: plan.prefix_rows].unique().numel() == 1
    assert labels[0][plan.prefix_rows :].unique().numel() == 1
    assert labels[0][0] != labels[0][-1]
    assert torch.count_nonzero(first[0][:, :, :2]) == 0
    assert metrics.counters["partitioned_modulation_validations"] == 1
    labels.clear()
    with torch.no_grad():
        second = WrapperExecutor.new_class_executor(
            dm._forward,
            dm,
            [transform.partitioned_diffusion_wrapper],
        ).execute(
            [video, audio],
            torch.tensor([750.0]),
            context,
            transformer_options=options,
            denoise_mask=mask,
        )
    assert len(calls) == 2
    assert metrics.counters["partitioned_modulation_validations"] == 2
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))


@pytest.mark.parametrize("prefix_t,temporal", [(2, 17), (12, 47)])
@pytest.mark.parametrize("with_reference", [False, True])
@pytest.mark.parametrize("video_dtype", [torch.float32, torch.bfloat16])
def test_same_grid_probe_and_high_match_through_real_core_blocks_and_sol(
    monkeypatch, prefix_t, temporal, with_reference, video_dtype
):
    cli = pytest.importorskip("comfy.cli_args")
    cli.args.cpu = True
    pytest.importorskip("sol_h3.runtime")
    import comfy.ops
    from comfy.ldm.minimax.model import VISUAL_COND_TIMESTEP, MiniMaxH3Model
    from comfy.patcher_extension import WrapperExecutor
    from sol_h3 import partitioned_request
    from sol_h3.contracts import Config
    from sol_h3.runtime import _FORWARD, _REQUEST, Request

    from h3_flow_regenerate.partitioned_scheduler import (
        _partitioned_high_stage_contract,
        _partitioned_stage_contract,
    )
    from h3_flow_regenerate.runtime import _flow_stage_contract, _high_stage_contract

    generator = torch.Generator().manual_seed(93)
    dm = MiniMaxH3Model(
        hidden_size=8,
        num_layers=2,
        token_refiner_num_layers=0,
        num_attention_heads=1,
        attention_head_dim=128,
        ffn_hidden_size=16,
        text_dim=8,
        time_embed_dim=2,
        adaln_curve_grid=4,
        rope_inv_freq_len=2,
        dtype=torch.bfloat16,
        device="cpu",
        operations=comfy.ops.disable_weight_init,
    )
    dm.requires_grad_(False)
    with torch.no_grad():
        for param in dm.parameters():
            param.copy_(torch.randn(param.shape, generator=generator) * 0.02)
        dm.adaln_t_table.copy_(torch.randn(dm.adaln_t_table.shape, generator=generator))
        dm.rope.inv_freq.fill_(0.2)

    prefix = torch.randn(1, 24, prefix_t, 4, 6, generator=generator)
    noise = torch.randn(prefix.shape, generator=generator)
    plan = PartitionedStagePlan(prefix, temporal, 4, 6, noise)
    video = torch.randn(1, 24, temporal, 4, 6, generator=generator)
    video[:, :, :prefix_t] = VISUAL_COND_TIMESTEP * prefix + (1 - VISUAL_COND_TIMESTEP) * noise
    video = video.to(video_dtype)
    audio = torch.randn(1, 32, 2, 5, generator=generator, dtype=torch.bfloat16)
    mask = torch.ones(1, 1, temporal, 4, 6)
    mask[:, :, :prefix_t] = 0
    context = torch.randn(1, 3, 8, generator=generator, dtype=torch.bfloat16)
    guider = SimpleNamespace(model_options={"transformer_options": {}})
    metrics = H3FlowMetrics()
    payload = {}
    if with_reference:
        payload = {
            "refs": [{"kind": "image", "latent_h": 4, "latent_w": 6}],
            "cond_video_latents": [torch.randn(1, 24, 1, 4, 6, generator=generator)],
        }
    originals = [tensor.clone() for tensor in (video, audio, mask, prefix, noise)]
    # CPU arithmetic harness: Core blocks and Sol dense math are unchanged;
    # Sol's device-admission predicate is covered by its separate GPU contract.
    monkeypatch.setattr(partitioned_request, "_validate_thd", lambda *_args: None)

    def forward(*, native=False):
        state = Request(Config(backend="sol"))
        request = _REQUEST.set(state)
        execution = _FORWARD.set((dm, state, 0, set(), []))
        try:
            with torch.no_grad():
                output = WrapperExecutor.new_class_executor(
                    dm._forward, dm, [] if native else [transform.partitioned_diffusion_wrapper]
                ).execute(
                    [video, audio],
                    torch.tensor([878.04878]),
                    context,
                    # Core writes its layout/block index to its per-call options.
                    # Keep those native writes separate from the stage owner.
                    transformer_options={} if native else guider.model_options["transformer_options"],
                    minimax_payload=payload,
                    denoise_mask=mask,
                )
            assert state.dense_calls == (0 if native else len(dm.blocks))
            return output
        finally:
            _FORWARD.reset(execution)
            _REQUEST.reset(request)

    with (
        _flow_stage_contract(guider, "probe"),
        _high_stage_contract(guider),
        _partitioned_stage_contract(guider, plan, metrics),
    ):
        probe = forward()
    with _flow_stage_contract(guider, "high"), _partitioned_high_stage_contract(guider, plan, metrics):
        high = forward()

    assert all(torch.equal(a, b) for a, b in zip(probe, high, strict=True))
    # Comparing two partitioned stages alone can miss a shared discrepancy.
    # Compare directly with unwrapped Core on the same conditioned input.
    native = forward(native=True)
    if video_dtype == torch.bfloat16:
        assert torch.equal(native[0][:, :, prefix_t:], high[0][:, :, prefix_t:])
    else:
        # Native and Sol dense arithmetic differ by up to 7.45e-9 in this
        # FP32 output island with a long prefix. Do not demand bitwise equality
        # across those kernels or interpret this tolerance as a quality bound.
        torch.testing.assert_close(native[0][:, :, prefix_t:], high[0][:, :, prefix_t:], rtol=0, atol=1e-8)
    assert torch.equal(native[1], high[1])
    assert torch.count_nonzero(high[0][:, :, prefix_t:]) > 0
    assert torch.count_nonzero(high[0][:, :, :prefix_t]) == 0
    assert all(
        torch.equal(before, after) for before, after in zip(originals, (video, audio, mask, prefix, noise), strict=True)
    )
    assert guider.model_options["transformer_options"] == {}
