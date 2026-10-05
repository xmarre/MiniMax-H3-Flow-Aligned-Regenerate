"""Target-band continuation through the actual Core MiniMax-H3 forward."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate import partitioned_transformer as transform
from h3_flow_regenerate.metrics import H3FlowMetrics
from h3_flow_regenerate.partitioned_band import pack_target_band_video, target_band_tail
from h3_flow_regenerate.partitioned_stage import (
    PARTITIONED_STAGE_KEY,
    PartitionedStagePlan,
    PartitionedStageRuntime,
    PartitionedTargetBandGeometry,
)

PROTECTED_T, BAND_T, TEMPORAL = 2, 2, 7
TARGET_HW, SOURCE_HW = (8, 12), (4, 6)


def _core(*, layers, generator, dtype=torch.float32):
    cli = pytest.importorskip("comfy.cli_args")
    cli.args.cpu = True
    import comfy.ops
    from comfy.ldm.minimax.model import MiniMaxH3Model

    dm = MiniMaxH3Model(
        hidden_size=8,
        num_layers=layers,
        token_refiner_num_layers=0,
        num_attention_heads=1,
        attention_head_dim=128,
        ffn_hidden_size=16,
        text_dim=8,
        time_embed_dim=2,
        adaln_curve_grid=4,
        rope_inv_freq_len=2,
        dtype=dtype,
        device="cpu",
        operations=comfy.ops.disable_weight_init,
    )
    dm.requires_grad_(False)
    with torch.no_grad():
        for param in dm.parameters():
            param.copy_(torch.randn(param.shape, generator=generator) * 0.02)
        dm.adaln_t_table.copy_(torch.randn(dm.adaln_t_table.shape, generator=generator))
        dm.rope.inv_freq.fill_(0.2)
    return dm


def _band_inputs(generator):
    geometry = PartitionedTargetBandGeometry(
        protected_t=PROTECTED_T,
        band_t=BAND_T,
        temporal=TEMPORAL,
        source_h=SOURCE_HW[0],
        source_w=SOURCE_HW[1],
        target_h=TARGET_HW[0],
        target_w=TARGET_HW[1],
    )
    prefix = torch.randn(1, 24, PROTECTED_T, *TARGET_HW, generator=generator)
    prefix_noise = torch.randn(prefix.shape, generator=generator)
    band = torch.randn(1, 24, BAND_T, *TARGET_HW, generator=generator)
    tail = torch.randn(1, 24, TEMPORAL - geometry.head_t, *SOURCE_HW, generator=generator)
    video = pack_target_band_video(torch.cat((prefix, band), dim=2), tail, geometry)
    mask = pack_target_band_video(
        torch.cat((torch.zeros(1, 1, PROTECTED_T, *TARGET_HW), torch.ones(1, 1, BAND_T, *TARGET_HW)), dim=2),
        torch.ones(1, 1, TEMPORAL - geometry.head_t, *SOURCE_HW),
        geometry,
    )
    owner = PartitionedStagePlan(prefix, TEMPORAL, *SOURCE_HW, prefix_noise)
    return geometry, owner, prefix, prefix_noise, band, tail, video, mask


def _execute(dm, wrappers, video, audio, context, timestep, options, mask):
    from comfy.patcher_extension import WrapperExecutor

    with torch.no_grad():
        return WrapperExecutor.new_class_executor(dm._forward, dm, wrappers).execute(
            [video, audio],
            torch.tensor([timestep]),
            context,
            transformer_options=options,
            denoise_mask=mask,
        )


def test_band_rows_round_trip_through_core_rows_and_final_layer():
    """With row-local blocks, band and tail outputs equal unwrapped Core on the same rows."""
    generator = torch.Generator().manual_seed(17)
    dm = _core(layers=3, generator=generator)
    for block in dm.blocks:
        block.forward = lambda img, *_args, **_kw: img.clone()
    geometry, owner, *_rest, video, mask = _band_inputs(generator)
    audio = torch.randn(1, 32, 2, 9, generator=generator)
    context = torch.randn(1, 3, 8, generator=generator)
    metrics = H3FlowMetrics()
    runtime = PartitionedStageRuntime(
        plan=owner,
        metrics=metrics,
        audio_position_domain="source_carrier",
        target_band=geometry,
    )
    band = _execute(
        dm,
        [transform.partitioned_diffusion_wrapper],
        video,
        audio,
        context,
        750.0,
        {PARTITIONED_STAGE_KEY: runtime},
        mask,
    )
    native = _execute(dm, [], video, audio, context, 750.0, {}, mask)
    head = geometry.head_t
    assert band[0].shape == video.shape
    assert torch.equal(band[0][:, :, PROTECTED_T:head], native[0][:, :, PROTECTED_T:head])
    assert torch.equal(target_band_tail(band[0], geometry), target_band_tail(native[0], geometry))
    assert torch.equal(band[1], native[1])
    assert torch.count_nonzero(band[0][:, :, :PROTECTED_T]) == 0
    padding = band[0][:, :, head:].clone()
    padding[:, :, :, : SOURCE_HW[0], : SOURCE_HW[1]] = 0
    assert torch.count_nonzero(padding) == 0
    assert torch.count_nonzero(band[0][:, :, PROTECTED_T:]) > 0
    events = [event for event in metrics.events if event.kind == "partitioned_exact_prefix_transformer"]
    assert events and events[0].fields["native_carrier_grid"] == "target"
    assert events[0].fields["target_band_t"] == BAND_T
    assert events[0].fields["prefix_t"] == geometry.head_t
    assert events[0].fields["protected_prefix_t"] == PROTECTED_T
    assert events[0].fields["partitioned_sequence_rows"] - events[0].fields["video_start"] == geometry.partitioned_rows


def _sol_contexts(monkeypatch):
    pytest.importorskip("sol_h3.runtime")
    from sol_h3 import partitioned_request

    # CPU arithmetic harness: Core blocks and Sol's dense reference arithmetic run
    # unchanged. Sol's device admission, CUDA-stream bias cache and SM120 weighted
    # kernel are covered by its GPU contracts; the substitutes carry the identical
    # key measure and dense reference.
    monkeypatch.setattr(partitioned_request, "_validate_thd", lambda *_args: None)

    def cpu_key_bias(state, *, kv_rows, prefix_range, log_measure, device, semantic_digest):
        if prefix_range is None or float(log_measure) == 0.0:
            return None
        start, end = prefix_range
        bias = torch.zeros(kv_rows, dtype=torch.float32, device=device)
        bias[start:end] = float(log_measure)
        return bias

    monkeypatch.setattr(partitioned_request, "_key_bias", cpu_key_bias)
    # The weighted (non-unit measure) dense route executes Sol's SM120 union and
    # gates it against its own dense reference; run that reference on CPU.
    monkeypatch.setattr(
        partitioned_request,
        "_checked_weighted_dense",
        lambda q, k, v, key_bias, *, scale, **_kw: partitioned_request._weighted_dense(q, k, v, key_bias, scale=scale),
    )


def _sol_forward(dm, wrappers, video, audio, context, timestep, options, mask):
    from sol_h3.contracts import Config
    from sol_h3.runtime import _FORWARD, _REQUEST, Request

    state = Request(Config(backend="sol"))
    request = _REQUEST.set(state)
    execution = _FORWARD.set((dm, state, 0, set(), []))
    try:
        output = _execute(dm, wrappers, video, audio, context, timestep, options, mask)
        return output, state
    finally:
        _FORWARD.reset(execution)
        _REQUEST.reset(request)


def test_band_rope_rows_equal_the_progressive_partition_with_the_same_head():
    """Head rows use the target grid and tail rows the reduced grid on one global timeline."""
    cli = pytest.importorskip("comfy.cli_args")
    cli.args.cpu = True
    import comfy.ldm.minimax.model as native

    from h3_flow_regenerate.partitioned_stage import partitioned_carrier_layout, partitioned_positions

    generator = torch.Generator().manual_seed(23)
    geometry, *_rest = _band_inputs(generator)
    head = geometry.head_t
    legacy_owner = PartitionedStagePlan(
        torch.zeros(1, 24, head, *TARGET_HW), TEMPORAL, *SOURCE_HW, torch.zeros(1, 24, head, *TARGET_HW)
    )
    payload = {}
    legacy_layout = partitioned_carrier_layout(native, legacy_owner, 3, 9, payload)
    band_layout = native.PackedLayout(3, TEMPORAL, *TARGET_HW, 9)
    legacy = partitioned_positions(native, legacy_owner, legacy_layout)
    band = partitioned_positions(native, geometry, band_layout)
    assert torch.equal(band, legacy)
    assert band.shape[0] - band_layout.segments[-1][0] == geometry.partitioned_rows


def test_band_forward_through_real_blocks_ignores_padding_and_prefix_carrier(monkeypatch):
    """Real Core blocks with Sol dense attention over [prefix | band | tail].

    Padding and the protected prefix's sampler carrier are never presented to
    the transformer, so changing them must not change any output.
    """
    _sol_contexts(monkeypatch)
    from h3_flow_regenerate.partitioned_scheduler import _partitioned_stage_contract
    from h3_flow_regenerate.runtime import _flow_stage_contract

    generator = torch.Generator().manual_seed(29)
    # FP32 keeps the small cross-row attention effects of these tiny random
    # weights above rounding, so the dependence check below is meaningful.
    dm = _core(layers=2, generator=generator)
    geometry, owner, *_rest, video, mask = _band_inputs(generator)
    audio = torch.randn(1, 32, 2, 9, generator=generator)
    context = torch.randn(1, 3, 8, generator=generator)
    head = geometry.head_t

    def forward(state):
        guider = SimpleNamespace(model_options={"transformer_options": {}})
        metrics = H3FlowMetrics()
        with (
            _flow_stage_contract(guider, "low"),
            _partitioned_stage_contract(guider, owner, metrics, target_band=geometry),
        ):
            output, sol_state = _sol_forward(
                dm,
                [transform.partitioned_diffusion_wrapper],
                state,
                audio,
                context,
                878.04878,
                guider.model_options["transformer_options"],
                mask,
            )
        assert sol_state.dense_calls == len(dm.blocks)
        assert guider.model_options["transformer_options"] == {}
        return output

    baseline = forward(video)
    perturbed = video.clone()
    perturbed[:, :, :PROTECTED_T] = torch.randn(perturbed[:, :, :PROTECTED_T].shape, generator=generator).to(perturbed)
    perturbed[:, :, head:, SOURCE_HW[0] :] = 7.0
    perturbed[:, :, head:, : SOURCE_HW[0], SOURCE_HW[1] :] = -7.0
    changed = forward(perturbed)
    assert all(torch.equal(a, b) for a, b in zip(baseline, changed, strict=True))

    assert bool(torch.isfinite(baseline[0]).all())
    assert torch.count_nonzero(baseline[0][:, :, :PROTECTED_T]) == 0
    assert torch.count_nonzero(baseline[0][:, :, PROTECTED_T:head]) > 0
    assert torch.count_nonzero(target_band_tail(baseline[0], geometry)) > 0
    padding = baseline[0][:, :, head:].clone()
    padding[:, :, :, : SOURCE_HW[0], : SOURCE_HW[1]] = 0
    assert torch.count_nonzero(padding) == 0

    # Band rows do read the generated tail: changing the tail changes the band.
    tail_changed = video.clone()
    tail_changed[:, :, -1, : SOURCE_HW[0], : SOURCE_HW[1]] += 1.0
    other = forward(tail_changed)
    assert not torch.equal(other[0][:, :, PROTECTED_T:head], baseline[0][:, :, PROTECTED_T:head])


def test_sol_history_recognizes_the_actual_band_replacement_closure():
    cli = pytest.importorskip("comfy.cli_args")
    cli.args.cpu = True
    interop = pytest.importorskip("sol_h3.interop")
    history = pytest.importorskip("sol_h3.partitioned_history")
    if "target" not in getattr(history, "PARTITIONED_NATIVE_CARRIER_GRIDS", ()):
        pytest.skip("installed Sol-H3 predates target native carrier recognition")

    generator = torch.Generator().manual_seed(41)
    dm = _core(layers=2, generator=generator)
    for block in dm.blocks:
        block.forward = lambda img, *_args, **_kw: img.clone()
    geometry, owner, *_rest, video, mask = _band_inputs(generator)
    audio = torch.randn(1, 32, 2, 9, generator=generator)
    context = torch.randn(1, 3, 8, generator=generator)
    runtime = PartitionedStageRuntime(plan=owner, metrics=H3FlowMetrics(), target_band=geometry)
    captured = {}

    class Capture:
        class_obj = dm

        def __call__(self, x, timestep, context, transformer_options, **kwargs):
            captured.update(transformer_options["patches_replace"]["dit"])
            return dm._forward(x, timestep, context, transformer_options, **kwargs)

    with torch.no_grad():
        transform.partitioned_diffusion_wrapper(
            Capture(),
            [video, audio],
            torch.tensor([750.0]),
            context,
            transformer_options={PARTITIONED_STAGE_KEY: runtime},
            denoise_mask=mask,
        )
    assert len(captured) == len(dm.blocks)
    for (_kind, index), patch in captured.items():
        resolved = history._partitioned_flow_replacement_identity(interop, patch, index)
        assert resolved is not None
        identity, _previous = resolved
        # native rows, partitioned rows, video start, temporal, head, source/target rows, target hw
        assert identity[5:10] == (TEMPORAL, geometry.head_t, geometry.source_rows, geometry.target_rows, TARGET_HW)
        assert identity[2] - identity[4] == geometry.native_rows
        assert identity[3] - identity[4] == geometry.partitioned_rows
