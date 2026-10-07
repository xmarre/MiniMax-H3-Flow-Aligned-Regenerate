"""Target-band domain-uniform low/probe execution through the actual Core forward.

The oracle compares each stream with an independent ordinary equal-grid
partitioned Core call on that stream's own clip. Equality establishes that the
two hidden streams share no rows, keys or conditioning state inside a model
call. Blocks use Sol's dense CPU reference; VDN routing is covered by VDN's own
domain-stream tests. Nothing here establishes rendered quality.
"""

from __future__ import annotations

import pytest
import torch

from h3_flow_regenerate import partitioned_transformer as transform
from h3_flow_regenerate.geometry import h3_patch_lattice_weight_square_sum, resize_spatial_5d_h3_patch_lattice
from h3_flow_regenerate.metrics import H3FlowMetrics
from h3_flow_regenerate.partitioned_band import pack_target_band_video, target_band_tail
from h3_flow_regenerate.partitioned_stage import (
    PARTITIONED_STAGE_KEY,
    TARGET_BAND_DOMAIN_UNIFORM_POLICY,
    PartitionedStagePlan,
    PartitionedStageRuntime,
    PartitionedTargetBandGeometry,
    TargetBandDomainContext,
)

PROTECTED_T, BAND_T, TEMPORAL = 2, 5, 12
TARGET_HW, SOURCE_HW = (8, 12), (4, 6)
# Native duration relation: 12 tokens span 39 frames (65 audio latents); the
# 7-token head spans 22 frames (37 audio latents).
AUDIO_T, HEAD_AUDIO_T = 65, 37
AUDIO_PROTECTED = 4
NOISE_SCALE = 1.3
TIMESTEP = 640.0


def _core(generator, layers=2):
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
        dtype=torch.float32,
        device="cpu",
        operations=comfy.ops.disable_weight_init,
    )
    dm.requires_grad_(False)
    with torch.no_grad():
        for param in dm.parameters():
            param.copy_(torch.randn(param.shape, generator=generator) * 0.05)
        dm.adaln_t_table.copy_(torch.randn(dm.adaln_t_table.shape, generator=generator))
        dm.rope.inv_freq.fill_(0.2)
    return dm


def _sol(monkeypatch):
    pytest.importorskip("sol_h3.runtime")
    from sol_h3 import partitioned_request

    monkeypatch.setattr(partitioned_request, "_validate_thd", lambda *_args: None)

    def cpu_key_bias(state, *, kv_rows, prefix_range, log_measure, device, semantic_digest):
        if prefix_range is None or float(log_measure) == 0.0:
            return None
        bias = torch.zeros(kv_rows, dtype=torch.float32, device=device)
        bias[prefix_range[0] : prefix_range[1]] = float(log_measure)
        return bias

    monkeypatch.setattr(partitioned_request, "_key_bias", cpu_key_bias)


def _forward(dm, wrappers, video, audio, context, options, mask, audio_mask):
    from comfy.patcher_extension import WrapperExecutor
    from sol_h3.contracts import Config
    from sol_h3.runtime import _FORWARD, _REQUEST, Request

    state = Request(Config(backend="sol"))
    request = _REQUEST.set(state)
    execution = _FORWARD.set((dm, state, 0, set(), []))
    try:
        with torch.no_grad():
            return WrapperExecutor.new_class_executor(dm._forward, dm, wrappers).execute(
                [video, audio],
                torch.tensor([TIMESTEP]),
                context,
                transformer_options=options,
                denoise_mask=mask,
                audio_denoise_mask=audio_mask,
            )
    finally:
        _FORWARD.reset(execution)
        _REQUEST.reset(request)


class _Case:
    def __init__(self, seed=31):
        generator = torch.Generator().manual_seed(seed)
        self.generator = generator
        self.dm = _core(generator)
        self.geometry = PartitionedTargetBandGeometry(
            protected_t=PROTECTED_T,
            band_t=BAND_T,
            temporal=TEMPORAL,
            source_h=SOURCE_HW[0],
            source_w=SOURCE_HW[1],
            target_h=TARGET_HW[0],
            target_w=TARGET_HW[1],
        )
        head = self.geometry.head_t
        self.prefix = torch.randn(1, 24, PROTECTED_T, *TARGET_HW, generator=generator)
        self.prefix_noise = torch.randn(self.prefix.shape, generator=generator)
        band = torch.randn(1, 24, BAND_T, *TARGET_HW, generator=generator)
        tail = torch.randn(1, 24, TEMPORAL - head, *SOURCE_HW, generator=generator)
        self.video = pack_target_band_video(torch.cat((self.prefix, band), dim=2), tail, self.geometry)
        self.mask = pack_target_band_video(
            torch.cat((torch.zeros(1, 1, PROTECTED_T, *TARGET_HW), torch.ones(1, 1, BAND_T, *TARGET_HW)), dim=2),
            torch.ones(1, 1, TEMPORAL - head, *SOURCE_HW),
            self.geometry,
        )
        self.audio = torch.randn(1, 32, 2, AUDIO_T, generator=generator)
        self.audio_mask = torch.ones(1, 1, 2, AUDIO_T)
        self.audio_mask[..., :AUDIO_PROTECTED] = 0
        self.context = torch.randn(1, 3, 8, generator=generator)
        self.owner = PartitionedStagePlan(self.prefix, TEMPORAL, *SOURCE_HW, self.prefix_noise)
        self.source_prefix = resize_spatial_5d_h3_patch_lattice(self.prefix, *SOURCE_HW)
        self.domain = TargetBandDomainContext(
            policy=TARGET_BAND_DOMAIN_UNIFORM_POLICY,
            source_prefix=self.source_prefix,
            source_prefix_noise=torch.randn(1, 24, PROTECTED_T, *SOURCE_HW, generator=generator),
            source_band_noise=torch.randn(1, 24, BAND_T, *SOURCE_HW, generator=generator),
            band_noise_complement=(1.0 - h3_patch_lattice_weight_square_sum(*TARGET_HW, *SOURCE_HW))
            .clamp_min(0)
            .sqrt(),
            model_noise_scale=NOISE_SCALE,
        )

    def runtime(self, metrics=None, domain=True):
        return PartitionedStageRuntime(
            plan=self.owner,
            metrics=metrics or H3FlowMetrics(),
            target_band=self.geometry,
            target_band_domain=self.domain if domain else None,
        )

    def domain_call(self, video=None, audio=None, metrics=None, extra_options=None):
        options = {PARTITIONED_STAGE_KEY: self.runtime(metrics), **(extra_options or {})}
        return _forward(
            self.dm,
            [transform.partitioned_diffusion_wrapper],
            self.video if video is None else video,
            self.audio if audio is None else audio,
            self.context,
            options,
            self.mask,
            self.audio_mask,
        )

    def carrier(self, video):
        head = self.geometry.head_t
        projected = resize_spatial_5d_h3_patch_lattice(video[:, :, PROTECTED_T:head], *SOURCE_HW)
        sigma = TIMESTEP / 1000.0
        return projected + sigma * NOISE_SCALE * self.domain.band_noise_complement * self.domain.source_band_noise

    def target_reference(self, video):
        head = self.geometry.head_t
        owner = PartitionedStagePlan(self.prefix, head, *TARGET_HW, self.prefix_noise)
        runtime = PartitionedStageRuntime(plan=owner, metrics=H3FlowMetrics())
        audio = self.audio[..., :HEAD_AUDIO_T]
        return _forward(
            self.dm,
            [transform.partitioned_diffusion_wrapper],
            video[:, :, :head].clone(),
            audio,
            self.context,
            {PARTITIONED_STAGE_KEY: runtime},
            self.mask[:, :, :head],
            self.audio_mask[..., :HEAD_AUDIO_T],
        )

    def source_reference(self, video, audio=None):
        owner = PartitionedStagePlan(self.source_prefix, TEMPORAL, *SOURCE_HW, self.domain.source_prefix_noise)
        runtime = PartitionedStageRuntime(plan=owner, metrics=H3FlowMetrics())
        source_video = torch.cat((self.source_prefix, self.carrier(video), target_band_tail(video, self.geometry)), 2)
        mask = torch.ones(1, 1, TEMPORAL, *SOURCE_HW)
        mask[:, :, :PROTECTED_T] = 0
        return _forward(
            self.dm,
            [transform.partitioned_diffusion_wrapper],
            source_video,
            self.audio if audio is None else audio,
            self.context,
            {PARTITIONED_STAGE_KEY: runtime},
            mask,
            self.audio_mask,
        )


def test_each_stream_equals_an_independent_uniform_core_call(monkeypatch):
    _sol(monkeypatch)
    case = _Case()
    head = case.geometry.head_t
    metrics = H3FlowMetrics()
    output = case.domain_call(metrics=metrics)
    target = case.target_reference(case.video)
    source = case.source_reference(case.video)
    torch.testing.assert_close(output[0][:, :, PROTECTED_T:head], target[0][:, :, PROTECTED_T:head], rtol=0, atol=1e-6)
    torch.testing.assert_close(target_band_tail(output[0], case.geometry), source[0][:, :, head:], rtol=0, atol=1e-6)
    torch.testing.assert_close(output[1], source[1], rtol=0, atol=1e-6)
    assert torch.count_nonzero(output[0][:, :, :PROTECTED_T]) == 0
    padding = output[0][:, :, head:].clone()
    padding[:, :, :, : SOURCE_HW[0], : SOURCE_HW[1]] = 0
    assert torch.count_nonzero(padding) == 0
    assert torch.count_nonzero(output[0][:, :, PROTECTED_T:head]) > 0
    blocks = len(case.dm.blocks)
    assert metrics.counters["partitioned_domain_uniform_calls"] == 1
    assert metrics.counters["partitioned_domain_uniform_target_block_calls"] == blocks
    assert metrics.counters["partitioned_domain_uniform_source_block_calls"] == blocks
    event = next(event.fields for event in metrics.events if event.kind == "partitioned_target_band_domain_transformer")
    assert event["shared_hidden_rows"] == 0 and event["cross_stream_attention_keys"] == 0
    streams = {stream["stream"]: stream for stream in event["streams"]}
    assert streams["target"]["temporal"] == head and streams["target"]["audio_rows"] == 2 * HEAD_AUDIO_T
    assert streams["source"]["temporal"] == TEMPORAL and streams["source"]["audio_rows"] == 2 * AUDIO_T
    assert streams["target"]["attention_head_t"] == head - 1
    assert streams["source"]["attention_head_t"] == head


def test_band_never_reads_the_tail_and_the_tail_reads_the_projected_band(monkeypatch):
    _sol(monkeypatch)
    case = _Case(seed=37)
    head = case.geometry.head_t
    baseline = case.domain_call()

    tail_changed = case.video.clone()
    tail_changed[:, :, -1, : SOURCE_HW[0], : SOURCE_HW[1]] += 1.0
    after_tail = case.domain_call(video=tail_changed)
    assert torch.equal(after_tail[0][:, :, PROTECTED_T:head], baseline[0][:, :, PROTECTED_T:head])
    assert not torch.equal(target_band_tail(after_tail[0], case.geometry), target_band_tail(baseline[0], case.geometry))

    band_changed = case.video.clone()
    band_changed[:, :, head - 1] += 1.0
    after_band = case.domain_call(video=band_changed)
    assert not torch.equal(target_band_tail(after_band[0], case.geometry), target_band_tail(baseline[0], case.geometry))

    # Audio after the head duration is visible only to the source stream.
    late_audio = case.audio.clone()
    late_audio[..., HEAD_AUDIO_T:] += 1.0
    after_audio = case.domain_call(audio=late_audio)
    assert torch.equal(after_audio[0][:, :, PROTECTED_T:head], baseline[0][:, :, PROTECTED_T:head])
    assert not torch.equal(after_audio[1], baseline[1])


def test_padding_and_prefix_carrier_never_reach_either_stream(monkeypatch):
    _sol(monkeypatch)
    case = _Case(seed=41)
    head = case.geometry.head_t
    baseline = case.domain_call()
    perturbed = case.video.clone()
    perturbed[:, :, :PROTECTED_T] = 5.0
    perturbed[:, :, head:, SOURCE_HW[0] :] = 7.0
    perturbed[:, :, head:, : SOURCE_HW[0], SOURCE_HW[1] :] = -7.0
    changed = case.domain_call(video=perturbed)
    assert all(torch.equal(a, b) for a, b in zip(baseline, changed, strict=True))


def test_domain_streams_publish_uniform_contracts_and_stream_views(monkeypatch):
    _sol(monkeypatch)
    from h3_flow_regenerate.partitioned_prefix import PARTITIONED_PREFIX_KEY
    from h3_flow_regenerate.partitioned_stage import TARGET_BAND_DOMAIN_STREAM_KEY

    case = _Case(seed=43)
    head = case.geometry.head_t
    seen = []
    native_options = transform._partitioned_transformer_options

    def capture(*args, **kwargs):
        result = native_options(*args, **kwargs)
        seen.append(
            (
                result[TARGET_BAND_DOMAIN_STREAM_KEY]["stream"],
                result[PARTITIONED_PREFIX_KEY],
                result[PARTITIONED_STAGE_KEY].attention_head_t,
                result["minimax_h3_layout"].signature,
            )
        )
        return result

    monkeypatch.setattr(transform, "_partitioned_transformer_options", capture)
    case.domain_call()
    assert [name for name, *_ in seen] == ["target", "source"] * len(case.dm.blocks)
    for name, contract, attention_head_t, signature in seen:
        assert contract["heterogeneous_spatial_domains"] is False
        assert contract["prefix_log_key_measure"] == 0.0
        assert contract["prefix_t"] == PROTECTED_T
        assert signature[-1] == (TARGET_BAND_DOMAIN_STREAM_KEY, TARGET_BAND_DOMAIN_UNIFORM_POLICY, name)
        if name == "target":
            assert contract["temporal"] == head and contract["target_rows_per_frame"] == case.geometry.target_rows
            assert attention_head_t == head - 1
        else:
            assert contract["temporal"] == TEMPORAL and contract["target_rows_per_frame"] == case.geometry.source_rows
            assert attention_head_t == head


def test_sol_history_recognizes_the_domain_replacement(monkeypatch):
    _sol(monkeypatch)
    interop = pytest.importorskip("sol_h3.interop")
    history = pytest.importorskip("sol_h3.partitioned_history")
    if getattr(history, "PARTITIONED_DOMAIN_STREAM_API", 0) != 1:
        pytest.skip("installed Sol-H3 predates domain-stream history recognition")
    case = _Case(seed=47)
    captured = {}

    class Capture:
        class_obj = case.dm

        def __call__(self, x, timestep, context, transformer_options, **kwargs):
            captured.update(transformer_options["patches_replace"]["dit"])
            return case.dm._forward(x, timestep, context, transformer_options, **kwargs)

    from sol_h3.contracts import Config
    from sol_h3.runtime import _FORWARD, _REQUEST, Request

    state = Request(Config(backend="sol"))
    request = _REQUEST.set(state)
    execution = _FORWARD.set((case.dm, state, 0, set(), []))
    try:
        with torch.no_grad():
            transform.partitioned_diffusion_wrapper(
                Capture(),
                [case.video, case.audio],
                torch.tensor([TIMESTEP]),
                case.context,
                transformer_options={PARTITIONED_STAGE_KEY: case.runtime()},
                denoise_mask=case.mask,
                audio_denoise_mask=case.audio_mask,
            )
    finally:
        _FORWARD.reset(execution)
        _REQUEST.reset(request)
    identities = []
    for (_kind, index), patch in captured.items():
        assert history._partitioned_flow_replacement_identity(interop, patch, index) is None
        resolved = history._partitioned_flow_domain_replacement_identity(interop, patch, index)
        assert resolved is not None
        identity, previous = resolved
        assert previous is None
        assert identity[0] == history.PARTITIONED_DOMAIN_UNIFORM_IDENTITY
        assert [stream[0] for stream in identity[5]] == ["target", "source"]
        history.install_partitioned_history_bridge()
        assert interop._flow_mixed_grid_replacement_identity(patch, index) == resolved
        identities.append(identity)
    assert len(set(identities)) == 1
    assert history._partitioned_flow_domain_replacement_identity(interop, captured[("double_block", 0)], 1) is None


def test_mismatched_audio_duration_fails_closed(monkeypatch):
    _sol(monkeypatch)
    case = _Case(seed=53)
    with pytest.raises(RuntimeError, match="native audio/video duration relation"):
        case.domain_call(audio=case.audio[..., :40])


def test_weight_square_sum_matches_the_explicit_resample_operator():
    generator = torch.Generator().manual_seed(3)
    for source, target in (((8, 12), (4, 6)), ((54, 72), (32, 44)), ((16, 16), (10, 6))):
        h, w = source
        basis = torch.eye(h * w).reshape(h * w, h, w)[None, :, None]
        operator = resize_spatial_5d_h3_patch_lattice(basis.permute(0, 1, 2, 3, 4), *target)[0, :, 0]
        expected = operator.square().sum(dim=0)
        torch.testing.assert_close(h3_patch_lattice_weight_square_sum(h, w, *target), expected, rtol=0, atol=1e-6)
    del generator
