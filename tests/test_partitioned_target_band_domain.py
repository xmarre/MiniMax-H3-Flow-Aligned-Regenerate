"""Target-band domain-uniform low/probe execution through the actual Core forward.

The oracle compares each stream with an independent ordinary equal-grid
partitioned Core call on that stream's own clip. Equality establishes that the
two hidden streams share no rows, keys or conditioning state inside a model
call. Blocks use Sol's dense CPU reference. VDN cases execute the real learned
readout with synthetic weights, including short convolution and recurrence.
Nothing here establishes rendered quality or production-checkpoint behavior.
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


def _forward(dm, wrappers, video, audio, context, options, mask, audio_mask, payload=None):
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
                minimax_payload=payload,
            )
    finally:
        _FORWARD.reset(execution)
        _REQUEST.reset(request)


class _Case:
    def __init__(self, seed=31, *, same_grid=False):
        generator = torch.Generator().manual_seed(seed)
        self.generator = generator
        self.dm = _core(generator)
        source_hw = TARGET_HW if same_grid else SOURCE_HW
        self.geometry = PartitionedTargetBandGeometry(
            protected_t=PROTECTED_T,
            band_t=BAND_T,
            temporal=TEMPORAL,
            source_h=source_hw[0],
            source_w=source_hw[1],
            target_h=TARGET_HW[0],
            target_w=TARGET_HW[1],
            same_grid_control=same_grid,
        )
        head = self.geometry.head_t
        self.prefix = torch.randn(1, 24, PROTECTED_T, *TARGET_HW, generator=generator)
        self.prefix_noise = torch.randn(self.prefix.shape, generator=generator)
        band = torch.randn(1, 24, BAND_T, *TARGET_HW, generator=generator)
        tail = torch.randn(1, 24, TEMPORAL - head, *source_hw, generator=generator)
        self.video = pack_target_band_video(torch.cat((self.prefix, band), dim=2), tail, self.geometry)
        self.mask = pack_target_band_video(
            torch.cat((torch.zeros(1, 1, PROTECTED_T, *TARGET_HW), torch.ones(1, 1, BAND_T, *TARGET_HW)), dim=2),
            torch.ones(1, 1, TEMPORAL - head, *source_hw),
            self.geometry,
        )
        self.audio = torch.randn(1, 32, 2, AUDIO_T, generator=generator)
        self.audio_mask = torch.ones(1, 1, 2, AUDIO_T)
        self.audio_mask[..., :AUDIO_PROTECTED] = 0
        self.context = torch.randn(1, 3, 8, generator=generator)
        self.payload = None
        self.wrappers = []
        self.owner = PartitionedStagePlan(self.prefix, TEMPORAL, *source_hw, self.prefix_noise)
        self.source_prefix = resize_spatial_5d_h3_patch_lattice(self.prefix, *source_hw)
        self.domain = TargetBandDomainContext(
            policy=TARGET_BAND_DOMAIN_UNIFORM_POLICY,
            source_prefix=self.source_prefix,
            source_prefix_noise=self.prefix_noise
            if same_grid
            else torch.randn(1, 24, PROTECTED_T, *source_hw, generator=generator),
            source_band_noise=torch.randn(1, 24, BAND_T, *source_hw, generator=generator),
            band_noise_complement=(1.0 - h3_patch_lattice_weight_square_sum(*TARGET_HW, *source_hw))
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
        if self.geometry.same_grid_control:
            options["h3_flow_stage"] = "high"
        return _forward(
            self.dm,
            [*self.wrappers, transform.partitioned_diffusion_wrapper],
            self.video if video is None else video,
            self.audio if audio is None else audio,
            self.context,
            options,
            self.mask,
            self.audio_mask,
            self.payload,
        )

    def carrier(self, video):
        head = self.geometry.head_t
        if self.geometry.same_grid_control:
            return video[:, :, PROTECTED_T:head]
        projected = resize_spatial_5d_h3_patch_lattice(
            video[:, :, PROTECTED_T:head], self.geometry.source_h, self.geometry.source_w
        )
        sigma = TIMESTEP / 1000.0
        return projected + sigma * NOISE_SCALE * self.domain.band_noise_complement * self.domain.source_band_noise

    def target_reference(self, video):
        head = self.geometry.head_t
        owner = PartitionedStagePlan(self.prefix, head, *TARGET_HW, self.prefix_noise)
        runtime = PartitionedStageRuntime(plan=owner, metrics=H3FlowMetrics())
        audio = self.audio[..., :HEAD_AUDIO_T]
        return _forward(
            self.dm,
            [*self.wrappers, transform.partitioned_diffusion_wrapper],
            video[:, :, :head].clone(),
            audio,
            self.context,
            {PARTITIONED_STAGE_KEY: runtime},
            self.mask[:, :, :head],
            self.audio_mask[..., :HEAD_AUDIO_T],
            self.payload,
        )

    def source_reference(self, video, audio=None):
        source_hw = (self.geometry.source_h, self.geometry.source_w)
        owner = PartitionedStagePlan(self.source_prefix, TEMPORAL, *source_hw, self.domain.source_prefix_noise)
        runtime = PartitionedStageRuntime(plan=owner, metrics=H3FlowMetrics())
        source_video = torch.cat((self.source_prefix, self.carrier(video), target_band_tail(video, self.geometry)), 2)
        mask = torch.ones(1, 1, TEMPORAL, *source_hw)
        mask[:, :, :PROTECTED_T] = 0
        return _forward(
            self.dm,
            [*self.wrappers, transform.partitioned_diffusion_wrapper],
            source_video,
            self.audio if audio is None else audio,
            self.context,
            {PARTITIONED_STAGE_KEY: runtime},
            mask,
            self.audio_mask,
            self.payload,
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


@pytest.mark.parametrize("band_t", [4, 5])
def test_high_domain_preserves_the_short_clip_operator_and_excludes_tail_and_late_audio(monkeypatch, band_t):
    import sys

    cli = pytest.importorskip("comfy.cli_args")
    cli.args.cpu = True
    from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE

    monkeypatch.setattr(sys.modules[__name__], "BAND_T", band_t)
    head_audio_t = round(sum(FRAME_PER_TOKEN[k % 5] for k in range(PROTECTED_T + band_t)) * FRAME_RESCALE)
    monkeypatch.setattr(sys.modules[__name__], "HEAD_AUDIO_T", head_audio_t)
    _sol(monkeypatch)
    case = _Case(same_grid=True)
    head = case.geometry.head_t
    metrics = H3FlowMetrics()
    output = case.domain_call(metrics=metrics)
    target = case.target_reference(case.video)
    source = case.source_reference(case.video)
    torch.testing.assert_close(output[0][:, :, PROTECTED_T:head], target[0][:, :, PROTECTED_T:head], rtol=0, atol=1e-6)
    torch.testing.assert_close(output[0][:, :, head:], source[0][:, :, head:], rtol=0, atol=1e-6)
    torch.testing.assert_close(output[1], source[1], rtol=0, atol=1e-6)
    changed = case.video.clone()
    changed[:, :, head:] += 2.0
    late_audio = case.audio.clone()
    late_audio[..., HEAD_AUDIO_T:] += 2.0
    for video, audio in ((changed, case.audio), (case.video, late_audio)):
        isolated = case.domain_call(video=video, audio=audio)
        assert torch.equal(output[0][:, :, PROTECTED_T:head], isolated[0][:, :, PROTECTED_T:head])
        full = case.source_reference(video, audio)
        # The ordinary full-clip high operator does read both inputs in the band.
        assert not torch.equal(source[0][:, :, PROTECTED_T:head], full[0][:, :, PROTECTED_T:head])
        delta = full[0][:, :, PROTECTED_T:head] - source[0][:, :, PROTECTED_T:head]
        assert torch.count_nonzero(delta.mean(dim=(-2, -1))) > 0
    event = next(event.fields for event in metrics.events if event.kind == "partitioned_target_band_domain_transformer")
    assert event["stage"] == "high"
    assert event["source_hw"] == event["target_hw"] == TARGET_HW
    assert event["band_carrier_policy"] == "native_target_sampler_state_v1"


def test_high_domain_failure_restores_the_model_options(monkeypatch):
    from types import SimpleNamespace

    from h3_flow_regenerate.partitioned_scheduler import _partitioned_high_stage_contract

    case = _Case()
    options = {"h3_flow_stage": "high", "h3_flow_partitioned_vdn_linear_diagnostic_v1": "raw_token_measure"}
    original = dict(options)
    guider = SimpleNamespace(model_options={"transformer_options": options})
    with (
        pytest.raises(RuntimeError, match="high failed"),
        _partitioned_high_stage_contract(
            guider, case.owner, H3FlowMetrics(), attention_head_t=case.geometry.head_t, domain_uniform_band_t=BAND_T
        ),
    ):
        runtime = options[PARTITIONED_STAGE_KEY]
        assert runtime.target_band.same_grid_control is True
        assert runtime.target_band_domain is not None
        raise RuntimeError("high failed")
    assert options == original
    assert (case.owner.source_h, case.owner.source_w) == SOURCE_HW


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


@pytest.mark.parametrize("same_grid", [False, True])
def test_sol_history_recognizes_the_domain_replacement(monkeypatch, same_grid):
    _sol(monkeypatch)
    interop = pytest.importorskip("sol_h3.interop")
    history = pytest.importorskip("sol_h3.partitioned_history")
    if getattr(history, "PARTITIONED_DOMAIN_STREAM_API", 0) != 1:
        pytest.skip("installed Sol-H3 predates domain-stream history recognition")
    case = _Case(seed=47, same_grid=same_grid)
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


@pytest.mark.parametrize("kind", ["image", "audio", "video_audio"])
def test_native_references_keep_each_streams_own_coordinates(monkeypatch, kind):
    _sol(monkeypatch)
    case = _Case(seed=61)
    ref_video = torch.randn(1, 24, 2 if kind == "video_audio" else 1, 4, 6, generator=case.generator)
    ref_audio = torch.randn(1, 32, 2, 3, generator=case.generator)
    ref = {"kind": kind, "latent_h": 4, "latent_w": 6, "latent_t": int(ref_video.shape[2]), "ref_audio_t": 3}
    case.payload = {
        "refs": [ref],
        "cond_video_latents": [ref_video] if kind != "audio" else [],
        "cond_audio_latents": [ref_audio] if kind != "image" else [],
    }
    output = case.domain_call()
    target = case.target_reference(case.video)
    source = case.source_reference(case.video)
    head = case.geometry.head_t
    torch.testing.assert_close(output[0][:, :, PROTECTED_T:head], target[0][:, :, PROTECTED_T:head], rtol=0, atol=1e-6)
    torch.testing.assert_close(target_band_tail(output[0], case.geometry), source[0][:, :, head:], rtol=0, atol=1e-6)
    torch.testing.assert_close(output[1], source[1], rtol=0, atol=1e-6)


def test_weight_square_sum_matches_the_explicit_resample_operator():
    generator = torch.Generator().manual_seed(3)
    for source, target in (((8, 12), (4, 6)), ((54, 72), (32, 44)), ((16, 16), (10, 6))):
        h, w = source
        basis = torch.eye(h * w).reshape(h * w, h, w)[None, :, None]
        operator = resize_spatial_5d_h3_patch_lattice(basis.permute(0, 1, 2, 3, 4), *target)[0, :, 0]
        expected = operator.square().sum(dim=0)
        torch.testing.assert_close(h3_patch_lattice_weight_square_sum(h, w, *target), expected, rtol=0, atol=1e-6)
    del generator


@pytest.mark.parametrize("prepared", [False, True])
def test_domain_keyframes_are_rejected_at_preflight(prepared):
    from types import SimpleNamespace

    pytest.importorskip("comfy.ldm.minimax.model")
    from h3_flow_regenerate.partitioned_scheduler import (
        PartitionedPreflightUnsupported,
        _validate_target_band_domain_inputs,
    )

    keyframes = [{"resolved_frame_index": 0}]
    entry = (
        {"model_conds": {"minimax_payload": SimpleNamespace(cond={"keyframes": keyframes})}}
        if prepared
        else {"minimax_keyframes": keyframes}
    )
    guider = SimpleNamespace(conds={"positive": [{}], "negative": [entry]})
    shapes = [(1, 24, TEMPORAL, *TARGET_HW), (1, 32, 2, AUDIO_T)]
    with pytest.raises(PartitionedPreflightUnsupported, match="keyframe-anchored"):
        _validate_target_band_domain_inputs(guider, shapes)


def test_domain_audio_duration_is_rejected_at_preflight():
    from types import SimpleNamespace

    pytest.importorskip("comfy.ldm.minimax.model")
    from h3_flow_regenerate.partitioned_scheduler import (
        PartitionedPreflightUnsupported,
        _validate_target_band_domain_inputs,
    )

    guider = SimpleNamespace(conds={"positive": [{}]})
    shapes = [(1, 24, TEMPORAL, *TARGET_HW), (1, 32, 2, 40)]
    with pytest.raises(PartitionedPreflightUnsupported, match="native audio/video duration relation"):
        _validate_target_band_domain_inputs(guider, shapes)


@pytest.mark.parametrize("retain", [False, True])
@pytest.mark.parametrize("anchors", ["rows", "both"])
@pytest.mark.parametrize("same_grid", [False, True])
def test_real_vdn_readout_keeps_uniform_stream_outputs_and_isolation(monkeypatch, retain, anchors, same_grid):
    """Execute Core, Sol dense arithmetic and VDN's learned branch without substituting its readout."""
    _sol(monkeypatch)
    case = _Case(seed=67, same_grid=same_grid)
    pytest.importorskip("vdn_h3.partitioned_runtime")
    from vdn_h3 import branch as branch_math
    from vdn_h3.hybrid import VDNState, make_layout_wrapper
    from vdn_h3.partitioned_runtime import _wrap_vdn_forward
    from vdn_h3.retained import RuntimeLinearBranch

    # CPU runs the same arithmetic eagerly; only compilation is bypassed.
    monkeypatch.setattr(branch_math, "_run_compiled", lambda _key, body, *args, **kwargs: body(*args, **kwargs))
    cfg = {"radius": 1, "chunk": 1, "anchor_frames": anchors, "enable_softmax_gate": True, "linear_enabled": True}
    branches = []
    hidden, head_dim = int(case.dm.hidden_size), int(case.dm.blocks[0].attn.head_dim)
    for _block in case.dm.blocks:

        def rand(*shape):
            return torch.randn(shape, generator=case.generator) * 0.05

        weights = {
            "beta_proj.weight": rand(1, hidden),
            "alpha.down.weight": rand(2, hidden),
            "alpha.up.weight": rand(head_dim, 2),
            "alpha.dt_bias": rand(head_dim),
            "alpha.A_log": rand(1),
            "output_gate.down.weight": rand(2, hidden),
            "output_gate.up.weight": rand(head_dim, 2),
            "output_gate.up.bias": rand(head_dim),
            "norm.weight": torch.ones(head_dim),
            "to_out_linear.weight": rand(hidden, head_dim),
            "softmax_gate.up.weight": rand(1, hidden),
            "softmax_gate.up.bias": rand(1),
        }
        for name in ("k", "v"):
            weights[f"short_conv.{name}_sp.weight"] = rand(head_dim, 1, 5, 5)
            weights[f"short_conv.{name}_tm.weight"] = rand(head_dim, 1, 5)
        branches.append(RuntimeLinearBranch(weights, 1, head_dim, enable_text_state=True))
    state = VDNState("domain-core-readout-test", cfg, branches, 1, head_dim, retain_buffers=retain)
    case.wrappers = [make_layout_wrapper(state)]

    def install(attention, block_index):
        base_branch = branches[block_index]
        heads = attention.heads
        head_dim = attention.head_dim
        qkv_proj, out_proj = attention.qkv_proj, attention.out_proj
        q_norm, k_norm = attention.q_norm, attention.k_norm
        original = attention.forward

        def current(x, rope_freqs=None, transformer_options=None):
            _ = (base_branch, block_index, cfg, head_dim, heads, k_norm, out_proj, q_norm, qkv_proj, state)
            return original(x, rope_freqs=rope_freqs, transformer_options=transformer_options)

        current._vdn_forward = True
        monkeypatch.setattr(attention, "forward", _wrap_vdn_forward(current))

    for index, block in enumerate(case.dm.blocks):
        install(block.attn, index)
    if same_grid:
        # Reuse the model and VDN owner across reduced low/probe streams and
        # native high streams, including retained workspace reallocations.
        low = _Case(seed=73)
        low.dm, low.wrappers = case.dm, case.wrappers
        low.domain_call()
    metrics = H3FlowMetrics()
    output = case.domain_call(metrics=metrics)
    target = case.target_reference(case.video)
    source = case.source_reference(case.video)
    head = case.geometry.head_t
    torch.testing.assert_close(output[0][:, :, PROTECTED_T:head], target[0][:, :, PROTECTED_T:head], rtol=0, atol=1e-6)
    torch.testing.assert_close(target_band_tail(output[0], case.geometry), source[0][:, :, head:], rtol=0, atol=1e-6)
    torch.testing.assert_close(output[1], source[1], rtol=0, atol=1e-6)
    assert metrics.counters["partitioned_vdn_uniform_linear_calls"] == 2 * len(case.dm.blocks)
    assert metrics.counters["partitioned_vdn_domain_stream_target_calls"] == len(case.dm.blocks)
    assert metrics.counters["partitioned_vdn_domain_stream_source_calls"] == len(case.dm.blocks)
    changed = case.video.clone()
    changed[:, :, -1, : SOURCE_HW[0], : SOURCE_HW[1]] += 1.0
    after_tail = case.domain_call(video=changed)
    assert torch.equal(output[0][:, :, PROTECTED_T:head], after_tail[0][:, :, PROTECTED_T:head])
    if same_grid:
        low.domain_call()
        repeated = case.domain_call()
        assert torch.equal(output[0], repeated[0])
        assert torch.equal(output[1], repeated[1])
    cfg["linear_enabled"] = False
    without_linear = case.domain_call()
    assert not torch.equal(output[0][:, :, PROTECTED_T:head], without_linear[0][:, :, PROTECTED_T:head])
