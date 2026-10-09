"""Partitioned exact-prefix transformer runtime.

This module owns the new production-shaped heterogeneous attention contract.  It
does not consume or publish the retired Mixed-Grid contract: target-grid exact
prefix rows and source-grid generated suffix rows are carried as explicit physical
domains, and the exact physical key-measure correction is transported directly
to the Sol/VDN partitioned backend.
"""

from __future__ import annotations

import contextlib
import copy
import inspect
from dataclasses import dataclass

import torch

from .geometry import resize_spatial_5d_h3_patch_lattice
from .partitioned_attention import (
    PARTITIONED_ATTENTION_PROVIDER_KEY,
    call_partitioned_attention,
    partitioned_block_extra,
    sol_attention_selected,
)
from .partitioned_diagnostics import (
    PARTITIONED_AUDIO_MODEL_TIMESTEP_CONTEXT_KEY,
    PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE,
    PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT,
    PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE,
    PARTITIONED_SOFTMAX_DIAGNOSTIC_KEY,
    PARTITIONED_SOFTMAX_DIAGNOSTIC_NORMAL,
    PARTITIONED_VDN_LINEAR_DIAGNOSTIC_KEY,
    PARTITIONED_VDN_TEMPORAL_CARRIER_DESTINATION,
    PARTITIONED_VDN_TEMPORAL_CARRIER_KEY,
    PARTITIONED_VDN_TEMPORAL_CARRIER_NATIVE,
    PartitionedAudioModelTimestepContext,
    build_vdn_temporal_carrier_contract,
    normalize_vdn_linear_diagnostic,
)
from .partitioned_prefix import (
    PARTITIONED_NATIVE_CARRIER_SOURCE,
    PARTITIONED_NATIVE_CARRIER_TARGET,
    PARTITIONED_PREFIX_KEY,
    PARTITIONED_PREFIX_TOPOLOGY,
    PartitionedExactPrefixPlan,
    validate_partitioned_contract,
)
from .partitioned_stage import (
    PARTITIONED_STAGE_KEY,
    TARGET_BAND_DOMAIN_BAND_CARRIER_POLICY,
    TARGET_BAND_DOMAIN_BAND_CONDITIONING_POLICY,
    TARGET_BAND_DOMAIN_STREAM_API,
    TARGET_BAND_DOMAIN_STREAM_KEY,
    PartitionedStagePlan,
    PartitionedStageRuntime,
    PartitionedStageStreamView,
    PartitionedTargetBandGeometry,
    TargetBandDomainContext,
    partitioned_carrier_layout,
    partitioned_mod_segments,
    partitioned_positions_for_runtime,
)
from .uniform_prefix_context import POLICY, add_exact_prefix_visual_context

PARTITIONED_WRAPPER_KEY = "h3_flow_regenerate.partitioned_exact_prefix.v1"
PARTITIONED_BLOCK_INDEX_KEY = "h3_flow_partitioned_block_index_v1"
VDN_EXTERNAL_SEQUENCE_KEY = "vdn_h3_external_sequence_v1"
VDN_PARTITIONED_SEQUENCE_API = 4
VDN_PARTITIONED_SEQUENCE_MODE = "partitioned_attention_variable_grid_linear"
_PREPROCESS_ATTR = "attention_preprocess_v1"
_DEPRECATED_MIXED_GRID_KEYS = (
    "h3_flow_mixed_grid_v1",
    "h3_flow_mixed_grid_attention_measure_v1",
)


def _vdn_external_contract(plan: PartitionedExactPrefixPlan) -> dict:
    flow = plan.to_contract()
    contract = {
        "api": VDN_PARTITIONED_SEQUENCE_API,
        "mode": VDN_PARTITIONED_SEQUENCE_MODE,
        "topology": PARTITIONED_PREFIX_TOPOLOGY,
        "sequence_rows": plan.sequence_rows,
        "video_start": plan.video_start,
        "temporal": plan.temporal,
        "prefix_t": plan.prefix_t,
        "source_rows_per_frame": plan.source_rows,
        "target_rows_per_frame": plan.target_rows,
        "flow_semantic_digest": flow["semantic_digest"],
    }
    if plan.native_carrier_grid != PARTITIONED_NATIVE_CARRIER_SOURCE:
        contract["native_carrier_rows_per_frame"] = plan.native_rows_per_frame
    return contract


def _preprocess_chain(previous, *, native_attention=False, seen=None):
    transforms = []
    provider = previous
    seen = set() if seen is None else seen
    while provider is not None and hasattr(provider, _PREPROCESS_ATTR):
        if id(provider) in seen:
            raise RuntimeError("partitioned attention inherited a cyclic preprocessing chain")
        seen.add(id(provider))
        entry = getattr(provider, _PREPROCESS_ATTR)
        if not isinstance(entry, tuple) or len(entry) != 2 or not callable(entry[0]):
            raise RuntimeError("partitioned attention inherited malformed preprocessing metadata")
        transform, provider = entry
        transforms.append(transform)
    if (
        native_attention
        and getattr(provider, "__module__", None) == "comfy_extras.nodes_sparse_attention"
        and getattr(provider, "__qualname__", None) == "make_attention_override.<locals>.override"
    ):
        if id(provider) in seen:
            raise RuntimeError("partitioned attention inherited a cyclic Core BSA chain")
        seen.add(id(provider))
        captured = inspect.getclosurevars(provider).nonlocals
        factory = provider.__globals__.get("make_attention_override")
        if "previous" not in captured or "patch" not in captured or not callable(factory):
            raise RuntimeError("partitioned attention cannot resolve Core BSA preprocessing ownership")
        inner_transforms, terminal = _preprocess_chain(captured["previous"], native_attention=True, seen=seen)
        if inner_transforms:
            # Core's backend wrapper can hide full-domain preprocessing below
            # it. Move that preprocessing before VDN gathers, then rebuild only
            # the backend dispatch around the stripped leaf. Numerical identity
            # remains bound to the original Core owner, not this new closure.
            original = provider
            provider = factory(captured["patch"], terminal)
            provider._h3_flow_native_provider_identity = (_provider_name(original), id(original))
            transforms.extend(inner_transforms)
    return tuple(transforms), provider


def _call_provider(provider, original, q, k, v, heads, mask, kw):
    if provider is None:
        return original(q, k, v, heads, mask=mask, **kw)
    return provider(original, q, k, v, heads, mask=mask, **kw)


def _provider_name(provider):
    """Mirror the Sol history-v1 provider implementation naming contract."""
    if provider is None:
        return "comfy.default"
    module = getattr(provider, "__module__", "<unknown>")
    qualname = getattr(provider, "__qualname__", type(provider).__name__)
    return f"{module}.{qualname}"


def _inherited_provider_state(previous, *, native_attention=False):
    """Resolve the numerical identity that Sol history-v1 already considers stable.

    Generic attention_preprocess_v1 wrappers are allowed to be rebuilt around
    the same terminal dense provider. Sol deliberately identifies those transforms
    by implementation name and the terminal provider by name + object identity.
    Partitioned Flow must use that same boundary when deciding whether its own
    visible provider object may remain stable; raw outer-wrapper identity is too
    strict and caused the 00500/00503/00506/00507 provider transitions.
    """
    transforms, terminal = _preprocess_chain(previous, native_attention=native_attention)
    terminal_identity = getattr(terminal, "_h3_flow_native_provider_identity", (_provider_name(terminal), id(terminal)))
    identity = (
        tuple(_provider_name(transform) for transform in transforms),
        *terminal_identity,
    )
    return identity, transforms, terminal


def _runtime_provider_state(runtime: PartitionedStageRuntime):
    transforms = runtime.attention_provider_transforms
    terminal = runtime.attention_provider_terminal
    identity = runtime.attention_provider_identity
    if not isinstance(transforms, tuple) or not all(callable(item) for item in transforms):
        raise RuntimeError("partitioned exact-prefix inherited preprocess state is malformed")
    if terminal is not None and not callable(terminal):
        raise RuntimeError("partitioned exact-prefix inherited terminal provider is malformed")
    if not isinstance(identity, tuple) or len(identity) != 3:
        raise RuntimeError("partitioned exact-prefix inherited provider identity is malformed")
    return transforms, terminal


def make_partitioned_attention_override(runtime: PartitionedStageRuntime, metrics):
    """Create one partitioned provider for one inherited numerical identity.

    The callable itself stays stable while an equivalent generic preprocess wrapper
    is reconstructed by an outer model-function wrapper. The runtime owner is
    rebound to the current call's preprocess functions before H3 executes, so
    dynamic per-call configuration is never frozen into the cached provider.
    """

    def apply_preprocess(q, k, v, heads, **kw):
        transforms, _terminal = _runtime_provider_state(runtime)
        for transform in transforms:
            q, k, v = transform(q, k, v, heads=heads, **kw)
        return q, k, v

    def partition_request(q, k, v, **kwargs):
        _transforms, terminal = _runtime_provider_state(runtime)
        return call_partitioned_attention(q, k, v, terminal=terminal, metrics=metrics, **kwargs)

    def vdn_preprocess(q, k, v, heads, **kwargs):
        # Generic Comfy preprocessing uses BHTD; VDN owns THD before gather.
        tensors = tuple(t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
        result = apply_preprocess(*tensors, heads=heads, **kwargs)
        return tuple(t[0].transpose(0, 1) for t in result)

    def partition_leaf(original, q, k, v, heads, mask=None, **kw):
        _transforms, terminal = _runtime_provider_state(runtime)
        options = kw.get("transformer_options") or {}
        raw_contract = options.get(PARTITIONED_PREFIX_KEY)
        if raw_contract is None:
            return _call_provider(terminal, original, q, k, v, heads, mask, kw)
        if any(key in options for key in _DEPRECATED_MIXED_GRID_KEYS):
            raise RuntimeError("partitioned exact-prefix refuses deprecated Mixed-Grid attention contracts")
        if mask is not None or not kw.get("skip_reshape") or kw.get("skip_output_reshape"):
            raise RuntimeError("partitioned exact-prefix attention requires unmasked skip_reshape H3 attention")
        if any(t.ndim != 4 for t in (q, k, v)) or q.shape != k.shape or q.shape != v.shape:
            raise RuntimeError("partitioned exact-prefix attention requires square BHTD input before VDN gathering")
        if q.shape[0] != 1 or q.shape[1] != int(heads) or q.shape[-1] != 128:
            raise RuntimeError("partitioned exact-prefix attention received unsupported H3 head geometry")
        plan = validate_partitioned_contract(raw_contract, sequence_rows=int(q.shape[2]))
        block_index = options.get(PARTITIONED_BLOCK_INDEX_KEY)
        if type(block_index) is not int or block_index < 0:
            raise RuntimeError("partitioned exact-prefix attention is missing its block identity")

        q_thd = q[0].transpose(0, 1)
        k_thd = k[0].transpose(0, 1)
        v_thd = v[0].transpose(0, 1)
        output = partition_request(
            q_thd,
            k_thd,
            v_thd,
            transformer_options=options,
            block_index=block_index,
            kind="full",
            scale=q.shape[-1] ** -0.5,
            sink_rows=plan.video_start,
            prefix_k_range=plan.prefix_range,
            prefix_log_key_measure=plan.prefix_log_key_measure,
            semantic_digest=str(raw_contract["semantic_digest"]),
        )
        metrics.increment("partitioned_attention_calls")
        metrics.increment("partitioned_requested_q_rows", int(q.shape[2]))
        metrics.increment("partitioned_kernel_q_rows", int(q.shape[2]))
        metrics.increment("partitioned_kv_rows", int(k.shape[2]))
        return output.reshape(q.shape[0], q.shape[2], -1)

    transforms, _terminal = _runtime_provider_state(runtime)
    if transforms:

        def override(original, q, k, v, heads, mask=None, **kw):
            q, k, v = apply_preprocess(q, k, v, heads, **kw)
            return partition_leaf(original, q, k, v, heads, mask=mask, **kw)

        override.attention_preprocess_v1 = (apply_preprocess, partition_leaf)
    else:
        override = partition_leaf

    override._h3_flow_partitioned_attention_override = True
    override.partitioned_attention_v1 = partition_request
    override.vdn_attention_preprocess_v1 = vdn_preprocess
    override._h3_flow_partitioned_previous = None
    override._h3_flow_partitioned_provider_identity = runtime.attention_provider_identity
    return override


def _stage_partitioned_attention_override(
    runtime: PartitionedStageRuntime, previous, metrics, *, native_attention=False
):
    """Return a stable partitioned provider across equivalent outer wrappers.

    Sol history-v1 treats generic preprocessing wrappers as part of a semantic
    chain: transform implementation names are significant, while the terminal
    dense provider keeps strict object identity. Mirror that exact distinction
    here. This preserves conservative transitions for genuine provider changes
    without turning an equivalent per-call Untwist wrapper reconstruction into a
    new partitioned numerical backend.
    """
    if not isinstance(runtime, PartitionedStageRuntime):
        raise RuntimeError("partitioned exact-prefix stage runtime owner is malformed")
    cache = runtime.attention_provider_cache
    if not isinstance(cache, dict):
        raise RuntimeError("partitioned exact-prefix attention provider cache is malformed")

    identity, transforms, terminal = _inherited_provider_state(previous, native_attention=native_attention)
    previous_identity = runtime.attention_provider_identity
    runtime.attention_provider_transforms = transforms
    runtime.attention_provider_terminal = terminal
    runtime.attention_provider_identity = identity

    if previous_identity is not None and previous_identity != identity:
        metrics.increment("partitioned_attention_inherited_provider_transitions")

    carrier_contract = runtime.vdn_temporal_carrier_contract
    carrier_digest = None if carrier_contract is None else carrier_contract.get("numerical_digest")
    numerical_identity = [("partitioned_native_attention_v1",)] if native_attention else []
    if carrier_digest is not None:
        numerical_identity.append(("vdn_temporal_carrier_v1", carrier_digest))
    if runtime.softmax_diagnostic != PARTITIONED_SOFTMAX_DIAGNOSTIC_NORMAL:
        numerical_identity.append((PARTITIONED_SOFTMAX_DIAGNOSTIC_KEY, runtime.softmax_diagnostic))
    if runtime.target_band_domain is not None:
        numerical_identity.append(runtime.target_band_domain.numerical_identity)
    cache_identity = identity if not numerical_identity else (tuple(numerical_identity), identity)
    cached = cache.get(cache_identity)
    if cached is not None:
        if not callable(cached) or not getattr(cached, "_h3_flow_partitioned_attention_override", False):
            raise RuntimeError("partitioned exact-prefix attention provider cache entry is malformed")
        if getattr(cached, "_h3_flow_partitioned_previous", None) is not previous:
            metrics.increment("partitioned_attention_equivalent_provider_rebindings")
        cached._h3_flow_partitioned_previous = previous
        cached._h3_flow_partitioned_provider_identity = identity
        metrics.increment("partitioned_attention_provider_reuses")
        return cached

    override = make_partitioned_attention_override(runtime, metrics)
    override._h3_flow_partitioned_previous = previous
    override._h3_flow_partitioned_provider_identity = identity
    cache[cache_identity] = override
    metrics.increment("partitioned_attention_provider_creations")
    return override


def _partitioned_transformer_options(
    options,
    partitioned_layout,
    partition_contract,
    metrics,
    *,
    runtime: PartitionedStageRuntime,
    block_index,
    stage_view: PartitionedStageStreamView | None = None,
    domain_stream: dict | None = None,
):
    block_options = dict(options)
    if stage_view is not None:
        block_options[PARTITIONED_STAGE_KEY] = stage_view
    if domain_stream is None:
        block_options.pop(TARGET_BAND_DOMAIN_STREAM_KEY, None)
    else:
        block_options[TARGET_BAND_DOMAIN_STREAM_KEY] = domain_stream
    if any(key in block_options for key in _DEPRECATED_MIXED_GRID_KEYS):
        raise RuntimeError("partitioned exact-prefix found a deprecated Mixed-Grid contract")
    block_options["minimax_h3_layout"] = partitioned_layout
    block_options[PARTITIONED_PREFIX_KEY] = partition_contract
    block_options[PARTITIONED_BLOCK_INDEX_KEY] = int(block_index)
    block_options[PARTITIONED_ATTENTION_PROVIDER_KEY] = block_options[
        "optimized_attention_override"
    ].partitioned_attention_v1
    if not sol_attention_selected(block_options) and "vdn_attention_preprocess_v1" not in block_options:
        entry = getattr(block_options["optimized_attention_override"], _PREPROCESS_ATTR, None)
        if entry is not None:
            block_options["vdn_attention_preprocess_v1"] = block_options[
                "optimized_attention_override"
            ].vdn_attention_preprocess_v1
    linear_mode = normalize_vdn_linear_diagnostic(runtime.vdn_linear_diagnostic)
    existing_linear_mode = block_options.get(PARTITIONED_VDN_LINEAR_DIAGNOSTIC_KEY)
    if existing_linear_mode is not None and existing_linear_mode != linear_mode:
        raise RuntimeError("partitioned exact-prefix VDN linear diagnostic transport drifted")
    block_options[PARTITIONED_VDN_LINEAR_DIAGNOSTIC_KEY] = linear_mode
    softmax_mode = str(runtime.softmax_diagnostic)
    if softmax_mode == PARTITIONED_SOFTMAX_DIAGNOSTIC_NORMAL:
        block_options.pop(PARTITIONED_SOFTMAX_DIAGNOSTIC_KEY, None)
    else:
        existing_softmax_mode = block_options.get(PARTITIONED_SOFTMAX_DIAGNOSTIC_KEY)
        if existing_softmax_mode is not None and existing_softmax_mode != softmax_mode:
            raise RuntimeError("partitioned exact-prefix softmax diagnostic transport drifted")
        block_options[PARTITIONED_SOFTMAX_DIAGNOSTIC_KEY] = softmax_mode
    carrier_contract = runtime.vdn_temporal_carrier_contract
    if carrier_contract is None:
        block_options.pop(PARTITIONED_VDN_TEMPORAL_CARRIER_KEY, None)
    else:
        existing_carrier = block_options.get(PARTITIONED_VDN_TEMPORAL_CARRIER_KEY)
        if existing_carrier is not None and existing_carrier != carrier_contract:
            raise RuntimeError("partitioned exact-prefix VDN temporal-carrier policy transport drifted")
        block_options[PARTITIONED_VDN_TEMPORAL_CARRIER_KEY] = carrier_contract
    if runtime.boundary_witness is not None:
        from .boundary_witness import WITNESS_KEY

        block_options[WITNESS_KEY] = runtime.boundary_witness
    expected_vdn = _vdn_external_contract(
        validate_partitioned_contract(
            partition_contract,
            sequence_rows=int(partitioned_layout.seq_len),
        )
    )
    existing_vdn = block_options.get(VDN_EXTERNAL_SEQUENCE_KEY)
    if existing_vdn is not None and existing_vdn != expected_vdn:
        raise RuntimeError("partitioned exact-prefix found an already-owned VDN external sequence")
    block_options[VDN_EXTERNAL_SEQUENCE_KEY] = expected_vdn
    metrics.increment("partitioned_contract_publications")
    return block_options


def _audio_model_timestep_kwargs(options, kwargs):
    """Verify exact native masks or override diagnostic inner timestep labels."""

    context = options.get(PARTITIONED_AUDIO_MODEL_TIMESTEP_CONTEXT_KEY)
    if context is None:
        return kwargs
    if not isinstance(context, PartitionedAudioModelTimestepContext):
        raise RuntimeError("partitioned audio model-timestep context is malformed")
    runtime_audio_mask = kwargs.get("audio_denoise_mask")
    context_audio_mask = context.audio_mask
    if context.mask_kind == "exact_authoritative" and runtime_audio_mask is None:
        # Core omits the condition when every audio row is fully generated.
        if not torch.is_tensor(context_audio_mask) or not bool((context_audio_mask == 1).all().item()):
            raise RuntimeError("exact protected audio requires the native runtime audio denoise mask")
        context.record_verification()
        return kwargs
    if not torch.is_tensor(runtime_audio_mask):
        raise RuntimeError("partitioned audio timestep override requires the native runtime audio denoise mask")
    if (
        context.mask_kind == "exact_authoritative"
        and torch.is_tensor(context_audio_mask)
        and context_audio_mask.ndim == runtime_audio_mask.ndim
        and context_audio_mask.shape[0] == 1
        and tuple(context_audio_mask.shape[1:]) == tuple(runtime_audio_mask.shape[1:])
    ):
        # Core can repeat the canonical condition when batching CFG branches.
        context_audio_mask = context_audio_mask.expand_as(runtime_audio_mask)
    if not torch.is_tensor(context_audio_mask) or tuple(context_audio_mask.shape) != tuple(runtime_audio_mask.shape):
        raise RuntimeError("partitioned audio timestep override mask geometry drifted")
    if context.mask_kind == "exact_authoritative":
        expected = context_audio_mask.to(device=runtime_audio_mask.device, dtype=runtime_audio_mask.dtype)
        if not torch.equal(runtime_audio_mask, expected):
            raise RuntimeError(
                "exact audio model labels require the same authoritative sampler input and velocity mask"
            )
        context.record_verification()
        return kwargs
    local = dict(kwargs)
    local["audio_denoise_mask"] = context_audio_mask.to(
        device=runtime_audio_mask.device,
        dtype=runtime_audio_mask.dtype,
    )
    context.record_call()
    return local


DOMAIN_UNIFORM_IDENTITY = "h3_flow_partitioned_domain_uniform_v1"
DOMAIN_STREAM_TARGET = "target"
DOMAIN_STREAM_SOURCE = "source"


@dataclass(frozen=True, slots=True)
class DomainUniformStream:
    """One uniform-grid hidden stream of target-band domain-uniform execution.

    ``layout`` is a native Core layout for the stream geometry whose signature
    carries the Flow partition identity. ``plan`` is a canonical equal-grid
    partition contract, so VDN and Sol execute their ordinary uniform routing
    for this stream: local windows, globals, row/column anchors and the learned
    linear complement all span only this stream's rows.
    """

    name: str
    plan: PartitionedExactPrefixPlan
    layout: object
    contract: dict
    view: PartitionedStageStreamView
    leaf: dict
    attention_head_t: int
    audio_rows: int

    @property
    def rows(self) -> int:
        return int(self.plan.sequence_rows)


def _domain_stream_leaf(*, name, contract, native_layout, policy):
    return {
        "api": TARGET_BAND_DOMAIN_STREAM_API,
        "policy": policy,
        "stream": name,
        "flow_semantic_digest": contract["semantic_digest"],
        "native_sequence_rows": int(native_layout.seq_len),
        "native_video_start": int(native_layout.segments[-1][0]),
    }


def _segment(layout, kind):
    matches = [(int(start), int(stop)) for start, stop, seg_kind in layout.segments if seg_kind == kind]
    if len(matches) != 1:
        raise RuntimeError(f"domain-uniform execution requires exactly one target {kind} segment")
    return matches[0]


def _audio_crop_rows(audio_start: int, audio_t: int, keep_t: int, device) -> torch.Tensor:
    # Core packs stereo audio channel-major: [left 0..T) then [right 0..T).
    first = torch.arange(audio_start, audio_start + keep_t, device=device)
    return torch.cat((first, first + audio_t))


def _video_frame_count(native, tokens: int) -> int:
    return sum(int(native.FRAME_PER_TOKEN[k % len(native.FRAME_PER_TOKEN)]) for k in range(int(tokens)))


def _domain_video_labels(row, *, protected_rows, generated_rows, band: PartitionedTargetBandGeometry, device):
    """Return (protected, generated) per-row timestep labels from Core's native video row."""
    if not torch.is_tensor(row):
        return row, row
    if row.ndim != 1:
        raise RuntimeError("domain-uniform execution requires native per-video-row timestep indices")
    protected = row[: band.protected_rows]
    generated = torch.cat((row[band.protected_rows : band.prefix_rows], row[band.tail_native_rows(row.device)]))
    checks = torch.stack(((protected == protected[0]).all(), (generated == generated[0]).all()))
    if not bool(checks.all().item()):
        raise RuntimeError("domain-uniform execution requires uniform protected-prefix and generated timestep labels")
    return protected[0], generated[0]


def _domain_mod_segments(native_segments, *, audio_range, native_video_start, video_range, crop_audio, video_row):
    result = []
    audio_start, audio_stop = audio_range
    video_start, video_end = video_range
    for start, stop, row in native_segments:
        if stop <= audio_start:
            result.append((start, stop, row))
        elif (start, stop) == (audio_start, audio_stop):
            if crop_audio is None:
                result.append((start, stop, row))
            else:
                rows, keep = crop_audio
                cropped = row.index_select(0, rows - audio_start) if torch.is_tensor(row) else row
                result.append((start, start + keep, cropped))
        elif start == native_video_start:
            result.append((video_start, video_end, video_row))
        else:
            raise RuntimeError("domain-uniform execution requires [conditioning | audio | video] native segments")
    if not result or result[-1][0] != video_start or video_end <= video_start:
        raise RuntimeError("domain-uniform execution lost the native video segment")
    return result


def _domain_partitioned_layout(stream_layout, plan, policy, name):
    partitioned = copy.copy(stream_layout)
    partitioned.signature = (
        PARTITIONED_PREFIX_KEY,
        *stream_layout.signature,
        plan.prefix_t,
        plan.target_grid_h * 2,
        plan.target_grid_w * 2,
        (TARGET_BAND_DOMAIN_STREAM_KEY, policy, name),
    )
    return partitioned


def _domain_uniform_forward(
    executor,
    x,
    timestep,
    context,
    options,
    payload,
    kwargs,
    *,
    native,
    inner,
    runtime: PartitionedStageRuntime,
    owner: PartitionedStagePlan,
    band: PartitionedTargetBandGeometry,
    layout,
    tail_rows,
):
    """Evaluate a target band and the full clip as separate uniform hidden streams.

    Routing contract (``domain_uniform_v1``):

    * Target stream: [text | references | audio covering the head duration |
      exact target prefix | target band]. Its Core layout is the native layout of
      a ``head_t``-token clip on the target grid; RoPE rows equal the matching
      rows of the chunk's native target layout.
    * Source stream: [text | references | full audio | projected exact prefix |
      projected band carrier | reduced-grid tail]. Its Core layout is the native
      layout of the whole chunk on the reduced grid.
    * Each stream owns its conditioning rows. No attention key, VDN local window,
      global or anchor query, linear-complement state or MLP/modulation row is
      shared between streams within a model call, so a stream's hidden states
      never read the other stream's hidden states.
    * Communication across streams happens only through the sampler state
      between model calls: the source stream reads the band's current state
      (projected), and both streams read the shared audio state.
    * Outputs: band velocity from the target stream; tail and audio velocity
      from the source stream; the protected prefix and tail padding stay zero.

    During equal-grid high refinement, the source stream is the full target-grid
    clip and reads the band's native state directly. The short target stream
    retains its low/probe clip extent and head-duration audio.
    """
    domain: TargetBandDomainContext = runtime.target_band_domain
    domain_policy = domain.policy
    metrics = runtime.metrics
    if payload.get("keyframes"):
        raise RuntimeError("domain-uniform target-band context does not support keyframe-anchored layouts")
    if runtime.vdn_temporal_carrier_policy != PARTITIONED_VDN_TEMPORAL_CARRIER_NATIVE:
        raise RuntimeError("domain-uniform target-band context requires the native VDN temporal-carrier policy")
    video_start, video_end, _ = layout.segments[-1]
    audio_start, audio_stop = _segment(layout, "audio")
    if audio_stop != video_start:
        raise RuntimeError("domain-uniform execution requires target audio directly before target video")
    audio_t = int(x[1].shape[-1])
    if audio_stop - audio_start != 2 * audio_t:
        raise RuntimeError("domain-uniform execution audio rows do not match the sampler audio state")
    rescale = float(native.FRAME_RESCALE)
    chunk_audio_t = round(_video_frame_count(native, band.temporal) * rescale)
    if abs(chunk_audio_t - audio_t) > 1:
        raise RuntimeError(
            "domain-uniform execution requires the native audio/video duration relation; "
            f"audio has {audio_t} latent frames but the video spans {chunk_audio_t}"
        )
    head_audio_t = min(audio_t, round(_video_frame_count(native, band.head_t) * rescale))
    if head_audio_t <= 0:
        raise RuntimeError("domain-uniform target stream would contain no audio rows")
    text_len = int(context.shape[1])
    refs = payload.get("refs")
    target_layout = native.PackedLayout(text_len, band.head_t, *owner.target_hw, head_audio_t, refs=refs)
    source_layout = native.PackedLayout(text_len, band.temporal, band.source_h, band.source_w, audio_t, refs=refs)
    device = x[0].device
    crop_rows = _audio_crop_rows(audio_start, audio_t, head_audio_t, device)

    # Physical coordinate invariants: both streams reuse the chunk's conditioning
    # coordinates and global timeline; only the video grid of each stream differs.
    t_audio_start, t_audio_stop = _segment(target_layout, "audio")
    s_audio_start, s_audio_stop = _segment(source_layout, "audio")
    t_video_start = int(target_layout.segments[-1][0])
    if (
        tuple(target_layout.segments[:-2]) != tuple(layout.segments[:-2])
        or tuple(source_layout.segments) != (*layout.segments[:-1], (video_start, source_layout.seq_len, "video"))
        or (t_audio_start, s_audio_start, s_audio_stop) != (audio_start, audio_start, audio_stop)
        or t_audio_stop != t_video_start
        or t_audio_stop - t_audio_start != 2 * head_audio_t
    ):
        raise RuntimeError("domain-uniform stream layouts drifted from the native conditioning layout")
    native_positions = layout.position_ids
    target_positions = target_layout.position_ids
    source_positions = source_layout.position_ids
    head_rows = band.head_t * band.target_rows
    # Audio-only references use the receiving clip's spatial endpoints. Their
    # times stay fixed, while image/video reference coordinates stay unchanged.
    source_conditioning_matches = all(
        torch.equal(
            source_positions[start:stop, :1] if kind == "ref_audio" else source_positions[start:stop],
            native_positions[start:stop, :1] if kind == "ref_audio" else native_positions[start:stop],
        )
        for start, stop, kind in source_layout.segments[:-2]
    )
    if not (
        torch.equal(target_positions[:audio_start], native_positions[:audio_start])
        and torch.equal(target_positions[t_audio_start:t_audio_stop], native_positions[crop_rows.cpu()])
        and torch.equal(target_positions[t_video_start:], native_positions[video_start : video_start + head_rows])
        and source_conditioning_matches
        and torch.equal(source_positions[audio_start:audio_stop, 0], native_positions[audio_start:audio_stop, 0])
    ):
        raise RuntimeError("domain-uniform stream RoPE rows drifted from the chunk's physical coordinates")
    native_frame_t = native_positions[video_start : video_end : band.target_rows, 0]
    source_frame_t = source_positions[video_start :: band.source_rows, 0]
    if not torch.equal(native_frame_t, source_frame_t):
        raise RuntimeError("domain-uniform source stream changed the chunk's temporal coordinates")

    target_plan = PartitionedExactPrefixPlan(
        video_start=t_video_start,
        temporal=int(band.head_t),
        prefix_t=int(band.protected_t),
        source_grid_h=int(band.target_grid[0]),
        source_grid_w=int(band.target_grid[1]),
        target_grid_h=int(band.target_grid[0]),
        target_grid_w=int(band.target_grid[1]),
        same_grid_control=True,
    )
    # The tail is a continuation of the band: once a clean band estimate exists,
    # the source stream holds the band as known prefix-conditioned frames.
    band_estimate = None if band.same_grid_control else domain.band_estimate.get("x0")
    if band_estimate is not None and tuple(band_estimate.shape) != (1, 24, band.band_t, *owner.target_hw):
        raise RuntimeError("domain-uniform band estimate does not match the band geometry")
    source_prefix_t = band.head_t if band_estimate is not None else band.protected_t
    source_plan = PartitionedExactPrefixPlan(
        video_start=int(video_start),
        temporal=int(band.temporal),
        prefix_t=int(source_prefix_t),
        source_grid_h=int(band.source_grid[0]),
        source_grid_w=int(band.source_grid[1]),
        target_grid_h=int(band.source_grid[0]),
        target_grid_w=int(band.source_grid[1]),
        same_grid_control=True,
    )
    if target_plan.sequence_rows != int(target_layout.seq_len) or source_plan.sequence_rows != int(
        source_layout.seq_len
    ):
        raise RuntimeError("domain-uniform stream plans do not match their native layouts")
    streams = []
    # Dense-query ownership matches the mixed-grid band path: every band query
    # stays dense in both streams, and the source stream keeps the local group
    # containing the first tail token dense as the mixed path's boundary group.
    for name, plan, stream_layout, head, audio_rows in (
        (DOMAIN_STREAM_TARGET, target_plan, target_layout, band.head_t - 1, 2 * head_audio_t),
        (DOMAIN_STREAM_SOURCE, source_plan, source_layout, band.head_t, 2 * audio_t),
    ):
        contract = plan.to_contract()
        partitioned_layout = _domain_partitioned_layout(stream_layout, plan, domain_policy, name)
        streams.append(
            DomainUniformStream(
                name=name,
                plan=plan,
                layout=partitioned_layout,
                contract=contract,
                view=PartitionedStageStreamView(runtime, stream=name, attention_head_t=head),
                leaf=_domain_stream_leaf(name=name, contract=contract, native_layout=layout, policy=domain_policy),
                attention_head_t=int(head),
                audio_rows=int(audio_rows),
            )
        )
    streams = tuple(streams)
    target_stream, source_stream = streams

    local = dict(options)
    local.pop(PARTITIONED_VDN_TEMPORAL_CARRIER_KEY, None)
    runtime.vdn_temporal_carrier_contract = None
    local["optimized_attention_override"] = _stage_partitioned_attention_override(
        runtime,
        local.get("optimized_attention_override"),
        metrics,
        native_attention=not sol_attention_selected(options),
    )
    patches = dict(local.get("patches_replace") or {})
    blocks = dict(patches.get("dit") or {})
    patches["dit"] = blocks
    local["patches_replace"] = patches
    cached = {}
    last_layer = len(inner.blocks) - 1

    def build_streams(img):
        aug = float(native.VISUAL_COND_TIMESTEP)
        prefix = owner.prefix.to(device=img.device, dtype=torch.float32)
        prefix_noise = owner.prefix_noise.to(device=img.device, dtype=torch.float32)
        target_prefix = inner.video_patch_proj(native.patchify_video(aug * prefix + (1.0 - aug) * prefix_noise)).to(img)
        source_prefix = domain.source_prefix.to(device=img.device, dtype=torch.float32)
        source_prefix_noise = domain.source_prefix_noise.to(device=img.device, dtype=torch.float32)
        projected_prefix = inner.video_patch_proj(
            native.patchify_video(aug * source_prefix + (1.0 - aug) * source_prefix_noise)
        ).to(img)
        # Band carrier on the reduced grid at the model call's own sigma. The
        # physical projection is convex, so it lowers Gaussian noise variance;
        # the fixed complementary field restores the nominal per-cell variance.
        band_state = x[0][:, :, band.protected_t : band.head_t].to(torch.float32)
        if band.same_grid_control:
            carrier = band_state
        elif band_estimate is not None:
            estimate = resize_spatial_5d_h3_patch_lattice(
                band_estimate.to(device=img.device, dtype=torch.float32), band.source_h, band.source_w
            )
            band_noise = domain.source_band_noise.to(device=img.device, dtype=torch.float32)
            carrier = aug * estimate + (1.0 - aug) * band_noise
        else:
            sigma = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
            projected_band = resize_spatial_5d_h3_patch_lattice(band_state, band.source_h, band.source_w)
            complement = domain.band_noise_complement.to(device=img.device, dtype=torch.float32)
            band_noise = domain.source_band_noise.to(device=img.device, dtype=torch.float32)
            carrier = projected_band + (sigma * float(domain.model_noise_scale)) * complement * band_noise
        band_rows = inner.video_patch_proj(native.patchify_video(carrier)).to(img)
        target_rows = torch.cat(
            (
                img[:audio_start],
                img.index_select(0, crop_rows),
                target_prefix,
                img[video_start + band.protected_rows : video_start + band.prefix_rows],
            )
        )
        source_rows = torch.cat(
            (
                img[:video_start],
                projected_prefix,
                band_rows,
                img.index_select(0, tail_rows),
            )
        )
        if len(target_rows) != target_stream.rows or len(source_rows) != source_stream.rows:
            raise RuntimeError("domain-uniform stream construction changed its row count")
        return torch.cat((target_rows, source_rows))

    def stream_mod_segments(native_segments):
        video_row = next(row for start, _stop, row in native_segments if start == video_start)
        protected, generated = _domain_video_labels(
            video_row,
            protected_rows=band.protected_rows,
            generated_rows=None,
            band=band,
            device=device,
        )
        segments = {}
        for stream, crop in ((target_stream, (crop_rows, 2 * head_audio_t)), (source_stream, None)):
            plan = stream.plan
            if torch.is_tensor(video_row):
                row = torch.cat(
                    (
                        protected.reshape(1).expand(plan.prefix_rows),
                        generated.reshape(1).expand(plan.sequence_rows - plan.video_start - plan.prefix_rows),
                    )
                )
            else:
                row = video_row
            segments[stream.name] = _domain_mod_segments(
                native_segments,
                audio_range=(audio_start, audio_stop),
                native_video_start=video_start,
                video_range=(plan.video_start, plan.sequence_rows),
                crop_audio=crop,
                video_row=row,
            )
        return segments

    def wrap(layer, previous):
        def call(args, extra):
            img = args["img"]
            if layer == 0:
                img = build_streams(img)
                metrics.increment("partitioned_transformer_calls")
                metrics.increment("partitioned_domain_uniform_calls")
                metrics.increment("partitioned_domain_uniform_target_rows", target_stream.rows)
                metrics.increment("partitioned_domain_uniform_source_rows", source_stream.rows)
                metrics.event(
                    "partitioned_target_band_domain_transformer",
                    policy=domain_policy,
                    band_carrier_policy=(
                        "native_target_sampler_state_v1"
                        if band.same_grid_control
                        else TARGET_BAND_DOMAIN_BAND_CONDITIONING_POLICY
                        if band_estimate is not None
                        else TARGET_BAND_DOMAIN_BAND_CARRIER_POLICY
                    ),
                    source_prefix_t=int(source_prefix_t),
                    stage=options.get("h3_flow_stage"),
                    native_sequence_rows=int(layout.seq_len),
                    protected_prefix_t=int(band.protected_t),
                    head_t=int(band.head_t),
                    temporal=int(band.temporal),
                    target_hw=tuple(map(int, owner.target_hw)),
                    source_hw=(int(band.source_h), int(band.source_w)),
                    streams=tuple(
                        {
                            "stream": stream.name,
                            "sequence_rows": stream.rows,
                            "video_start": int(stream.plan.video_start),
                            "temporal": int(stream.plan.temporal),
                            "rows_per_frame": int(stream.plan.target_rows),
                            "audio_rows": stream.audio_rows,
                            "attention_head_t": stream.attention_head_t,
                            "semantic_digest": stream.contract["semantic_digest"],
                        }
                        for stream in streams
                    ),
                    shared_hidden_rows=0,
                    conditioning_rows_per_stream=True,
                    cross_stream_attention_keys=0,
                    output_owner={
                        "band": DOMAIN_STREAM_TARGET,
                        "tail": DOMAIN_STREAM_SOURCE,
                        "audio": DOMAIN_STREAM_SOURCE,
                    },
                )
            if len(img) != target_stream.rows + source_stream.rows:
                raise RuntimeError("domain-uniform transformer row count mismatch")
            if "rope" not in cached:
                cached["rope"] = {
                    stream.name: native.rope_rotation_table(
                        inner.rope_freqs(stream.layout.position_ids, img.device),
                        img.dtype,
                    )
                    for stream in streams
                }
            if cached.get("mod_source") is not args["mod_segments"]:
                cached["mod_segments"] = stream_mod_segments(args["mod_segments"])
                cached["mod_source"] = args["mod_segments"]
                metrics.increment("partitioned_modulation_validations")
            views = {
                target_stream.name: img[: target_stream.rows],
                source_stream.name: img[target_stream.rows :],
            }
            calls = []
            for stream in streams:
                view = views[stream.name]
                forwarded = dict(args)
                forwarded.update(
                    img=view,
                    layout=stream.layout,
                    rope_freqs=cached["rope"][stream.name],
                    mod_segments=[
                        (start, stop, row.clone() if start == stream.plan.video_start and torch.is_tensor(row) else row)
                        for start, stop, row in cached["mod_segments"][stream.name]
                    ],
                )
                forwarded["transformer_options"] = _partitioned_transformer_options(
                    args["transformer_options"],
                    stream.layout,
                    stream.contract,
                    metrics,
                    runtime=runtime,
                    block_index=layer,
                    stage_view=stream.view,
                    domain_stream=stream.leaf,
                )
                calls.append((stream, forwarded))

            # The two attention domains are independent, but normalization,
            # AdaLN, residual operations and the MLP are row-wise. An unwrapped
            # native Core DiT block may process their union once while the
            # attention callback still executes the independent VDN streams.
            # Leave external block replacements on the two-call path: they may
            # inspect a stream's layout or mutate per-stream modulation rows.
            original = extra["original_block"]
            native_block = None
            if previous is None:
                with contextlib.suppress(TypeError, ValueError):
                    native_block = inspect.getclosurevars(original).nonlocals.get("block")
            if (
                native_block is inner.blocks[layer]
                and type(native_block) is native.DiTBlock
                and getattr(native_block.forward, "__func__", None) is native.DiTBlock.forward
            ):
                offsets = (0, target_stream.rows)
                combined_mod_segments = [
                    (first + offset, last + offset, row)
                    for offset, (_, forwarded) in zip(offsets, calls, strict=True)
                    for first, last, row in forwarded["mod_segments"]
                ]

                def independent_attention(hidden, rope_freqs=None, transformer_options=None):
                    if hidden.shape != img.shape:
                        raise RuntimeError("domain-uniform fused block changed the hidden rows")
                    outputs = [
                        native_block.attn(
                            hidden[offset : offset + stream.rows],
                            rope_freqs=forwarded["rope_freqs"],
                            transformer_options=forwarded["transformer_options"],
                        )
                        for offset, (stream, forwarded) in zip(offsets, calls, strict=True)
                    ]
                    return torch.cat(outputs, dim=0)

                combined_args = dict(args)
                combined_args.update(
                    img=img,
                    mod_segments=combined_mod_segments,
                    rope_freqs=None,
                    attention=independent_attention,
                )
                output = original(combined_args)
                result = output["img"]
                if result.shape != img.shape:
                    raise RuntimeError("domain-uniform fused Core block returned incompatible hidden state")
                img = result
                views = {
                    target_stream.name: img[: target_stream.rows],
                    source_stream.name: img[target_stream.rows :],
                }
                metrics.increment("partitioned_domain_uniform_fused_core_block_calls")
                for stream in streams:
                    metrics.increment(f"partitioned_domain_uniform_{stream.name}_block_calls")
            else:
                block_extra = extra if sol_attention_selected(options) else partitioned_block_extra(extra)
                output = None
                for stream, forwarded in calls:
                    view = views[stream.name]
                    output = previous(forwarded, block_extra) if previous else block_extra["original_block"](forwarded)
                    result = output["img"]
                    if result.shape != view.shape:
                        raise RuntimeError("domain-uniform stream block returned incompatible hidden state")
                    if result.data_ptr() != view.data_ptr():
                        view.copy_(result)
                    metrics.increment(f"partitioned_domain_uniform_{stream.name}_block_calls")
                metrics.increment("partitioned_domain_uniform_separate_block_calls")
            if layer == last_layer:
                target_view = views[target_stream.name]
                source_view = views[source_stream.name]
                native_rows = img.new_zeros((int(layout.seq_len), img.shape[1]))
                native_rows[:video_start] = source_view[:video_start]
                band_start = target_stream.plan.video_start + band.protected_rows
                native_rows[video_start + band.protected_rows : video_start + band.prefix_rows] = target_view[
                    band_start : band_start + band.band_t * band.target_rows
                ]
                native_rows.index_copy_(0, tail_rows, source_view[video_start + band.head_t * band.source_rows :])
                return {**output, "img": native_rows}
            return {**output, "img": img}

        return call

    for layer in range(len(inner.blocks)):
        blocks[("double_block", layer)] = wrap(layer, blocks.get(("double_block", layer)))
    result = executor(x, timestep, context, local, minimax_payload=payload, **kwargs)
    if not band.same_grid_control:
        # Flow-matching clean estimate x0 = x - sigma * v of the band from the
        # target stream, for the next call's source-stream band conditioning.
        sigma = (timestep.flatten()[0] / 1000.0).float()
        span = slice(band.protected_t, band.head_t)
        domain.band_estimate["x0"] = (
            (x[0][:, :, span].float() - sigma.to(x[0].device) * result[0][:, :, span].float()).detach().clone()
        )
    video = result[0].clone()
    video[:, :, : owner.prefix_t] = 0
    video[:, :, band.head_t :, band.source_h :] = 0
    video[:, :, band.head_t :, : band.source_h, band.source_w :] = 0
    return [video, result[1]]


def partitioned_diffusion_wrapper(
    executor,
    x,
    timestep,
    context,
    transformer_options=None,
    minimax_payload=None,
    **kwargs,
):
    """Expose exact target-prefix and low-grid suffix as explicit physical domains."""
    options = transformer_options or {}
    kwargs = _audio_model_timestep_kwargs(options, kwargs)
    runtime = options.get(PARTITIONED_STAGE_KEY)
    if runtime is None:
        return executor(
            x,
            timestep,
            context,
            options,
            minimax_payload=minimax_payload,
            **kwargs,
        )
    if any(key in options for key in _DEPRECATED_MIXED_GRID_KEYS):
        raise RuntimeError("partitioned exact-prefix refuses deprecated Mixed-Grid stage state")
    if not isinstance(runtime, PartitionedStageRuntime):
        raise RuntimeError("partitioned exact-prefix stage contract must be a runtime owner object")

    # ``owner`` owns the authoritative protected prefix. ``plan`` is the attention
    # geometry published to VDN and Sol: the owner itself, or the target-band head
    # geometry whose ``prefix_t`` covers the protected prefix plus the band.
    owner = runtime.plan
    band = runtime.target_band
    plan = owner if band is None else band
    metrics = runtime.metrics
    if runtime.audio_position_domain == PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE:
        metrics.increment("partitioned_audio_position_candidate_wrapper_entries")
    inner = executor.class_obj
    if not isinstance(owner, PartitionedStagePlan) or metrics is None or len(inner.blocks) == 0:
        raise RuntimeError("partitioned exact-prefix requires a valid stage plan and metrics owner")
    if band is not None and (
        not isinstance(band, PartitionedTargetBandGeometry)
        or band.protected_t != owner.prefix_t
        or band.temporal != owner.temporal
        or (band.source_h, band.source_w) != (owner.source_h, owner.source_w)
        or band.target_hw != owner.target_hw
    ):
        raise RuntimeError("target-band geometry does not match the protected-prefix stage plan")
    prefix_context = str(runtime.prefix_transformer_context)
    if prefix_context not in (
        PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT,
        PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE,
    ):
        raise RuntimeError(f"unsupported partitioned prefix transformer context {prefix_context!r}")
    attention_override = options.get("optimized_attention_override")
    if getattr(attention_override, "_h3_flow_attention_override", False):
        raise RuntimeError("partitioned exact-prefix does not support uniform-grid Flow Attention Lab overrides")
    carrier_hw = owner.target_hw if band is not None else (owner.source_h, owner.source_w)
    if tuple(x[0].shape) != (1, 24, owner.temporal, *carrier_hw):
        raise RuntimeError("stale partitioned exact-prefix plan does not match sampler geometry")
    if tuple(owner.prefix_noise.shape) != tuple(owner.prefix.shape):
        raise RuntimeError("partitioned exact-prefix plan is missing protected-prefix sampler noise")
    if band is not None and prefix_context != PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT:
        raise RuntimeError("target-band continuation requires the exact target-grid prefix transformer context")

    if prefix_context == PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE:
        # Diagnostic only: keep Core's complete low/probe hidden sequence on the
        # native source grid. This intentionally does not publish Flow's
        # heterogeneous partition contract, so VDN and Sol execute their ordinary
        # uniform-grid paths. The outer exact mask still owns caller-visible carried
        # prefix values; learned handoff/high-stage behavior is unchanged.
        metrics.increment("partitioned_source_carrier_uniform_transformer_calls")
        metrics.increment("partitioned_source_carrier_uniform_prefix_frames", int(plan.prefix_t))
        metrics.event(
            "partitioned_prefix_transformer_context",
            mode=prefix_context,
            temporal=int(plan.temporal),
            prefix_t=int(plan.prefix_t),
            source_hw=(int(plan.source_h), int(plan.source_w)),
            target_hw=tuple(map(int, plan.target_hw)),
            heterogeneous_partition_contract_published=False,
            exact_target_prefix_injected_into_transformer=False,
            native_source_carrier_preserved=True,
            external_exact_prefix_owner_unchanged=True,
            diagnostic_only=True,
        )
        return executor(
            x,
            timestep,
            context,
            options,
            minimax_payload=minimax_payload,
            **kwargs,
        )

    import comfy.ldm.minimax.model as native

    payload = dict(minimax_payload or {})
    if band is None:
        layout = partitioned_carrier_layout(
            native,
            owner,
            context.shape[1],
            x[1].shape[-1],
            payload,
        )
    else:
        # The sampler state is on the uniform target grid: the native layout is
        # Core's own target layout and only the transformer sequence is partitioned.
        layout = native.PackedLayout(
            context.shape[1],
            owner.temporal,
            *owner.target_hw,
            x[1].shape[-1],
            keyframes=payload.get("keyframes"),
            refs=payload.get("refs"),
        )
    payload["layout"] = layout
    exact_context_range = None
    exact_context = runtime.exact_prefix_visual_context
    if exact_context is not None:
        if band is not None or options.get("h3_flow_stage") not in ("low", "probe"):
            raise RuntimeError("exact uniform-source visual context is restricted to low/probe without a target band")
        layout, payload, exact_context_range = add_exact_prefix_visual_context(
            native, layout, payload, owner, exact_context
        )
    video_start, video_end, _ = layout.segments[-1]
    carrier_prefix_rows = plan.prefix_t * (plan.target_rows if band is not None else plan.source_rows)
    # Bound on every path: Sol reads every closure cell of the block replacement
    # for history identity, and an unbound cell makes that identity opaque.
    tail_rows = None
    if band is not None:
        if video_end - video_start != band.native_rows:
            raise RuntimeError("target-band native layout does not match the target-grid sampler state")
        tail_rows = band.tail_native_rows(x[0].device) + int(video_start)
        if runtime.target_band_domain is not None:
            return _domain_uniform_forward(
                executor,
                x,
                timestep,
                context,
                options,
                payload,
                kwargs,
                native=native,
                inner=inner,
                runtime=runtime,
                owner=owner,
                band=band,
                layout=layout,
                tail_rows=tail_rows,
            )
    positions, position_policy = partitioned_positions_for_runtime(native, runtime, layout)
    partitioned_layout = copy.copy(layout)
    partitioned_layout.position_ids = positions
    partitioned_layout.seq_len = video_start + plan.partitioned_rows
    partitioned_layout.segments = [
        *layout.segments[:-1],
        (video_start, partitioned_layout.seq_len, "video"),
    ]
    partitioned_layout.signature = (
        PARTITIONED_PREFIX_KEY,
        *layout.signature,
        plan.prefix_t,
        *plan.target_hw,
    )
    if position_policy is not None:
        partitioned_layout.signature = (*partitioned_layout.signature, position_policy.signature)
    if exact_context is not None:
        partitioned_layout.signature = (
            *partitioned_layout.signature,
            (POLICY, exact_context.prefix_t, *exact_context.target_hw),
        )
    if runtime.softmax_diagnostic != PARTITIONED_SOFTMAX_DIAGNOSTIC_NORMAL:
        # Sol history-v1 includes the complete partitioned layout signature in
        # numerical identity. Keep the geometry digest unchanged and add only
        # this diagnostic leaf so sparse and same-domain dense samples cannot
        # share history identity.
        partitioned_layout.signature = (
            *partitioned_layout.signature,
            (PARTITIONED_SOFTMAX_DIAGNOSTIC_KEY, runtime.softmax_diagnostic),
        )
    keep = layout.img_pos < video_start
    partitioned_layout.img_pos = torch.cat(
        (layout.img_pos[keep], torch.arange(video_start, partitioned_layout.seq_len))
    )
    partitioned_layout.img_update = torch.cat(
        (layout.img_update[keep], torch.ones(plan.partitioned_rows, dtype=torch.bool))
    )
    partition_plan = PartitionedExactPrefixPlan(
        video_start=int(video_start),
        temporal=int(plan.temporal),
        prefix_t=int(plan.prefix_t),
        source_grid_h=int(plan.source_grid[0]),
        source_grid_w=int(plan.source_grid[1]),
        target_grid_h=int(plan.target_grid[0]),
        target_grid_w=int(plan.target_grid[1]),
        same_grid_control=plan.source_grid == plan.target_grid,
        native_carrier_grid=(
            PARTITIONED_NATIVE_CARRIER_TARGET if band is not None else PARTITIONED_NATIVE_CARRIER_SOURCE
        ),
    )
    partition_contract = partition_plan.to_contract()
    if runtime.vdn_temporal_carrier_policy == PARTITIONED_VDN_TEMPORAL_CARRIER_DESTINATION:
        if not runtime.vdn_temporal_carrier_short_conv_spec:
            raise RuntimeError("destination-grid temporal stencil is missing paired VDN checkpoint capability")
        carrier_contract = build_vdn_temporal_carrier_contract(
            policy=runtime.vdn_temporal_carrier_policy,
            flow_semantic_digest=partition_contract["semantic_digest"],
            diagnostic_mode=runtime.vdn_linear_diagnostic,
            short_conv_spec=runtime.vdn_temporal_carrier_short_conv_spec,
        )
        if (
            runtime.vdn_temporal_carrier_contract is not None
            and runtime.vdn_temporal_carrier_contract != carrier_contract
        ):
            raise RuntimeError("partitioned exact-prefix temporal-carrier numerical identity changed within one stage")
        runtime.vdn_temporal_carrier_contract = carrier_contract
    elif runtime.vdn_temporal_carrier_policy == PARTITIONED_VDN_TEMPORAL_CARRIER_NATIVE:
        runtime.vdn_temporal_carrier_contract = None
    else:
        raise RuntimeError(f"unsupported VDN temporal-carrier policy {runtime.vdn_temporal_carrier_policy!r}")
    if partition_plan.sequence_rows != int(partitioned_layout.seq_len):
        raise RuntimeError("partitioned exact-prefix plan does not match transformed sequence rows")

    local = dict(options)
    if runtime.vdn_temporal_carrier_contract is None:
        local.pop(PARTITIONED_VDN_TEMPORAL_CARRIER_KEY, None)
    else:
        local[PARTITIONED_VDN_TEMPORAL_CARRIER_KEY] = runtime.vdn_temporal_carrier_contract
    local["optimized_attention_override"] = _stage_partitioned_attention_override(
        runtime,
        local.get("optimized_attention_override"),
        metrics,
        native_attention=not sol_attention_selected(options),
    )
    patches = dict(local.get("patches_replace") or {})
    blocks = dict(patches.get("dit") or {})
    patches["dit"] = blocks
    local["patches_replace"] = patches
    cached = {}

    def wrap(layer, previous):
        def call(args, extra):
            img = args["img"]
            if layer == 0:
                # Preserve native MiniMax-H3 visual-conditioning semantics for
                # the authoritative target-grid prefix.  The low-grid carrier
                # prefix is never presented to the transformer as exact context.
                aug = float(native.VISUAL_COND_TIMESTEP)
                prefix = owner.prefix.to(device=img.device, dtype=torch.float32)
                prefix_noise = owner.prefix_noise.to(device=img.device, dtype=torch.float32)
                prefix_rows = native.patchify_video(aug * prefix + (1.0 - aug) * prefix_noise)
                prefix_embed = inner.video_patch_proj(prefix_rows).to(img)
                if band is None:
                    img = torch.cat((img[:video_start], prefix_embed, img[video_start + carrier_prefix_rows :]))
                else:
                    # Band rows are the sampler's own target-grid state; tail rows
                    # are gathered from each frame's reduced-grid storage window.
                    img = torch.cat(
                        (
                            img[:video_start],
                            prefix_embed,
                            img[video_start + band.protected_rows : video_start + band.prefix_rows],
                            img.index_select(0, tail_rows),
                        )
                    )
                metrics.increment("partitioned_transformer_calls")
                if exact_context_range is not None:
                    metrics.increment("partitioned_exact_visual_prefix_calls")
                    metrics.event(
                        "partitioned_exact_visual_prefix",
                        policy=POLICY,
                        stage=options.get("h3_flow_stage"),
                        context_range=exact_context_range,
                        target_hw=exact_context.target_hw,
                        prefix_t=exact_context.prefix_t,
                        exact_prefix_unresampled=True,
                        prefix_time_colocated=True,
                        native_visual_condition_augmentation=True,
                        generated_video_streams=1,
                        video_recurrence_grid=owner.target_hw,
                        cross_grid_video_temporal_taps=False,
                        extra_h3_nfe=0,
                        extra_sampler_lifetimes=0,
                    )
                metrics.event(
                    "partitioned_exact_prefix_transformer",
                    stage=options.get("h3_flow_stage"),
                    native_sequence_rows=int(layout.seq_len),
                    partitioned_sequence_rows=int(partitioned_layout.seq_len),
                    video_start=int(video_start),
                    temporal=int(plan.temporal),
                    prefix_t=int(plan.prefix_t),
                    protected_prefix_t=int(owner.prefix_t),
                    attention_head_t=int(
                        plan.prefix_t if runtime.attention_head_t is None else runtime.attention_head_t
                    ),
                    target_band_t=int(band.band_t) if band is not None else 0,
                    native_carrier_grid=(
                        PARTITIONED_NATIVE_CARRIER_TARGET if band is not None else PARTITIONED_NATIVE_CARRIER_SOURCE
                    ),
                    source_rows_per_frame=int(plan.source_rows),
                    target_rows_per_frame=int(plan.target_rows),
                    prefix_log_key_measure=float(partition_plan.prefix_log_key_measure),
                    semantic_digest=partition_contract["semantic_digest"],
                    prefix_exact_latent_resized=False,
                    prefix_native_inpaint_augmentation=True,
                    prefix_visual_cond_timestep=aug,
                    prefix_target_grid_rope=True,
                    suffix_source_grid_rope=True,
                    low_suffix_real_latent=options.get("h3_flow_stage") != "high",
                    native_target_suffix=options.get("h3_flow_stage") == "high",
                    vdn_external_sequence_api=VDN_PARTITIONED_SEQUENCE_API,
                    vdn_temporal_carrier_policy=runtime.vdn_temporal_carrier_policy,
                    vdn_temporal_carrier_numerical_digest=(
                        runtime.vdn_temporal_carrier_contract["numerical_digest"]
                        if runtime.vdn_temporal_carrier_contract is not None
                        else None
                    ),
                    sol_single_union=sol_attention_selected(options),
                    deprecated_mixed_grid_contract_active=False,
                    audio_position_domain=str(runtime.audio_position_domain),
                    audio_position_policy_active=(position_policy is not None),
                    audio_position_policy_signature=(
                        position_policy.signature if position_policy is not None else None
                    ),
                    audio_position_stage_owner_generation=(
                        position_policy.stage_owner_generation if position_policy is not None else None
                    ),
                    audio_position_target_spatial_endpoints=(
                        position_policy.target_audio_spatial_endpoints if position_policy is not None else None
                    ),
                    audio_position_source_spatial_endpoints=(
                        position_policy.source_audio_spatial_endpoints if position_policy is not None else None
                    ),
                    audio_position_temporal_equal=(
                        position_policy.audio_temporal_equal if position_policy is not None else None
                    ),
                    audio_position_temporal_digest=(
                        position_policy.temporal_digest if position_policy is not None else None
                    ),
                    audio_position_non_audio_before_digest=(
                        position_policy.non_audio_before_digest if position_policy is not None else None
                    ),
                    audio_position_non_audio_after_digest=(
                        position_policy.non_audio_after_digest if position_policy is not None else None
                    ),
                    prefix_rope_position_digest=(
                        position_policy.prefix_rope_position_digest if position_policy is not None else None
                    ),
                    suffix_rope_position_digest=(
                        position_policy.suffix_rope_position_digest if position_policy is not None else None
                    ),
                    position_digest=(position_policy.position_digest if position_policy is not None else None),
                )
                if runtime.audio_position_domain == PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE:
                    metrics.increment("partitioned_audio_position_source_carrier_block0_calls")
            if len(img) != partitioned_layout.seq_len:
                raise RuntimeError("partitioned exact-prefix transformer row count mismatch")
            if "rope" not in cached:
                cached["rope"] = native.rope_rotation_table(
                    inner.rope_freqs(positions, img.device),
                    img.dtype,
                )
            # Core builds this table once per forward. Validate its protected
            # prefix once, rather than reading a CUDA scalar at every block.
            if cached.get("mod_source") is not args["mod_segments"]:
                cached["mod_segments"] = partitioned_mod_segments(
                    args["mod_segments"],
                    plan,
                    video_start,
                    video_end,
                )
                cached["mod_source"] = args["mod_segments"]
                metrics.increment("partitioned_modulation_validations")
            forwarded = dict(args)
            forwarded.update(
                img=img,
                layout=partitioned_layout,
                rope_freqs=cached["rope"],
                # Replacement patches receive their own video index tensor,
                # matching the previous per-block expansion's write isolation.
                mod_segments=[
                    (start, stop, row.clone() if start == video_start else row)
                    for start, stop, row in cached["mod_segments"]
                ],
            )
            forwarded["transformer_options"] = _partitioned_transformer_options(
                args["transformer_options"],
                partitioned_layout,
                partition_contract,
                metrics,
                runtime=runtime,
                block_index=layer,
            )
            block_extra = extra if sol_attention_selected(options) else partitioned_block_extra(extra)
            output = previous(forwarded, block_extra) if previous else block_extra["original_block"](forwarded)
            result = output["img"]
            if result.shape != img.shape:
                raise RuntimeError("partitioned exact-prefix transformer returned incompatible hidden state")
            if layer == len(inner.blocks) - 1:
                if band is None:
                    # Prefix transformer outputs are context-only.  Native low-grid
                    # unpatchify receives only genuine low-grid suffix carrier rows.
                    result = torch.cat(
                        (
                            result[:video_start],
                            result.new_zeros((carrier_prefix_rows, result.shape[1])),
                            result[video_start + plan.prefix_rows :],
                        )
                    )
                else:
                    # Return to Core's native target-grid rows: the protected prefix
                    # and the tail padding stay zero, band rows return in place, and
                    # tail rows return to their reduced-grid storage window.
                    native_rows = result.new_zeros((int(layout.seq_len), result.shape[1]))
                    native_rows[:video_start] = result[:video_start]
                    native_rows[video_start + band.protected_rows : video_start + band.prefix_rows] = result[
                        video_start + band.protected_rows : video_start + band.prefix_rows
                    ]
                    native_rows.index_copy_(0, tail_rows, result[video_start + band.prefix_rows :])
                    result = native_rows
                output = {**output, "img": result}
            return output

        return call

    for layer in range(len(inner.blocks)):
        blocks[("double_block", layer)] = wrap(layer, blocks.get(("double_block", layer)))
    output = executor(
        x,
        timestep,
        context,
        local,
        minimax_payload=payload,
        **kwargs,
    )
    video = output[0].clone()
    video[:, :, : owner.prefix_t] = 0
    if band is not None:
        video[:, :, band.head_t :, band.source_h :] = 0
        video[:, :, band.head_t :, : band.source_h, band.source_w :] = 0
    return [video, output[1]]


__all__ = [
    "DOMAIN_STREAM_SOURCE",
    "DOMAIN_STREAM_TARGET",
    "DOMAIN_UNIFORM_IDENTITY",
    "PARTITIONED_BLOCK_INDEX_KEY",
    "PARTITIONED_WRAPPER_KEY",
    "VDN_PARTITIONED_SEQUENCE_API",
    "VDN_PARTITIONED_SEQUENCE_MODE",
    "DomainUniformStream",
    "make_partitioned_attention_override",
    "partitioned_diffusion_wrapper",
]
