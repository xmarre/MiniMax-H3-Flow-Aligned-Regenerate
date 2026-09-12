"""PR #32 diagnostic: keep high-stage suffix on the source spatial RoPE coordinate law.

00395 falsified the protected-prefix value/representation proxy as the primary
cause of the persistent framing contraction. The first raw high-stage H3
prediction still contracts the continuation even though clean transfer geometry
is consistent and the model-only prefix values were changed substantially.

The remaining discrete boundary is positional: Mixed-Grid's generated suffix
uses the source spatial grid before handoff and native target-grid spatial RoPE
after handoff. This temporary diagnostic keeps target-grid latent values, token
count, timesteps, masks, prefix/noise, audio and layout ownership unchanged, but
for *actual* high-stage H3 evaluations only it evaluates generated-suffix
spatial RoPE on the source grid's native area-normalized coordinate law at the
target-grid sampling density. Protected-prefix and every non-video RoPE row
remain bit-identical to native. Spectrum forecast calls are not directly
modified. No model/VAE/RNG/NFE is added.
"""

from __future__ import annotations

import contextvars
import math
from dataclasses import dataclass
from typing import Any

import torch

from . import runtime as _runtime

_MODE = "source_coordinate_suffix_rope_v1"
_ORIGINAL_FLOW_PREDICT_WRAPPER = _runtime.flow_predict_wrapper
_ORIGINAL_BUILD_MIXED_GRID_PLAN = _runtime.build_mixed_grid_plan


@dataclass(frozen=True, slots=True)
class _RopePlan:
    source_hw: tuple[int, int]
    target_hw: tuple[int, int]
    prefix_t: int


_plan_var: contextvars.ContextVar[_RopePlan | None] = contextvars.ContextVar(
    "h3_flow_pr32_suffix_rope_plan", default=None
)


def _rms(value: torch.Tensor) -> float:
    result = float(value.float().square().mean().sqrt().detach().cpu().item())
    if not math.isfinite(result):
        raise RuntimeError("PR32 suffix RoPE diagnostic produced a non-finite RMS")
    return result


def build_mixed_grid_plan_capture(*args, **kwargs):
    """Capture only geometry already authorized by the active Mixed-Grid plan."""

    _plan_var.set(None)
    plan = _ORIGINAL_BUILD_MIXED_GRID_PLAN(*args, **kwargs)
    _plan_var.set(
        _RopePlan(
            source_hw=(int(kwargs["source_h"]), int(kwargs["source_w"])),
            target_hw=tuple(map(int, plan.target_hw)),
            prefix_t=int(plan.prefix_t),
        )
    )
    return plan


def _native_axis(dim: int, sqrt_area: float, samples: int) -> torch.Tensor:
    ratio = float(dim) / float(sqrt_area)
    return (torch.arange(samples, dtype=torch.float64) * (ratio / samples) + (1.0 - ratio) / 2.0) * 32.0


def _source_coordinate_target_frame(
    *,
    source_hw: tuple[int, int],
    target_hw: tuple[int, int],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Sample the source native coordinate law at target-grid density.

    Native H3 uses, per spatial axis,
      32 * ((1-ratio)/2 + ratio * index / grid_size),
    where ratio is axis/sqrt(HW). At handoff the target grid changes both the
    number of samples and, when rounding changes aspect ratio, ``ratio``. This
    diagnostic changes only the latter back to the source value while retaining
    the target number of rows.
    """

    source_h, source_w = map(int, source_hw)
    target_h, target_w = map(int, target_hw)
    if min(source_h, source_w, target_h, target_w) < 2 or any(
        value % 2 for value in (source_h, source_w, target_h, target_w)
    ):
        raise ValueError("PR32 suffix RoPE diagnostic requires positive even H3 latent axes")
    if source_h > target_h or source_w > target_w or (source_h, source_w) == (target_h, target_w):
        raise ValueError("PR32 suffix RoPE diagnostic requires a strictly reduced source grid")

    source_area = math.sqrt(source_h * source_w)
    target_area = math.sqrt(target_h * target_w)
    source_h_ratio = source_h / source_area
    source_w_ratio = source_w / source_area
    target_h_ratio = target_h / target_area
    target_w_ratio = target_w / target_area
    target_gh, target_gw = target_h // 2, target_w // 2

    h_axis = _native_axis(source_h, source_area, target_gh)
    w_axis = _native_axis(source_w, source_area, target_gw)
    native_target_h_axis = _native_axis(target_h, target_area, target_gh)
    native_target_w_axis = _native_axis(target_w, target_area, target_gw)
    hh, ww = torch.meshgrid(h_axis, w_axis, indexing="ij")
    frame = torch.stack((hh.reshape(-1), ww.reshape(-1)), dim=-1)

    report = {
        "suffix_rope_proxy_mode": _MODE,
        "suffix_rope_proxy_source_hw": (source_h, source_w),
        "suffix_rope_proxy_target_hw": (target_h, target_w),
        "suffix_rope_proxy_h_coordinate_scale": source_h_ratio / target_h_ratio,
        "suffix_rope_proxy_w_coordinate_scale": source_w_ratio / target_w_ratio,
        "suffix_rope_proxy_source_h_ratio": source_h_ratio,
        "suffix_rope_proxy_source_w_ratio": source_w_ratio,
        "suffix_rope_proxy_target_h_ratio": target_h_ratio,
        "suffix_rope_proxy_target_w_ratio": target_w_ratio,
        "suffix_rope_proxy_h_first": float(h_axis[0]),
        "suffix_rope_proxy_h_last": float(h_axis[-1]),
        "suffix_rope_proxy_w_first": float(w_axis[0]),
        "suffix_rope_proxy_w_last": float(w_axis[-1]),
        "suffix_rope_proxy_native_target_h_first": float(native_target_h_axis[0]),
        "suffix_rope_proxy_native_target_h_last": float(native_target_h_axis[-1]),
        "suffix_rope_proxy_native_target_w_first": float(native_target_w_axis[0]),
        "suffix_rope_proxy_native_target_w_last": float(native_target_w_axis[-1]),
    }
    return frame, report


@dataclass(slots=True)
class _SuffixRopeContext:
    native_model: Any
    contract: dict[str, Any]
    plan: _RopePlan
    cached_rope: torch.Tensor | None = None
    telemetry: dict[str, Any] | None = None

    def rope_for(self, args: dict[str, Any]) -> torch.Tensor:
        if self.cached_rope is not None:
            return self.cached_rope

        native = __import__("comfy.ldm.minimax.model", fromlist=["model"])
        layout = args.get("layout")
        baseline = args.get("rope_freqs")
        image = args.get("img")
        if layout is None or baseline is None or not torch.is_tensor(image):
            raise RuntimeError("PR32 suffix RoPE diagnostic requires native H3 block layout/rope/image arguments")
        shapes = [tuple(int(value) for value in shape) for shape in self.contract.get("shapes", ())]
        if len(shapes) != 2:
            raise RuntimeError("PR32 suffix RoPE diagnostic contract is missing target AV geometry")
        target_h, target_w = map(int, shapes[0][-2:])
        prefix_t = int(self.contract["prefix_t"])
        temporal = int(shapes[0][2])
        if self.plan.target_hw != (target_h, target_w) or self.plan.prefix_t != prefix_t:
            raise RuntimeError("PR32 suffix RoPE diagnostic captured plan is stale for the active high stage")
        source_h, source_w = self.plan.source_hw
        target_rows = target_h * target_w // 4

        video_segments = [segment for segment in layout.segments if segment[2] == "video"]
        if len(video_segments) != 1:
            raise RuntimeError("PR32 suffix RoPE diagnostic requires exactly one native target video segment")
        va, vb, _ = video_segments[0]
        if vb - va != temporal * target_rows:
            raise RuntimeError("PR32 suffix RoPE diagnostic target video row count does not match contract geometry")
        if not 0 < prefix_t < temporal:
            raise RuntimeError("PR32 suffix RoPE diagnostic requires a non-empty prefix and suffix")

        source_frame, report = _source_coordinate_target_frame(
            source_hw=(source_h, source_w),
            target_hw=(target_h, target_w),
        )
        positions = layout.position_ids.detach().clone()
        original_positions = positions.clone()
        video_positions = positions[va:vb].reshape(temporal, target_rows, 3)
        source_frame = source_frame.to(device=video_positions.device, dtype=video_positions.dtype)
        video_positions[prefix_t:, :, 1:] = source_frame.unsqueeze(0)

        suffix_start = va + prefix_t * target_rows
        suffix_stop = vb
        if not torch.equal(positions[:suffix_start], original_positions[:suffix_start]):
            raise RuntimeError("PR32 suffix RoPE diagnostic modified protected-prefix/non-video positions")
        if not torch.equal(positions[suffix_start:suffix_stop, 0], original_positions[suffix_start:suffix_stop, 0]):
            raise RuntimeError("PR32 suffix RoPE diagnostic modified temporal positions")
        if not torch.equal(positions[suffix_stop:], original_positions[suffix_stop:]):
            raise RuntimeError("PR32 suffix RoPE diagnostic modified rows after target video")

        proxy_full = native.rope_rotation_table(
            self.native_model.rope_freqs(positions, image.device),
            image.dtype,
        )
        if tuple(proxy_full.shape) != tuple(baseline.shape):
            raise RuntimeError("PR32 suffix RoPE diagnostic changed native RoPE tensor shape")
        proxied = baseline.clone()
        proxied[:, suffix_start:suffix_stop] = proxy_full[:, suffix_start:suffix_stop]
        if not torch.equal(proxied[:, :suffix_start], baseline[:, :suffix_start]):
            raise RuntimeError("PR32 suffix RoPE diagnostic changed prefix/non-video RoPE rows")
        if suffix_stop < baseline.shape[1] and not torch.equal(proxied[:, suffix_stop:], baseline[:, suffix_stop:]):
            raise RuntimeError("PR32 suffix RoPE diagnostic changed trailing non-video RoPE rows")

        spatial_delta = positions[suffix_start:suffix_stop, 1:] - original_positions[suffix_start:suffix_stop, 1:]
        report.update(
            suffix_rope_proxy_prefix_t=prefix_t,
            suffix_rope_proxy_temporal=temporal,
            suffix_rope_proxy_target_rows_per_frame=target_rows,
            suffix_rope_proxy_modified_rows=suffix_stop - suffix_start,
            suffix_rope_proxy_position_delta_rms=_rms(spatial_delta),
            suffix_rope_proxy_position_delta_abs_max=float(spatial_delta.abs().max().detach().cpu().item()),
            suffix_rope_proxy_temporal_delta_abs_max=0.0,
            suffix_rope_proxy_prefix_rope_exact=True,
            suffix_rope_proxy_layout_unchanged=True,
        )
        self.telemetry = report
        self.cached_rope = proxied
        return proxied


def _resolve_native_h3_model(guider: Any) -> Any:
    base = getattr(guider, "inner_model", None)
    model = getattr(base, "diffusion_model", None)
    if model is None:
        raise RuntimeError("PR32 suffix RoPE diagnostic cannot resolve the active H3 diffusion model")
    if not hasattr(model, "blocks") or not hasattr(model, "rope_freqs"):
        raise RuntimeError("PR32 suffix RoPE diagnostic requires native MiniMax H3 block/RoPE APIs")
    return model


def _block_replacement(previous: Any, context: _SuffixRopeContext):
    def replace(args, extra):
        forwarded = dict(args)
        forwarded["rope_freqs"] = context.rope_for(args)
        if previous is not None:
            return previous(forwarded, extra)
        return extra["original_block"](forwarded)

    return replace


def _model_options_with_suffix_rope_proxy(
    model_options: dict[str, Any] | None,
    *,
    native_model: Any,
    contract: dict[str, Any],
    plan: _RopePlan,
) -> tuple[dict[str, Any], _SuffixRopeContext]:
    options = dict(model_options or {})
    transformer = dict(options.get("transformer_options") or {})
    patches_replace = dict(transformer.get("patches_replace") or {})
    dit = dict(patches_replace.get("dit") or {})
    context = _SuffixRopeContext(native_model=native_model, contract=contract, plan=plan)
    for index in range(len(native_model.blocks)):
        key = ("double_block", index)
        dit[key] = _block_replacement(dit.get(key), context)
    patches_replace["dit"] = dit
    transformer["patches_replace"] = patches_replace
    options["transformer_options"] = transformer
    return options, context


def flow_predict_wrapper_with_suffix_rope_proxy(executor, x, timestep, model_options=None, seed=None):
    """Change only generated-suffix spatial RoPE for actual high-stage H3 calls."""

    transformer = (model_options or {}).get("transformer_options") or {}
    contract = transformer.get(_runtime.HIGH_STAGE_DIAGNOSTIC_KEY)
    plan = _plan_var.get()
    if not isinstance(contract, dict) or plan is None:
        return _ORIGINAL_FLOW_PREDICT_WRAPPER(executor, x, timestep, model_options, seed)
    if transformer.get(_runtime.SPECTRUM_ACTUAL_KEY) is False:
        return _ORIGINAL_FLOW_PREDICT_WRAPPER(executor, x, timestep, model_options, seed)

    guider = executor.class_obj
    native_model = _resolve_native_h3_model(guider)
    proxied_options, context = _model_options_with_suffix_rope_proxy(
        model_options,
        native_model=native_model,
        contract=contract,
        plan=plan,
    )
    result = _ORIGINAL_FLOW_PREDICT_WRAPPER(executor, x, timestep, proxied_options, seed)
    if context.telemetry is None:
        raise RuntimeError("PR32 suffix RoPE diagnostic was armed but did not reach a native H3 block")

    binding = _runtime._resolve_binding(guider)
    holder = contract.get("holder")
    call_fields = getattr(holder, "last_call", None)
    if binding is not None:
        fields = dict(call_fields) if isinstance(call_fields, dict) else {}
        binding.metrics.event(
            "mixed_grid_high_suffix_rope_proxy",
            **fields,
            **context.telemetry,
        )
    return result


def install() -> None:
    if getattr(_runtime.flow_predict_wrapper, "_h3_pr32_suffix_rope_proxy", False):
        return
    _runtime.build_mixed_grid_plan = build_mixed_grid_plan_capture
    flow_predict_wrapper_with_suffix_rope_proxy._h3_pr32_suffix_rope_proxy = True
    _runtime.flow_predict_wrapper = flow_predict_wrapper_with_suffix_rope_proxy


install()
