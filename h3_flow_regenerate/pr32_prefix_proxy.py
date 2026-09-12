"""PR #32 diagnostic: test high-stage protected-prefix representation causally.

00394 localized the remaining framing drift to the first high-stage H3 denoised
prediction. The clean handoff itself is geometrically consistent, while the
native H3 prediction contracts the continuation before Flow guidance or sampler
updates can create the visible shift.

This temporary diagnostic keeps the caller-owned sampler state, mask, noise and
returned protected prefix unchanged. It changes only the protected video values
seen by actual high-stage H3 predictions. Native inpaint has already produced
``aug * exact_prefix + (1 - aug) * caller_noise`` when Flow's PREDICT_NOISE
wrapper runs, so the model-only proxy needs only
``aug * (tone_aligned_learned_prefix - exact_prefix)`` on protected rows.
KSamplerX0Inpaint still restores the original caller latent on output. No model,
VAE, RNG or solver call is added.
"""

from __future__ import annotations

import contextvars
import math
from dataclasses import dataclass
from typing import Any

import torch

from . import runtime as _runtime
from .geometry import pack_streams, unpack_streams

_MODE = "learned_prefix_tone_aligned_model_input_v1"


@dataclass(slots=True)
class _PrefixProxyPayload:
    exact_prefix: torch.Tensor
    proxy_prefix: torch.Tensor
    video_shape: tuple[int, ...]
    prefix_t: int
    tone_bias_rms: float
    proxy_clean_delta_rms: float
    bias_deviation_rms: float
    bridge_reason: str


_payload_var: contextvars.ContextVar[_PrefixProxyPayload | None] = contextvars.ContextVar(
    "h3_flow_pr32_prefix_proxy", default=None
)
_ORIGINAL_APPLY_SUFFIX_REPRESENTATION_BRIDGE = _runtime.apply_suffix_representation_bridge
_ORIGINAL_DISABLED_SUFFIX_REPRESENTATION_BRIDGE_METRICS = (
    _runtime.disabled_suffix_representation_bridge_metrics
)
_ORIGINAL_FLOW_PREDICT_WRAPPER = _runtime.flow_predict_wrapper


def _rms(value: torch.Tensor) -> float:
    result = float(value.float().square().mean().sqrt().detach().cpu().item())
    if not math.isfinite(result):
        raise RuntimeError("PR32 prefix proxy produced a non-finite RMS")
    return result


def _capture_candidate(
    learned_clean_video: torch.Tensor,
    exact_prefix: torch.Tensor,
    corrected_clean_video: torch.Tensor,
    metrics: dict[str, Any],
) -> tuple[_PrefixProxyPayload | None, dict[str, Any]]:
    """Build a model-input proxy only from the already-authorized tone rebase."""

    report = {
        "suffix_representation_bridge_high_prefix_proxy_mode": _MODE,
        "suffix_representation_bridge_high_prefix_proxy_ready": False,
        "suffix_representation_bridge_high_prefix_proxy_reason": "bridge_not_eligible",
        "suffix_representation_bridge_high_prefix_proxy_clean_delta_rms": 0.0,
        "suffix_representation_bridge_high_prefix_proxy_bias_deviation_rms": 0.0,
    }
    if not bool(metrics.get("suffix_representation_bridge_tone_bias_accepted")):
        report["suffix_representation_bridge_high_prefix_proxy_reason"] = (
            "tone_rebase_not_authorized"
        )
        return None, report
    if bool(metrics.get("suffix_representation_bridge_geometry_accepted")):
        report["suffix_representation_bridge_high_prefix_proxy_reason"] = (
            "geometry_rebase_would_confound_proxy"
        )
        return None, report
    geometry_reason = str(
        metrics.get("suffix_representation_bridge_geometry_reason", "")
    )
    if geometry_reason != "native_boundary_replacement_geometry_consistent":
        report["suffix_representation_bridge_high_prefix_proxy_reason"] = (
            "clean_geometry_not_proven_consistent"
        )
        return None, report

    prefix_t = int(exact_prefix.shape[2])
    if prefix_t <= 0 or prefix_t >= int(learned_clean_video.shape[2]):
        raise ValueError(
            "PR32 prefix proxy requires a non-empty learned prefix and suffix"
        )
    if tuple(corrected_clean_video.shape) != tuple(learned_clean_video.shape):
        raise ValueError("PR32 prefix proxy bridge outputs changed video geometry")

    learned_suffix = learned_clean_video[:, :, prefix_t:].float()
    corrected_suffix = corrected_clean_video[:, :, prefix_t:].float()
    suffix_delta = corrected_suffix - learned_suffix
    tone_bias = suffix_delta.mean(dim=(2, 3, 4), keepdim=True)
    deviation_rms = _rms(suffix_delta - tone_bias)
    # v6's accepted tone path is a constant per-channel addition. Do not proxy
    # any future bridge output carrying another correction under the same flag.
    tolerance = max(2.0e-6, 2.0e-5 * max(_rms(tone_bias), 1.0))
    if deviation_rms > tolerance:
        report.update(
            suffix_representation_bridge_high_prefix_proxy_reason=(
                "authorized_bridge_not_pure_tone"
            ),
            suffix_representation_bridge_high_prefix_proxy_bias_deviation_rms=deviation_rms,
        )
        return None, report

    learned_prefix = learned_clean_video[:, :, :prefix_t].detach().float()
    exact = exact_prefix.detach().to(
        device=learned_prefix.device, dtype=torch.float32
    )
    proxy_prefix = learned_prefix + tone_bias
    proxy_delta_rms = _rms(proxy_prefix - exact)
    payload = _PrefixProxyPayload(
        exact_prefix=exact,
        proxy_prefix=proxy_prefix.detach(),
        video_shape=tuple(int(value) for value in learned_clean_video.shape),
        prefix_t=prefix_t,
        tone_bias_rms=_rms(tone_bias),
        proxy_clean_delta_rms=proxy_delta_rms,
        bias_deviation_rms=deviation_rms,
        bridge_reason=geometry_reason,
    )
    report.update(
        suffix_representation_bridge_high_prefix_proxy_ready=True,
        suffix_representation_bridge_high_prefix_proxy_reason=(
            "tone_aligned_learned_prefix_ready"
        ),
        suffix_representation_bridge_high_prefix_proxy_clean_delta_rms=proxy_delta_rms,
        suffix_representation_bridge_high_prefix_proxy_bias_deviation_rms=deviation_rms,
    )
    return payload, report


def _apply_suffix_representation_bridge_capture(
    learned_clean_video: torch.Tensor,
    exact_prefix: torch.Tensor,
    *,
    requested: bool,
):
    # Clear first so an exception or ineligible new run cannot reuse a prior GPU
    # payload.
    _payload_var.set(None)
    corrected, metrics = _ORIGINAL_APPLY_SUFFIX_REPRESENTATION_BRIDGE(
        learned_clean_video,
        exact_prefix,
        requested=requested,
    )
    metrics = dict(metrics)
    payload, report = _capture_candidate(
        learned_clean_video, exact_prefix, corrected, metrics
    )
    _payload_var.set(payload)
    metrics.update(report)
    return corrected, metrics


def _disabled_suffix_representation_bridge_metrics_capture(
    *, prefix_t: int, requested: bool = False
):
    _payload_var.set(None)
    metrics = dict(
        _ORIGINAL_DISABLED_SUFFIX_REPRESENTATION_BRIDGE_METRICS(
            prefix_t=prefix_t,
            requested=requested,
        )
    )
    metrics.update(
        suffix_representation_bridge_high_prefix_proxy_mode=_MODE,
        suffix_representation_bridge_high_prefix_proxy_ready=False,
        suffix_representation_bridge_high_prefix_proxy_reason=(
            "representation_bridge_disabled"
        ),
        suffix_representation_bridge_high_prefix_proxy_clean_delta_rms=0.0,
        suffix_representation_bridge_high_prefix_proxy_bias_deviation_rms=0.0,
    )
    return metrics


def _native_visual_cond_timestep() -> float:
    # Resolve from installed native H3 at execution time; do not duplicate 0.999
    # as an independent behavioral constant.
    import comfy.ldm.minimax.model as native

    value = float(native.VISUAL_COND_TIMESTEP)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise RuntimeError(
            "native MiniMax H3 visual conditioning timestep is invalid"
        )
    return value


def _proxy_packed_model_input(
    packed: torch.Tensor,
    contract: dict[str, Any],
    payload: _PrefixProxyPayload,
    *,
    visual_cond_timestep: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    shapes = [
        tuple(int(value) for value in shape) for shape in contract["shapes"]
    ]
    if len(shapes) != 2:
        raise RuntimeError(
            "PR32 prefix proxy requires packed MiniMax H3 video/audio shapes"
        )
    video, audio = unpack_streams(packed, shapes)
    if tuple(video.shape) != payload.video_shape:
        raise RuntimeError(
            "PR32 prefix proxy payload geometry does not match the active "
            "high-stage model input"
        )
    prefix_t = int(contract["prefix_t"])
    if prefix_t != payload.prefix_t or not 0 < prefix_t < int(video.shape[2]):
        raise RuntimeError("PR32 prefix proxy payload temporal boundary is stale")
    aug = float(visual_cond_timestep)
    if not math.isfinite(aug) or not 0.0 <= aug <= 1.0:
        raise ValueError(
            "PR32 prefix proxy visual conditioning timestep must be in [0, 1]"
        )

    exact = payload.exact_prefix.to(device=video.device, dtype=torch.float32)
    proxy = payload.proxy_prefix.to(device=video.device, dtype=torch.float32)
    delta = (proxy - exact) * aug
    proxied_video = video.clone()
    proxied_video[:, :, :prefix_t] = (
        proxied_video[:, :, :prefix_t].float() + delta
    ).to(proxied_video.dtype)
    proxied, proxied_shapes = pack_streams((proxied_video, audio))
    if proxied_shapes != shapes:
        raise RuntimeError("PR32 prefix proxy changed packed H3 geometry")

    prefix_delta = (
        proxied_video[:, :, :prefix_t].float()
        - video[:, :, :prefix_t].float()
    )
    suffix_delta = (
        proxied_video[:, :, prefix_t:].float()
        - video[:, :, prefix_t:].float()
    )
    audio_delta = unpack_streams(proxied - packed, shapes)[1]
    telemetry = {
        "prefix_proxy_mode": _MODE,
        "prefix_proxy_visual_cond_timestep": aug,
        "prefix_proxy_prefix_t": prefix_t,
        "prefix_proxy_clean_delta_rms": payload.proxy_clean_delta_rms,
        "prefix_proxy_tone_bias_rms": payload.tone_bias_rms,
        "prefix_proxy_bias_deviation_rms": payload.bias_deviation_rms,
        "prefix_proxy_input_delta_rms": _rms(prefix_delta),
        "prefix_proxy_suffix_delta_rms": _rms(suffix_delta),
        "prefix_proxy_audio_delta_rms": _rms(audio_delta),
        "prefix_proxy_bridge_reason": payload.bridge_reason,
    }
    if (
        telemetry["prefix_proxy_suffix_delta_rms"] != 0.0
        or telemetry["prefix_proxy_audio_delta_rms"] != 0.0
    ):
        raise RuntimeError(
            "PR32 prefix proxy modified generated suffix or audio input"
        )
    return proxied, telemetry


class _ModelOnlyPrefixProxyExecutor:
    def __init__(
        self,
        executor: Any,
        contract: dict[str, Any],
        payload: _PrefixProxyPayload,
    ) -> None:
        self._executor = executor
        self.class_obj = executor.class_obj
        self._contract = contract
        self._payload = payload
        self.telemetry: dict[str, Any] | None = None
        self.proxied_input: torch.Tensor | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._executor, name)

    def __call__(self, x, timestep, model_options=None, seed=None):
        proxied, telemetry = _proxy_packed_model_input(
            x,
            self._contract,
            self._payload,
            visual_cond_timestep=_native_visual_cond_timestep(),
        )
        self.telemetry = telemetry
        self.proxied_input = proxied
        return self._executor(proxied, timestep, model_options, seed)


def flow_predict_wrapper_with_prefix_proxy(
    executor, x, timestep, model_options=None, seed=None
):
    """Proxy only actual inner H3 calls while Flow keeps the real sampler state."""

    transformer = (model_options or {}).get("transformer_options") or {}
    contract = transformer.get(_runtime.HIGH_STAGE_DIAGNOSTIC_KEY)
    payload = _payload_var.get()
    if not isinstance(contract, dict) or payload is None:
        return _ORIGINAL_FLOW_PREDICT_WRAPPER(
            executor, x, timestep, model_options, seed
        )

    # Spectrum forecast execution substitutes for the transformer and is not the
    # causal mechanism under test. Let it consume the untouched model input; its
    # changed history may still reflect earlier proxied actual H3 predictions.
    if transformer.get(_runtime.SPECTRUM_ACTUAL_KEY) is False:
        return _ORIGINAL_FLOW_PREDICT_WRAPPER(
            executor, x, timestep, model_options, seed
        )

    adapted = _ModelOnlyPrefixProxyExecutor(executor, contract, payload)
    result = _ORIGINAL_FLOW_PREDICT_WRAPPER(
        adapted, x, timestep, model_options, seed
    )
    if adapted.telemetry is None or adapted.proxied_input is None:
        raise RuntimeError(
            "PR32 high-stage prefix proxy was armed but did not reach the H3 executor"
        )

    binding = _runtime._resolve_binding(executor.class_obj)
    holder = contract.get("holder")
    call_fields = getattr(holder, "last_call", None)
    if binding is not None:
        fields = dict(call_fields) if isinstance(call_fields, dict) else {}
        binding.metrics.event(
            "mixed_grid_high_prefix_proxy_input",
            **fields,
            **adapted.telemetry,
            **_runtime.packed_boundary_fields(
                adapted.proxied_input,
                contract,
                field_prefix="proxy_model_input_",
            ),
        )

    phases = tuple(contract.get("phases") or ())
    call_index = int(getattr(holder, "call_index", 0))
    if phases and call_index >= len(phases):
        _payload_var.set(None)
    return result


def install() -> None:
    """Install the temporary PR #32 overlay before comfy_compat imports wrappers."""

    if getattr(_runtime.flow_predict_wrapper, "_h3_pr32_prefix_proxy", False):
        return
    _runtime.apply_suffix_representation_bridge = (
        _apply_suffix_representation_bridge_capture
    )
    _runtime.disabled_suffix_representation_bridge_metrics = (
        _disabled_suffix_representation_bridge_metrics_capture
    )
    flow_predict_wrapper_with_prefix_proxy._h3_pr32_prefix_proxy = True
    _runtime.flow_predict_wrapper = flow_predict_wrapper_with_prefix_proxy


install()
