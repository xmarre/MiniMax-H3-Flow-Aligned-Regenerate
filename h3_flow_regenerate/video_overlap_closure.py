"""Close temporary video-prefix release coherently before high sampling ends."""

from __future__ import annotations

import contextlib
from typing import Any

import torch

from .geometry import unpack_streams

VIDEO_OVERLAP_CLOSURE_POLICY = "video_prefix_release_then_exact_tail_v1"
_WRAPPER_KEY = "h3_flow_video_overlap_closure_v1"
_APPLY_MODEL = "apply_model"  # Core WrappersMP.APPLY_MODEL public identifier.
_EXACT_TAIL_STEPS = 2


def validate_video_overlap_closure_options(options: dict[str, Any]) -> None:
    """Reject conflicting ownership before entering any sampler lifetime."""

    transformer = options.get("transformer_options", {})
    wrappers = transformer.get("wrappers", {})
    if _WRAPPER_KEY in wrappers.get(_APPLY_MODEL, {}):
        raise RuntimeError("stale or nested video overlap closure is unsupported")
    if options.get("denoise_mask_function") is not None:
        raise RuntimeError("video overlap closure cannot share Core denoise_mask_function ownership")


class VideoOverlapClosure:
    """One high-lifetime mask owner shared by Core's sampler and APPLY_MODEL.

    The spatial/temporal release ramp decays linearly in sigma and reaches zero
    at the start of the final two scheduled evaluations. Those evaluations see
    the same carried video context that final exact restoration will return.
    Short schedules use exact conditioning throughout. Audio and generated video
    mask values are never modified. Working buffers are released on exit even
    if Core retains a copied wrapper registry.
    """

    def __init__(self, runtime_mask, shapes, *, prefix_t, tokens, sigmas):
        self.shapes = shapes
        self.prefix_t = int(prefix_t)
        self.tokens = min(int(tokens), self.prefix_t)
        self.start = self.prefix_t - self.tokens
        sigma_values = sigmas.detach().to(device="cpu").tolist()
        self.start_sigma = float(sigma_values[0])
        self.close_sigma = float(sigma_values[max(0, len(sigma_values) - 1 - _EXACT_TAIL_STEPS)])
        self.release_enabled = self.start_sigma > self.close_sigma
        video, _ = unpack_streams(runtime_mask, shapes)
        self.base_ramp = video[0, 0, self.start : self.prefix_t, 0, 0].detach().float().clone()
        self.current_mask = None
        self.video_condition = None
        self.current_sigma = None
        self.last_model_sigma = None
        self.mask_calls = 0
        self.model_calls = 0
        self.closed = False

    def mask(self, sigma, denoise_mask, *, extra_options=None):
        del extra_options
        if self.closed:
            raise RuntimeError("video overlap closure used outside its sampler lifetime")
        if self.current_mask is None:
            self.current_mask = denoise_mask.clone()
            video, _ = unpack_streams(self.current_mask, self.shapes)
            self.video_condition = video[:1, :1].contiguous().clone()
            self.base_ramp = self.base_ramp.to(device=video.device)
        elif (
            self.current_mask.shape != denoise_mask.shape
            or self.current_mask.device != denoise_mask.device
            or self.current_mask.dtype != denoise_mask.dtype
        ):
            raise RuntimeError("video overlap closure sampler-mask geometry/device/dtype changed")
        self.current_sigma = sigma.detach().reshape(-1)[:1]
        if self.release_enabled:
            factor = ((self.current_sigma - self.close_sigma) / (self.start_sigma - self.close_sigma)).clamp(0, 1)
            ramp = torch.ceil(self.base_ramp * factor.to(self.base_ramp) * 256.0) / 256.0
        else:
            ramp = torch.zeros_like(self.base_ramp)
        video, _ = unpack_streams(self.current_mask, self.shapes)
        ramp = ramp.to(video).view(1, 1, -1, 1, 1)
        video[:, :, self.start : self.prefix_t] = ramp
        self.video_condition[:, :, self.start : self.prefix_t] = ramp
        self.mask_calls += 1
        return self.current_mask

    def apply_model(
        self, executor, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options=None, **kwargs
    ):
        if self.closed or self.video_condition is None:
            raise RuntimeError("video overlap model entry has no current sampler-mask publication")
        if not torch.equal(t.detach().reshape(-1)[:1].to(self.current_sigma), self.current_sigma):
            raise RuntimeError("video overlap sampler/model sigma publication differs")
        prior = kwargs.get("denoise_mask")
        if not torch.is_tensor(prior) or tuple(prior.shape[1:]) != tuple(self.video_condition.shape[1:]):
            raise RuntimeError("video overlap model-mask condition geometry changed")
        local = dict(kwargs)
        local["denoise_mask"] = self.video_condition.to(prior).expand_as(prior)
        self.model_calls += 1
        self.last_model_sigma = self.current_sigma.clone()
        return executor(x, t, c_concat, c_crossattn, control, transformer_options, **local)

    def close(self):
        self.closed = True
        self.base_ramp = None
        self.current_mask = None
        self.video_condition = None
        self.current_sigma = None
        self.last_model_sigma = None


@contextlib.contextmanager
def video_overlap_closure(guider, runtime_mask, shapes, *, prefix_t, tokens, sigmas, metrics):
    """Install only call-local public Core hooks and restore their prior owners."""

    if tokens <= 0 or prefix_t <= 0:
        yield None
        return
    options = guider.model_options
    validate_video_overlap_closure_options(options)
    closure = VideoOverlapClosure(runtime_mask, shapes, prefix_t=prefix_t, tokens=tokens, sigmas=sigmas)
    had_transformer = "transformer_options" in options
    transformer = options.setdefault("transformer_options", {})
    had_wrappers = "wrappers" in transformer
    wrappers = transformer.setdefault("wrappers", {})
    had_apply = _APPLY_MODEL in wrappers
    apply_wrappers = wrappers.setdefault(_APPLY_MODEL, {})

    # Plain closures retain this one lifetime owner when Core clones options.
    # Cloning a bound method could otherwise duplicate its mutable publications.
    def mask_function(*args, **kwargs):
        return closure.mask(*args, **kwargs)

    def apply_wrapper(*args, **kwargs):
        return closure.apply_model(*args, **kwargs)

    had_hook = "denoise_mask_function" in options
    old_hook = options.get("denoise_mask_function")
    options["denoise_mask_function"] = mask_function
    apply_wrappers[_WRAPPER_KEY] = [apply_wrapper]
    failed = True
    terminal = {}
    try:
        yield closure
        if closure.mask_calls == 0 or closure.model_calls == 0:
            raise RuntimeError("video overlap closure observed no native sampler/model entries")
        # One bounded host transfer at the sampler boundary, not per block or
        # per forecast. The receipt observes actual execution, not just its plan.
        final_sigma, model_sigma, prefix_mask_max = (
            torch.cat(
                (
                    closure.current_sigma.double(),
                    closure.last_model_sigma.to(closure.current_sigma).double(),
                    closure.video_condition[:, :, :prefix_t].abs().amax().reshape(1).double(),
                )
            )
            .to(device="cpu")
            .tolist()
        )
        terminal = {
            "final_sampler_sigma": final_sigma,
            "last_core_apply_model_sigma": model_sigma,
            "final_prefix_mask_max": prefix_mask_max,
            "exact_model_context_before_completion": model_sigma <= closure.close_sigma and prefix_mask_max == 0,
        }
        if not terminal["exact_model_context_before_completion"]:
            raise RuntimeError("video overlap did not reach exact model context before completion")
        failed = False
    finally:
        if had_hook:
            options["denoise_mask_function"] = old_hook
        else:
            options.pop("denoise_mask_function", None)
        apply_wrappers.pop(_WRAPPER_KEY, None)
        if not had_apply and not apply_wrappers:
            wrappers.pop(_APPLY_MODEL, None)
        if not had_wrappers and not wrappers:
            transformer.pop("wrappers", None)
        if not had_transformer and not transformer:
            options.pop("transformer_options", None)
        fields = dict(
            policy=VIDEO_OVERLAP_CLOSURE_POLICY,
            requested_tokens=int(tokens),
            applied_tokens=(closure.tokens if closure.release_enabled else 0),
            video_prefix_tokens=int(prefix_t),
            start_sigma=closure.start_sigma,
            close_sigma=closure.close_sigma,
            exact_tail_steps=_EXACT_TAIL_STEPS,
            release_enabled=closure.release_enabled,
            sampler_mask_calls=closure.mask_calls,
            core_apply_model_entries=closure.model_calls,
            model_mask_matches_sampler=closure.model_calls > 0,
            failed=failed,
            extra_h3_nfe=0,
            extra_sampler_lifetimes=0,
            extra_vae_calls=0,
            **terminal,
        )
        closure.close()
        metrics.event("partitioned_video_overlap_closure", **fields)
