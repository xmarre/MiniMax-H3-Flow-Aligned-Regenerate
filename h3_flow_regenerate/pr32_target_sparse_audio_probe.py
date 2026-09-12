"""PR #32 diagnostic: isolate Target-Sparse audio continuity at the handoff.

00403 shows a visually clean Target-Sparse Continuum boundary but an audible
chunk seam. Target-Sparse keeps every audio row while reducing the video token
stream during its early sampler lifetime, so the audio trajectory is still
predicted against an approximate video context.

For Continuum Target-Sparse runs only, this diagnostic performs two one-call
handoff probes from the exact same target-grid sampler state:

* one probe with the Target-Sparse transformer contract still active;
* one probe with the sparse contract removed, i.e. the full H3 transformer.

Only the audio x0 disagreement is used. The high-stage initial audio state is
rebased by the exact flow-interpolation identity

    x_sigma' = x_sigma + (1 - sigma) * (x0_full - x0_sparse)

while the caller's video noise/state remains bit-identical. The probes are
excluded from trajectory capture and are telemetry-only apart from that bounded
audio-state rebase. No video state, mask, conditioning, RoPE, scheduler state,
VDN/Sol/Spectrum ownership, or normal sampler schedule is changed.

This is temporary diagnostic work. It intentionally adds two H3 probe NFEs and
must not be promoted without decoded-media evidence.
"""

from __future__ import annotations

import contextlib
import copy
import math
from typing import Any

import torch

from . import runtime as _runtime
from .geometry import pack_streams, unpack_streams

_MODE = "target_sparse_audio_full_x0_rebase_v1"
_ORIGINAL_RUN_TARGET_SPARSE = _runtime._run_target_sparse_exact_prefix


def _rms(value: torch.Tensor) -> float:
    result = float(value.float().square().mean().sqrt().detach().cpu().item())
    if not math.isfinite(result):
        raise RuntimeError("PR32 Target-Sparse audio diagnostic produced a non-finite RMS")
    return result


def _abs_mean(value: torch.Tensor) -> float:
    result = float(value.float().abs().mean().detach().cpu().item())
    if not math.isfinite(result):
        raise RuntimeError("PR32 Target-Sparse audio diagnostic produced a non-finite absolute mean")
    return result


def _abs_max(value: torch.Tensor) -> float:
    result = float(value.float().abs().max().detach().cpu().item())
    if not math.isfinite(result):
        raise RuntimeError("PR32 Target-Sparse audio diagnostic produced a non-finite absolute maximum")
    return result


def _boundary_rms(delta: torch.Tensor, *, region: str) -> float:
    if delta.ndim < 1 or int(delta.shape[-1]) < 1:
        raise RuntimeError("PR32 Target-Sparse audio diagnostic requires a non-empty audio time axis")
    length = int(delta.shape[-1])
    width = min(16, max(1, length // 8))
    if region == "start":
        part = delta[..., :width]
    elif region == "middle":
        center = length // 2
        lo = max(0, center - width // 2)
        part = delta[..., lo : min(length, lo + width)]
    elif region == "end":
        part = delta[..., -width:]
    else:
        raise ValueError(f"unknown audio diagnostic region {region!r}")
    return _rms(part)


def _conditioning_template(guider: Any) -> dict[str, list[Any]]:
    current = getattr(guider, "conds", None)
    if not isinstance(current, dict):
        raise RuntimeError("PR32 Target-Sparse audio diagnostic requires ComfyUI guider conditioning state")
    return {
        key: [entry.copy() if isinstance(entry, dict) else copy.copy(entry) for entry in entries]
        for key, entries in current.items()
    }


def _continuum_target_sparse_enabled(guider: Any) -> bool:
    options = getattr(guider, "model_options", None)
    transformer = (options or {}).get("transformer_options") or {}
    continuum = transformer.get("h3_continuum")
    return isinstance(continuum, dict) and continuum.get("active") is True


@contextlib.contextmanager
def _without_target_sparse_contract(guider: Any):
    options = getattr(guider, "model_options", None)
    if not isinstance(options, dict):
        raise RuntimeError("PR32 Target-Sparse audio diagnostic requires mutable model options")
    transformer = options.setdefault("transformer_options", {})
    if not isinstance(transformer, dict):
        raise RuntimeError("PR32 Target-Sparse audio diagnostic requires mutable transformer options")
    previous = transformer.pop(_runtime.TARGET_SPARSE_CONTRACT_KEY, None)
    if previous is None:
        raise RuntimeError("PR32 Target-Sparse audio full probe could not find the active sparse contract")
    try:
        yield
    finally:
        transformer[_runtime.TARGET_SPARSE_CONTRACT_KEY] = previous


def _probe_once(
    executor: Any,
    guider: Any,
    *,
    probe_noise: torch.Tensor,
    latent_image: torch.Tensor,
    source_sampler: Any,
    sigma_tensor: torch.Tensor,
    denoise_mask: torch.Tensor,
    disable_pbar: bool,
    seed: int | None,
    latent_shapes: list[tuple[int, ...]],
    template: dict[str, list[Any]],
    outer_step: int,
    stage: str,
    full_transformer: bool,
) -> torch.Tensor:
    options = getattr(guider, "model_options", None)
    transformer = (options or {}).get("transformer_options")
    if not isinstance(transformer, dict):
        raise RuntimeError("PR32 Target-Sparse audio diagnostic requires mutable transformer options")

    previous_probe = transformer.get(_runtime.PROBE_CONTEXT_KEY)
    transformer[_runtime.PROBE_CONTEXT_KEY] = {
        "outer_step": int(outer_step),
        "pr32_audio_probe": stage,
    }
    _runtime._reset_guider_conds(guider, template=template)
    try:
        sparse_context = _without_target_sparse_contract(guider) if full_transformer else contextlib.nullcontext()
        with (
            _runtime._flow_stage_contract(guider, stage),
            _runtime._high_stage_contract(guider),
            sparse_context,
        ):
            return executor(
                probe_noise,
                latent_image,
                _runtime._make_probe_sampler(source_sampler),
                sigma_tensor,
                denoise_mask,
                None,
                disable_pbar,
                seed,
                latent_shapes=latent_shapes,
            )
    finally:
        if previous_probe is None:
            transformer.pop(_runtime.PROBE_CONTEXT_KEY, None)
        else:
            transformer[_runtime.PROBE_CONTEXT_KEY] = previous_probe


class _TargetSparseAudioProbeExecutor:
    def __init__(self, executor: Any, guider: Any, binding: Any) -> None:
        self._executor = executor
        self.class_obj = getattr(executor, "class_obj", guider)
        self._guider = guider
        self._binding = binding
        self._calls = 0
        self._corrected_high_noise: torch.Tensor | None = None
        self._target_shapes: list[tuple[int, ...]] | None = None

    def __getattr__(self, name: str):
        return getattr(self._executor, name)

    def _prepare_audio_rebase(
        self,
        *,
        low_result: torch.Tensor,
        original_noise: torch.Tensor,
        latent_image: torch.Tensor,
        source_sampler: Any,
        call_sigmas: torch.Tensor,
        denoise_mask: torch.Tensor,
        disable_pbar: bool,
        seed: int | None,
        latent_shapes: list[tuple[int, ...]],
        template: dict[str, list[Any]],
    ) -> None:
        if call_sigmas.ndim != 1 or call_sigmas.numel() < 2:
            raise RuntimeError("PR32 Target-Sparse audio diagnostic requires the low split sigma schedule")
        sigma = float(call_sigmas[-1].item())
        if not 0.0 < sigma < 1.0:
            raise RuntimeError("PR32 Target-Sparse audio diagnostic requires a non-terminal handoff sigma")
        sigma_tensor = call_sigmas[-1:]
        outer_step = int(call_sigmas.numel() - 1)

        base_model = self._guider.model_patcher.model
        raw_state = _runtime._raw_sampler_state(base_model, low_result, latent_shapes, sigma)
        latent_internal = _runtime._process_latent_in(base_model, latent_image, latent_shapes)
        probe_noise = _runtime._noise_argument(base_model, raw_state, sigma, latent_internal)
        probe_noise = _runtime._merge_preserved_noise(probe_noise, original_noise, denoise_mask)

        active_capture = self._binding.active_capture
        self._binding.active_capture = None
        try:
            sparse_probe = _probe_once(
                self._executor,
                self._guider,
                probe_noise=probe_noise,
                latent_image=latent_image,
                source_sampler=source_sampler,
                sigma_tensor=sigma_tensor,
                denoise_mask=denoise_mask,
                disable_pbar=disable_pbar,
                seed=seed,
                latent_shapes=latent_shapes,
                template=template,
                outer_step=outer_step,
                stage="audio_probe_sparse",
                full_transformer=False,
            )
            full_probe = _probe_once(
                self._executor,
                self._guider,
                probe_noise=probe_noise,
                latent_image=latent_image,
                source_sampler=source_sampler,
                sigma_tensor=sigma_tensor,
                denoise_mask=denoise_mask,
                disable_pbar=disable_pbar,
                seed=seed,
                latent_shapes=latent_shapes,
                template=template,
                outer_step=outer_step,
                stage="audio_probe_full",
                full_transformer=True,
            )
        finally:
            self._binding.active_capture = active_capture
            _runtime._reset_guider_conds(self._guider, template=template)

        sparse_x0 = _runtime._process_latent_in(base_model, sparse_probe, latent_shapes)
        full_x0 = _runtime._process_latent_in(base_model, full_probe, latent_shapes)
        _sparse_video, sparse_audio = unpack_streams(sparse_x0, latent_shapes)
        _full_video, full_audio = unpack_streams(full_x0, latent_shapes)
        state_video, state_audio = unpack_streams(raw_state, latent_shapes)
        if sparse_audio.shape != full_audio.shape or sparse_audio.shape != state_audio.shape:
            raise RuntimeError("PR32 Target-Sparse audio diagnostic observed inconsistent audio geometry")
        if not bool(torch.isfinite(sparse_audio).all().item() and torch.isfinite(full_audio).all().item()):
            raise RuntimeError("PR32 Target-Sparse audio probe produced NaN or Inf values")

        clean_delta = full_audio.float() - sparse_audio.float()
        audio_state_correction = (1.0 - sigma) * clean_delta
        corrected_audio_state = state_audio + audio_state_correction.to(
            device=state_audio.device,
            dtype=state_audio.dtype,
        )
        corrected_state = pack_streams((state_video, corrected_audio_state))[0]
        corrected_noise = _runtime._noise_argument(base_model, corrected_state, sigma, latent_internal)
        corrected_noise = _runtime._merge_preserved_noise(corrected_noise, original_noise, denoise_mask)

        self._corrected_high_noise = corrected_noise.detach()
        self._target_shapes = list(latent_shapes)
        self._binding.metrics.increment("target_sparse_audio_probe_runs")
        self._binding.metrics.increment("target_sparse_audio_probe_nfe", 2)
        self._binding.metrics.event(
            "target_sparse_audio_probe",
            mode=_MODE,
            sigma=sigma,
            sparse_probe_nfe=1,
            full_probe_nfe=1,
            video_state_modified=False,
            audio_state_rebase_applied=True,
            sparse_audio_x0_rms=_rms(sparse_audio),
            full_audio_x0_rms=_rms(full_audio),
            audio_x0_delta_rms=_rms(clean_delta),
            audio_x0_delta_abs_mean=_abs_mean(clean_delta),
            audio_x0_delta_abs_max=_abs_max(clean_delta),
            audio_x0_delta_start_rms=_boundary_rms(clean_delta, region="start"),
            audio_x0_delta_middle_rms=_boundary_rms(clean_delta, region="middle"),
            audio_x0_delta_end_rms=_boundary_rms(clean_delta, region="end"),
            audio_state_correction_rms=_rms(audio_state_correction),
            audio_state_correction_scale=1.0 - sigma,
            audio_elements=int(clean_delta.numel()),
        )

    def __call__(
        self,
        noise,
        latent_image,
        sampler,
        sigmas,
        denoise_mask,
        callback,
        disable_pbar,
        seed,
        *,
        latent_shapes,
    ):
        self._calls += 1
        if self._calls == 1:
            template = _conditioning_template(self._guider)
            result = self._executor(
                noise,
                latent_image,
                sampler,
                sigmas,
                denoise_mask,
                callback,
                disable_pbar,
                seed,
                latent_shapes=latent_shapes,
            )
            self._prepare_audio_rebase(
                low_result=result,
                original_noise=noise,
                latent_image=latent_image,
                source_sampler=sampler,
                call_sigmas=sigmas,
                denoise_mask=denoise_mask,
                disable_pbar=disable_pbar,
                seed=seed,
                latent_shapes=list(latent_shapes),
                template=template,
            )
            return result

        if self._calls == 2:
            if self._corrected_high_noise is None or self._target_shapes != list(latent_shapes):
                raise RuntimeError("PR32 Target-Sparse audio diagnostic lost the prepared high-stage audio state")
            original_video_noise, _original_audio_noise = unpack_streams(noise, latent_shapes)
            _prepared_video_noise, prepared_audio_noise = unpack_streams(self._corrected_high_noise, latent_shapes)
            replacement_noise = pack_streams((original_video_noise, prepared_audio_noise.to(original_video_noise)))[0]
            replacement_video, replacement_audio = unpack_streams(replacement_noise, latent_shapes)
            if not torch.equal(replacement_video, original_video_noise):
                raise RuntimeError("PR32 Target-Sparse audio diagnostic changed high-stage video noise")
            _orig_video, original_audio = unpack_streams(noise, latent_shapes)
            audio_noise_delta = replacement_audio.float() - original_audio.float()
            self._binding.metrics.event(
                "target_sparse_audio_high_init",
                mode=_MODE,
                video_noise_exact=True,
                audio_noise_delta_rms=_rms(audio_noise_delta),
                audio_noise_delta_abs_max=_abs_max(audio_noise_delta),
            )
            return self._executor(
                replacement_noise,
                latent_image,
                sampler,
                sigmas,
                denoise_mask,
                callback,
                disable_pbar,
                seed,
                latent_shapes=latent_shapes,
            )

        return self._executor(
            noise,
            latent_image,
            sampler,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            seed,
            latent_shapes=latent_shapes,
        )


def run_target_sparse_with_audio_probe(
    executor,
    guider,
    binding,
    config,
    noise,
    latent_image,
    sampler,
    sigmas,
    denoise_mask,
    callback,
    disable_pbar,
    seed,
    latent_shapes,
):
    if not _continuum_target_sparse_enabled(guider):
        return _ORIGINAL_RUN_TARGET_SPARSE(
            executor,
            guider,
            binding,
            config,
            noise,
            latent_image,
            sampler,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            seed,
            latent_shapes,
        )
    proxy = _TargetSparseAudioProbeExecutor(executor, guider, binding)
    return _ORIGINAL_RUN_TARGET_SPARSE(
        proxy,
        guider,
        binding,
        config,
        noise,
        latent_image,
        sampler,
        sigmas,
        denoise_mask,
        callback,
        disable_pbar,
        seed,
        latent_shapes,
    )


def install() -> None:
    if getattr(_runtime._run_target_sparse_exact_prefix, "_h3_pr32_audio_probe", False):
        return
    run_target_sparse_with_audio_probe._h3_pr32_audio_probe = True
    _runtime._run_target_sparse_exact_prefix = run_target_sparse_with_audio_probe


install()
