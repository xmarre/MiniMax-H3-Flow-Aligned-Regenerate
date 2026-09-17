"""OUTER_SAMPLE boundary for partitioned exact-prefix progressive continuation."""

from __future__ import annotations

import logging

from .audio_guided_overlap import (
    apply_audio_guided_overlap_mask,
    configured_audio_guided_overlap_ticks,
)
from .comfy_compat import _ProgressiveExactMaskExecutor
from .handoff import ProgressiveTargetInputConfig
from .runtime import (
    FLOW_BINDING_KEY,
    PROGRESSIVE_KEY,
    FlowBinding,
    flow_outer_wrapper,
)

LOG = logging.getLogger(__name__)


def partitioned_outer_wrapper(
    executor,
    noise,
    latent_image,
    sampler,
    sigmas,
    denoise_mask=None,
    callback=None,
    disable_pbar=False,
    seed=None,
    latent_shapes=None,
):
    """Run Flow progressive sampling with exact output restore and audio overlap.

    The standard released wrapper enables four-tick guided audio overlap only for
    its conservative target-grid fallback. The partitioned experimental path is
    also an exact-prefix continuation, so it needs the same sampler-lifetime
    audio guidance while retaining the original exact mask for caller-visible
    final restoration.
    """
    guider = executor.class_obj
    model_options = getattr(guider, "model_options", None) or {}
    binding = model_options.get(FLOW_BINDING_KEY)
    progressive = model_options.get(PROGRESSIVE_KEY)
    if not isinstance(binding, FlowBinding) or not isinstance(
        progressive,
        ProgressiveTargetInputConfig,
    ):
        return flow_outer_wrapper(
            executor,
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

    runtime_denoise_mask = denoise_mask
    guided_ticks = configured_audio_guided_overlap_ticks()
    if guided_ticks:
        runtime_denoise_mask, guided_report = apply_audio_guided_overlap_mask(
            denoise_mask,
            latent_shapes,
            ticks=guided_ticks,
        )
        binding.metrics.event(
            "audio_guided_overlap",
            partitioned_exact_prefix=True,
            **guided_report,
        )
        if bool(guided_report.get("applied")):
            LOG.info(
                "partitioned audio guided overlap active ticks=%d exact_prefix=%d ramp=%s final_exact_restore=true",
                guided_ticks,
                int(guided_report.get("audio_prefix_ticks", 0)),
                guided_report.get("ramp_values"),
            )

    adapted = _ProgressiveExactMaskExecutor(
        executor,
        binding=binding,
        progressive=progressive,
        latent_image=latent_image,
        denoise_mask=denoise_mask,
        sampler=sampler,
    )
    return flow_outer_wrapper(
        adapted,
        noise,
        latent_image,
        sampler,
        sigmas,
        runtime_denoise_mask,
        callback,
        disable_pbar,
        seed,
        latent_shapes=latent_shapes,
    )


__all__ = ["partitioned_outer_wrapper"]
