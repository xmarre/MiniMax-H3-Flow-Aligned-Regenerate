"""Invocation-local source-trajectory context for mixed-grid learned transfer."""

from __future__ import annotations

import torch

from .geometric_bridge import SOURCE_CLEAN_CONTEXT_ATTR


class SourceTrajectoryUpscalerProxy:
    """Delegate the upscaler contract while exposing its clean source input once.

    The learned provider receives the genuine source-grid clean sequence immediately
    before upsampling. Attaching a detached view to its output makes that sequence
    available to the mixed-grid geometric diagnostic in the same synchronous call.
    The proxy itself retains no tensor and delegates every provider attribute.
    """

    def __init__(self, provider):
        self._provider = provider

    def __getattr__(self, name):
        return getattr(self._provider, name)

    def upscale_clean_video(self, video, *, target_h, target_w):
        output = self._provider.upscale_clean_video(video, target_h=target_h, target_w=target_w)
        if isinstance(output, torch.Tensor):
            setattr(output, SOURCE_CLEAN_CONTEXT_ATTR, video.detach())
        return output


def with_source_trajectory_context(provider):
    if provider is None or isinstance(provider, SourceTrajectoryUpscalerProxy):
        return provider
    return SourceTrajectoryUpscalerProxy(provider)
