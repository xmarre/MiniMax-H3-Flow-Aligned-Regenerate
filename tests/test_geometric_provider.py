from __future__ import annotations

import torch

from h3_flow_regenerate.geometric_bridge import SOURCE_CLEAN_CONTEXT_ATTR
from h3_flow_regenerate.geometric_provider import SourceTrajectoryUpscalerProxy, with_source_trajectory_context


class Provider:
    api_version = 1
    kind = "minimax_h3_learned_latent_upscaler"
    model_name = "provider.safetensors"
    device = "cpu"
    inference_device = "cpu"
    precision = "fp32"
    offload_after_upscale = False

    def __init__(self):
        self.calls = []

    def upscale_clean_video(self, video, *, target_h, target_w):
        self.calls.append((video, target_h, target_w))
        return torch.nn.functional.interpolate(
            video.permute(0, 2, 1, 3, 4).reshape(-1, video.shape[1], *video.shape[-2:]),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).reshape(video.shape[0], video.shape[2], video.shape[1], target_h, target_w).permute(0, 2, 1, 3, 4)


def test_proxy_delegates_provider_contract_and_attaches_invocation_local_source():
    provider = Provider()
    wrapped = with_source_trajectory_context(provider)
    assert isinstance(wrapped, SourceTrajectoryUpscalerProxy)
    assert wrapped.api_version == provider.api_version
    assert wrapped.kind == provider.kind
    assert wrapped.model_name == provider.model_name

    source = torch.randn(1, 24, 5, 40, 54, requires_grad=True)
    output = wrapped.upscale_clean_video(source, target_h=56, target_w=76)
    context = getattr(output, SOURCE_CLEAN_CONTEXT_ATTR)
    assert torch.equal(context, source)
    assert not context.requires_grad
    assert provider.calls == [(source, 56, 76)]
    assert SOURCE_CLEAN_CONTEXT_ATTR not in vars(wrapped)


def test_proxy_wrapping_is_idempotent_and_none_safe():
    provider = Provider()
    wrapped = with_source_trajectory_context(provider)
    assert with_source_trajectory_context(wrapped) is wrapped
    assert with_source_trajectory_context(None) is None
