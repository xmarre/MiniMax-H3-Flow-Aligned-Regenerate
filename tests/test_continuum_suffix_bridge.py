from __future__ import annotations

import torch

from h3_flow_regenerate.geometry import pack_streams
from h3_flow_regenerate.runtime import _contiguous_exact_video_prefix


class _IdentityBase:
    def process_latent_in(self, value):
        return value


def _packed_mask(video_mask, audio_mask):
    return torch.cat(
        (
            video_mask.reshape(video_mask.shape[0], 1, -1),
            audio_mask.reshape(audio_mask.shape[0], 1, -1),
        ),
        dim=-1,
    )


def test_contiguous_exact_prefix_extraction_uses_model_domain_latent_and_whole_frames():
    video = torch.arange(1 * 24 * 4 * 2 * 2, dtype=torch.float32).reshape(1, 24, 4, 2, 2)
    audio = torch.zeros(1, 32, 2, 3)
    latent, shapes = pack_streams((video, audio))
    video_mask = torch.ones_like(video)
    video_mask[:, :, :2] = 0
    mask = _packed_mask(video_mask, torch.ones_like(audio))

    prefix = _contiguous_exact_video_prefix(_IdentityBase(), latent, mask, shapes)

    assert prefix is not None
    assert torch.equal(prefix, video[:, :, :2])


def test_contiguous_exact_prefix_extraction_skips_noncanonical_partial_mask():
    video = torch.zeros(1, 24, 4, 2, 2)
    audio = torch.zeros(1, 32, 2, 3)
    latent, shapes = pack_streams((video, audio))
    video_mask = torch.ones_like(video)
    video_mask[:, :, :2] = 0
    video_mask[:, :, 2, 0, 0] = 0
    mask = _packed_mask(video_mask, torch.ones_like(audio))

    assert _contiguous_exact_video_prefix(_IdentityBase(), latent, mask, shapes) is None
