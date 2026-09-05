from __future__ import annotations

from types import SimpleNamespace

import torch

from h3_flow_regenerate.geometry import pack_streams, unpack_streams
from h3_flow_regenerate.runtime import (
    EXACT_PREFIX_BRIDGE_KEY,
    FLOW_BINDING_KEY,
    SPECTRUM_ACTUAL_KEY,
    FlowBinding,
    _contiguous_exact_video_prefix,
    flow_predict_wrapper,
)


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


class _PredictExecutor:
    def __init__(self, guider, result):
        self.class_obj = guider
        self.result = result

    def __call__(self, _x, _timestep, _model_options, _seed):
        return self.result.clone()


def _predict_bridge_fixture(*, actual=True):
    video = torch.zeros(1, 24, 3, 2, 2)
    video[:, :, 0] = 1.0
    video[:, :, 1] = 3.0
    video[:, :, 2] = 5.0
    audio = torch.zeros(1, 32, 2, 3)
    packed, shapes = pack_streams((video, audio))
    exact_prefix = torch.full((1, 24, 1, 2, 2), 2.0)
    contract = {"exact_prefix": exact_prefix, "source": "unit", "applied": False}
    transformer = {
        EXACT_PREFIX_BRIDGE_KEY: contract,
        SPECTRUM_ACTUAL_KEY: bool(actual),
    }
    binding = FlowBinding()
    guider = SimpleNamespace(
        model_options={FLOW_BINDING_KEY: binding, "transformer_options": transformer},
        inner_model=SimpleNamespace(latent_shapes=list(shapes)),
    )
    executor = _PredictExecutor(guider, packed)
    return video, packed, shapes, contract, binding, guider, executor


def test_predict_bridge_applies_once_to_first_suffix_token_and_records_provenance():
    video, packed, shapes, contract, binding, guider, executor = _predict_bridge_fixture(actual=True)

    first = flow_predict_wrapper(
        executor,
        packed,
        torch.tensor([0.5]),
        guider.model_options,
        7,
    )
    first_video, _ = unpack_streams(first, shapes)
    assert torch.equal(first_video[:, :, :1], video[:, :, :1])
    assert torch.equal(first_video[:, :, 2:], video[:, :, 2:])
    assert torch.equal(first_video[:, :, 1], video[:, :, 1] + 1.0)
    assert contract["applied"] is True

    bridge_events = [event for event in binding.metrics.events if event.kind == "exact_prefix_suffix_dc_bridge"]
    assert len(bridge_events) == 1
    assert bridge_events[0].fields["source"] == "unit"
    assert bridge_events[0].fields["actual"] is True
    assert bridge_events[0].fields["suffix_dc_bridge_corrected_tokens"] == 1
    assert bridge_events[0].fields["suffix_dc_bridge_state_mapping"] == "first_actual_model_x0"

    second = flow_predict_wrapper(
        executor,
        packed,
        torch.tensor([0.4]),
        guider.model_options,
        7,
    )
    assert torch.equal(second, packed)
    assert len([event for event in binding.metrics.events if event.kind == "exact_prefix_suffix_dc_bridge"]) == 1


def test_predict_bridge_waits_for_first_actual_call_instead_of_calibrating_from_forecast():
    _video, packed, shapes, contract, binding, guider, executor = _predict_bridge_fixture(actual=False)

    forecast = flow_predict_wrapper(
        executor,
        packed,
        torch.tensor([0.6]),
        guider.model_options,
        7,
    )
    assert torch.equal(forecast, packed)
    assert contract["applied"] is False
    assert not [event for event in binding.metrics.events if event.kind == "exact_prefix_suffix_dc_bridge"]

    guider.model_options["transformer_options"][SPECTRUM_ACTUAL_KEY] = True
    actual = flow_predict_wrapper(
        executor,
        packed,
        torch.tensor([0.5]),
        guider.model_options,
        7,
    )
    actual_video, _ = unpack_streams(actual, shapes)
    assert torch.equal(actual_video[:, :, 1], torch.full_like(actual_video[:, :, 1], 4.0))
    assert contract["applied"] is True
