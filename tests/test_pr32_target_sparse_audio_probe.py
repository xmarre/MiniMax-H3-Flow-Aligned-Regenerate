from __future__ import annotations

from types import SimpleNamespace

import torch

from h3_flow_regenerate import runtime
from h3_flow_regenerate.geometry import pack_streams, unpack_streams
from h3_flow_regenerate.metrics import H3FlowMetrics
from h3_flow_regenerate.pr32_target_sparse_audio_probe import (
    _MODE,
    _continuum_target_sparse_enabled,
    _TargetSparseAudioProbeExecutor,
)


class _IdentitySamplerModel:
    def __init__(self):
        self.model_sampling = SimpleNamespace(noise_scale=1.0)
        self.latent_shapes = None

    def process_latent_in(self, value):
        return value


def test_audio_probe_rebases_only_high_audio_state(monkeypatch):
    video = torch.zeros(1, 24, 2, 4, 4)
    audio = torch.full((1, 32, 2, 4), 0.25)
    latent, shapes = pack_streams((video, audio))
    original_noise = latent.clone()
    mask = torch.ones(1, 1, latent.shape[-1])
    sigma_schedule = torch.tensor([1.0, 0.75, 0.5])

    source_sampler = object()
    probe_sampler = object()
    monkeypatch.setattr(runtime, "_make_probe_sampler", lambda _sampler: probe_sampler)

    guider = SimpleNamespace(
        model_options={
            "transformer_options": {
                "h3_continuum": {"active": True, "chunk_index": 2},
                runtime.TARGET_SPARSE_CONTRACT_KEY: {"api": 1, "active": True},
            }
        },
        model_patcher=SimpleNamespace(model=_IdentitySamplerModel()),
        conds={"positive": [{"cross_attn": torch.zeros(1, 2, 4)}]},
    )
    binding = runtime.FlowBinding(metrics=H3FlowMetrics())
    calls: list[dict] = []

    class Executor:
        class_obj = guider

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
            del callback, disable_pbar, seed
            sparse_active = runtime.TARGET_SPARSE_CONTRACT_KEY in guider.model_options["transformer_options"]
            calls.append(
                {
                    "noise": noise.clone(),
                    "sampler": sampler,
                    "sigmas": sigmas.clone(),
                    "sparse": sparse_active,
                }
            )
            if sampler is source_sampler and len(calls) == 1:
                # _raw_sampler_state multiplies the returned value by (1-sigma).
                return latent_image / (1.0 - float(sigmas[-1]))
            if sampler is probe_sampler:
                out_video = torch.zeros_like(video)
                out_audio = torch.ones_like(audio) if sparse_active else torch.full_like(audio, 3.0)
                return pack_streams((out_video, out_audio))[0]
            return latent_image

    proxy = _TargetSparseAudioProbeExecutor(Executor(), guider, binding)
    low_result = proxy(
        original_noise,
        latent,
        source_sampler,
        sigma_schedule,
        mask,
        None,
        True,
        7,
        latent_shapes=shapes,
    )
    assert torch.equal(low_result, latent / 0.5)

    # The two probe calls must see identical sampler inputs; only sparse/full H3
    # sequence ownership differs.
    assert len(calls) == 3
    assert calls[1]["sampler"] is probe_sampler and calls[1]["sparse"] is True
    assert calls[2]["sampler"] is probe_sampler and calls[2]["sparse"] is False
    assert torch.equal(calls[1]["noise"], calls[2]["noise"])

    high_original_noise = original_noise.clone()
    proxy(
        high_original_noise,
        latent,
        source_sampler,
        torch.tensor([0.5, 0.0]),
        mask,
        None,
        True,
        7,
        latent_shapes=shapes,
    )
    assert len(calls) == 4
    high_video_before, high_audio_before = unpack_streams(high_original_noise, shapes)
    high_video_after, high_audio_after = unpack_streams(calls[-1]["noise"], shapes)
    assert torch.equal(high_video_after, high_video_before)
    # x0_full - x0_sparse == 2 and (1-sigma)==0.5. Mapping that state
    # correction back through x_sigma=(1-sigma)*latent+sigma*noise adds 2 to
    # the audio noise argument while leaving video untouched.
    assert torch.allclose(high_audio_after - high_audio_before, torch.full_like(high_audio_before, 2.0))

    probe_event = [event for event in binding.metrics.events if event.kind == "target_sparse_audio_probe"][-1]
    assert probe_event.fields["mode"] == _MODE
    assert probe_event.fields["video_state_modified"] is False
    assert probe_event.fields["audio_x0_delta_rms"] == 2.0
    assert probe_event.fields["audio_state_correction_rms"] == 1.0
    high_event = [event for event in binding.metrics.events if event.kind == "target_sparse_audio_high_init"][-1]
    assert high_event.fields["video_noise_exact"] is True
    assert binding.metrics.counters["target_sparse_audio_probe_nfe"] == 2


def test_audio_probe_only_arms_for_continuum_target_sparse_context():
    guider = SimpleNamespace(model_options={"transformer_options": {}})
    assert _continuum_target_sparse_enabled(guider) is False
    guider.model_options["transformer_options"]["h3_continuum"] = {"active": True}
    assert _continuum_target_sparse_enabled(guider) is True
