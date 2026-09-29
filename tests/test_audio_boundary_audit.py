import json
from types import SimpleNamespace

import torch

from h3_flow_regenerate.audio_boundary_audit import H3FlowAudioBoundaryAudit, audit_audio_boundaries
from h3_flow_regenerate.runtime import FLOW_BINDING_KEY, FlowBinding


class Decoder:
    def __init__(self, global_context=False):
        self.calls = []
        self.global_context = global_context

    def decode(self, x):
        self.calls.append(x.clone())
        value = x.mean(dim=1)
        if self.global_context:
            value = value + x.mean()
        return value.repeat_interleave(800, dim=-1).movedim(1, -1)


def inputs():
    left = torch.full((1, 32, 2, 160), 0.01)
    left[..., -97:-65] = 0.03
    right = torch.cat((left[..., -65:], torch.full((1, 32, 2, 150), 0.04)), dim=-1)
    latents = [{"samples": left}, {"samples": right}]
    vae = Decoder()
    audios = [{"waveform": vae.decode(x["samples"]).movedim(-1, 1), "sample_rate": 32000} for x in latents]
    plan = {
        "magic": "H3_CONTINUUM_ASSEMBLY_PLAN",
        "fps": 24,
        "chunks": [
            {"trim_frames": 0, "chunk_index": 1},
            {"trim_frames": 39, "chunk_index": 2},
        ],
    }
    return latents, audios, plan


def test_same_payload_context_pair_and_production_passthrough():
    latents, audios, plan = inputs()
    saved = [x["samples"].clone() for x in latents]
    vae = Decoder()
    output, report = H3FlowAudioBoundaryAudit().audit([vae], latents, audios, [plan])
    assert output is audios
    assert '"extra_vae_calls": 2' in report
    assert len(vae.calls) == 2
    assert torch.equal(vae.calls[0], vae.calls[1][..., 32:])
    assert all(torch.equal(x["samples"], y) for x, y in zip(latents, saved, strict=True))
    receipt = audit_audio_boundaries(Decoder(), latents, audios, plan)[0]
    assert receipt["same_suffix_context_comparison"]["difference_rms"] == 0
    assert receipt["production_crop_comparison"]["difference_rms"] == 0
    # A loudness jump can survive completely identical context decodes.
    assert receipt["common_decode_boundary"]["candidate_over_reference_db"] > 12


def test_context_sensitive_decoder_is_detected_without_normalizing_pair():
    latents, audios, plan = inputs()
    receipt = audit_audio_boundaries(Decoder(global_context=True), latents, audios, plan)[0]
    assert receipt["same_suffix_context_comparison"]["difference_rms"] > 0
    audios[1]["waveform"].fill_(1.0)
    audios[1]["waveform"][..., ::2] = -1.0
    receipt = audit_audio_boundaries(Decoder(), latents, audios, plan)[0]
    assert receipt["production_crop_comparison_valid"] is False
    assert receipt["context_comparison_valid"] is True


def test_nonexact_overlap_never_decodes():
    latents, audios, plan = inputs()
    latents[1]["samples"][..., 0] += 1
    vae = Decoder()
    receipt = audit_audio_boundaries(vae, latents, audios, plan)[0]
    assert receipt["reason"] == "carried_overlap_not_exact"
    assert not vae.calls


def test_low_probe_witness_localizes_generated_stage_under_matched_decode_context():
    latents, audios, plan = inputs()
    saved = [x["samples"].clone() for x in latents]
    witness = latents[1]["samples"].clone()
    witness[..., 65:] = 0.012

    binding = FlowBinding()
    binding.audio_stage_witnesses[("session-a", "2")] = {
        "stage": "low_probe_clean",
        "domain": "caller_vae_latent",
        "audio": witness,
    }
    flow_model = SimpleNamespace(model_options={FLOW_BINDING_KEY: binding})
    vae = Decoder()

    output, report_json = H3FlowAudioBoundaryAudit().audit(
        [vae],
        latents,
        audios,
        [plan],
        [flow_model],
    )

    assert output is audios
    assert len(vae.calls) == 3
    receipt = json.loads(report_json)[0]
    assert receipt["extra_vae_calls"] == 3
    localizer = receipt["stage_localizer"]
    assert localizer["status"] == "measured"
    assert localizer["generated_suffix_replaced_only"] is True
    assert localizer["authoritative_prefix_from_final_latent"] is True
    assert localizer["low_probe_common_decode_boundary"]["candidate_over_reference_db"] < 4
    assert localizer["final_common_decode_boundary"]["candidate_over_reference_db"] > 12
    assert localizer["low_probe_vs_final_suffix"]["candidate_over_reference_db"] > 8
    assert all(torch.equal(x["samples"], y) for x, y in zip(latents, saved, strict=True))


def test_stage_witness_lookup_fails_closed_when_multiple_sessions_are_present():
    latents, audios, plan = inputs()
    witness = latents[1]["samples"].clone()
    witnesses = {
        ("session-a", "2"): {
            "stage": "low_probe_clean",
            "domain": "caller_vae_latent",
            "audio": witness,
        },
        ("session-b", "2"): {
            "stage": "low_probe_clean",
            "domain": "caller_vae_latent",
            "audio": witness,
        },
    }
    vae = Decoder()
    receipt = audit_audio_boundaries(
        vae,
        latents,
        audios,
        plan,
        stage_witnesses=witnesses,
    )[0]
    assert receipt["extra_vae_calls"] == 2
    assert len(vae.calls) == 2
    assert receipt["stage_localizer"]["status"] == "not_evaluated"
    assert receipt["stage_localizer"]["reason"] == "ambiguous_low_probe_witness"
