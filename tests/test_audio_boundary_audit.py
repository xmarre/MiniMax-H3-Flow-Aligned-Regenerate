import json

import torch

from h3_flow_regenerate.audio_boundary_audit import (
    LOW_PROBE_AUDIO_WITNESS_KIND,
    H3FlowAudioBoundaryAudit,
    audit_audio_boundaries,
)
from h3_flow_regenerate.metrics import H3FlowMetrics
from h3_flow_regenerate.partitioned_stage import tensor_sha256


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
    plan = {"magic": "H3_CONTINUUM_ASSEMBLY_PLAN", "fps": 24, "chunks": [{"trim_frames": 0}, {"trim_frames": 39}]}
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


def test_low_probe_counterfactual_decodes_exact_final_prefix_with_probe_suffix():
    latents, audios, plan = inputs()
    right = latents[1]["samples"]
    prefix = 65
    low_probe = right.clone()
    low_probe[..., prefix:] = 0.01

    metrics = H3FlowMetrics()
    metrics.append_transient_witness(
        LOW_PROBE_AUDIO_WITNESS_KIND,
        low_probe,
        session_id="test-session",
        chunk_id="1",
        audio_prefix_ticks=prefix,
        audio_total_ticks=int(right.shape[-1]),
        final_prefix_sha256=tensor_sha256(right[..., :prefix]),
        source_stage="low_probe",
        source_domain="caller_output_latent",
    )

    vae = Decoder()
    output, report_json = H3FlowAudioBoundaryAudit().audit(
        [vae],
        latents,
        audios,
        [plan],
        [metrics],
    )
    assert output is audios
    assert len(vae.calls) == 3
    report = json.loads(report_json)[0]
    assert report["extra_vae_calls"] == 3
    counterfactual = report["low_probe_counterfactual"]
    assert counterfactual["status"] == "measured"
    assert counterfactual["witness_session_id"] == "test-session"
    assert counterfactual["witness_chunk_id"] == "1"
    assert abs(counterfactual["low_probe_common_decode_boundary"]["candidate_over_reference_db"]) < 1.0e-6
    assert counterfactual["final_vs_low_probe_suffix"]["candidate_over_reference_db"] > 12
    assert LOW_PROBE_AUDIO_WITNESS_KIND not in metrics.to_json()


def test_low_probe_counterfactual_rejects_stale_prefix_witness_without_extra_decode():
    latents, audios, plan = inputs()
    right = latents[1]["samples"]
    metrics = H3FlowMetrics()
    metrics.append_transient_witness(
        LOW_PROBE_AUDIO_WITNESS_KIND,
        right.clone(),
        audio_prefix_ticks=65,
        audio_total_ticks=int(right.shape[-1]),
        final_prefix_sha256="stale-prefix",
    )

    vae = Decoder()
    _output, report_json = H3FlowAudioBoundaryAudit().audit(
        [vae],
        latents,
        audios,
        [plan],
        [metrics],
    )
    assert len(vae.calls) == 2
    report = json.loads(report_json)[0]
    assert report["extra_vae_calls"] == 2
    assert report["low_probe_counterfactual"]["status"] == "not_evaluated"
    assert report["low_probe_counterfactual"]["reason"] == "no_matching_low_probe_witness"
