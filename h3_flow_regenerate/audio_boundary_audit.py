"""Compare identical generated audio under independently extended left context.

This optional decode audit never replaces production audio. A latent norm is
not a loudness proxy; context attribution requires decoding the same latent
payload while changing only context outside the carried overlap.
"""

from __future__ import annotations

import json
import logging

import torch

from .runtime import FLOW_BINDING_KEY, FlowBinding

LOG = logging.getLogger(__name__)
HOP = 800
RATE = 32000


def _rms(x):
    return float(x.float().square().mean().sqrt().item())


def _comparison(reference, candidate):
    a, b = reference.float(), candidate.float()
    before, after = _rms(a), _rms(b)
    return {
        "reference_rms": before,
        "candidate_rms": after,
        "candidate_over_reference_db": (
            float(20 * torch.log10(torch.tensor(after / before)).item()) if before > 0 and after > 0 else None
        ),
        "difference_rms": _rms(b - a),
        "relative_difference_rms": _rms(b - a) / before if before > 0 else None,
        "max_abs_difference": float((b - a).abs().max().item()),
    }


def _find_stage_witness(stage_witnesses, group):
    if stage_witnesses is None:
        return None, {"status": "not_requested", "reason": "flow_model_not_connected"}
    chunk_index = group.get("chunk_index")
    if chunk_index is None:
        return None, {"status": "not_evaluated", "reason": "decode_group_has_no_chunk_index"}
    matches = [
        (key, value)
        for key, value in stage_witnesses.items()
        if isinstance(key, tuple) and len(key) == 2 and str(key[1]) == str(chunk_index)
    ]
    if not matches:
        return None, {
            "status": "not_evaluated",
            "reason": "low_probe_witness_not_found",
            "chunk_index": int(chunk_index),
        }
    if len(matches) != 1:
        return None, {
            "status": "not_evaluated",
            "reason": "ambiguous_low_probe_witness",
            "chunk_index": int(chunk_index),
            "matches": len(matches),
        }
    (session_id, stored_chunk), record = matches[0]
    if not isinstance(record, dict) or record.get("stage") != "low_probe_clean":
        return None, {
            "status": "not_evaluated",
            "reason": "invalid_low_probe_witness_record",
            "chunk_index": int(chunk_index),
        }
    witness = record.get("audio")
    if not torch.is_tensor(witness):
        return None, {
            "status": "not_evaluated",
            "reason": "low_probe_witness_audio_missing",
            "chunk_index": int(chunk_index),
        }
    return witness, {
        "status": "available",
        "session_id": str(session_id),
        "chunk_index": str(stored_chunk),
        "stage": str(record.get("stage")),
        "domain": str(record.get("domain")),
    }


def audit_audio_boundaries(audio_vae, latents, audios, plan, *, stage_witnesses=None):
    if plan.get("magic") != "H3_CONTINUUM_ASSEMBLY_PLAN" or plan.get("fps") != 24:
        raise ValueError("audio boundary audit requires a native H3 Continuum assembly plan")
    groups = plan.get("decode_groups", plan.get("chunks"))
    if not isinstance(groups, list) or len(groups) != len(latents) or len(audios) != len(latents):
        raise ValueError("audio audit inputs must match physical decode groups")
    reports = []
    for i in range(1, len(latents)):
        report = {"boundary_index": i, "policy": "audio_left_context_audit_v1", "production_audio_modified": False}
        reports.append(report)
        prefix = round(int(groups[i]["trim_frames"]) * 40 / 24)
        left, right = latents[i - 1]["samples"], latents[i]["samples"]
        if (
            not torch.is_tensor(left)
            or not torch.is_tensor(right)
            or left.ndim != 4
            or right.ndim != 4
            or tuple(left.shape[:3]) != tuple(right.shape[:3])
            or tuple(right.shape[1:3]) != (32, 2)
        ):
            report.update(status="not_evaluated", reason="native_audio_latents_required", extra_vae_calls=0)
            continue
        # 20 measured ticks, 32 right-context ticks, and 32 extra left ticks.
        # The 128-tick overlap cap keeps diagnostic decoding bounded.
        if not 20 <= prefix <= 128 or left.shape[-1] < prefix + 32 or right.shape[-1] < prefix + 52:
            report.update(status="not_evaluated", reason="insufficient_bounded_context", extra_vae_calls=0)
            continue
        if left.dtype != right.dtype or not torch.equal(left[..., -prefix:].cpu(), right[..., :prefix].cpu()):
            report.update(status="not_evaluated", reason="carried_overlap_not_exact", extra_vae_calls=0)
            continue
        audio = audios[i]
        production = audio["waveform"]
        if (
            int(audio["sample_rate"]) != RATE
            or production.ndim != 3
            or tuple(production.shape[:2]) != (right.shape[0], 2)
            or production.shape[-1] < (prefix + 20) * HOP
        ):
            report.update(status="not_evaluated", reason="production_audio_geometry", extra_vae_calls=0)
            continue
        baseline_latent = right[..., : prefix + 52].detach().clone()
        extended = torch.cat((left[..., -prefix - 32 : -prefix].to(right), baseline_latent), dim=-1)

        def decode(value):
            result = audio_vae.decode(value)
            if result.ndim != 3 or tuple(result.shape) != (value.shape[0], value.shape[-1] * HOP, 2):
                raise ValueError("audio audit VAE must return native [B, samples, 2] audio")
            return result.movedim(-1, 1).detach().cpu()

        # Equal right context in the paired intervention. The production crop
        # comparison separately detects effects of truncation or normalization.
        baseline = decode(baseline_latent)
        decoded = decode(extended)
        cut = (prefix + 32) * HOP
        window = 20 * HOP
        observed = production[..., prefix * HOP : prefix * HOP + window].detach().cpu()
        extended_prefix = decoded[..., cut - window : cut]
        extended_suffix = decoded[..., cut : cut + window]
        baseline_suffix = baseline[..., prefix * HOP : prefix * HOP + window]
        if not all(bool(torch.isfinite(x).all().item()) for x in (baseline, decoded, observed)):
            report.update(status="not_evaluated", reason="nonfinite_audio", extra_vae_calls=2)
            continue
        inactive = bool((production.float().std(dim=(1, 2)) < 0.2 - 1e-6).all().item())

        witness, witness_receipt = _find_stage_witness(stage_witnesses, groups[i])
        stage_localizer = dict(witness_receipt)
        extra_vae_calls = 2
        if witness is not None:
            if (
                witness.ndim != 4
                or tuple(witness.shape) != tuple(right.shape)
                or tuple(witness.shape[1:3]) != (32, 2)
                or not bool(torch.isfinite(witness).all().item())
            ):
                stage_localizer.update(status="not_evaluated", reason="low_probe_witness_geometry_or_values")
            else:
                low_probe_candidate = right.detach().clone()
                low_probe_candidate[..., prefix:] = witness[..., prefix:].to(low_probe_candidate)
                low_probe_extended_latent = torch.cat(
                    (
                        left[..., -prefix - 32 : -prefix].to(right),
                        low_probe_candidate[..., : prefix + 52],
                    ),
                    dim=-1,
                )
                low_probe_decoded = decode(low_probe_extended_latent)
                extra_vae_calls += 1
                low_probe_prefix = low_probe_decoded[..., cut - window : cut]
                low_probe_suffix = low_probe_decoded[..., cut : cut + window]
                if not bool(torch.isfinite(low_probe_decoded).all().item()):
                    stage_localizer.update(status="not_evaluated", reason="nonfinite_low_probe_decode")
                else:
                    stage_localizer.update(
                        status="measured",
                        generated_suffix_replaced_only=True,
                        authoritative_prefix_from_final_latent=True,
                        identical_extra_left_context=True,
                        authoritative_prefix_shared=True,
                        stage_consistent_generated_right_context=True,
                        low_probe_common_decode_boundary=_comparison(low_probe_prefix, low_probe_suffix),
                        final_common_decode_boundary=_comparison(extended_prefix, extended_suffix),
                        low_probe_vs_final_suffix=_comparison(low_probe_suffix, extended_suffix),
                        low_probe_vs_final_preboundary=_comparison(low_probe_prefix, extended_prefix),
                        interpretation=("decode_matched_low_probe_vs_final_generated_suffix_stage_localizer"),
                    )

        report.update(
            status="measured",
            prefix_ticks=prefix,
            extra_left_ticks=32,
            right_context_ticks=32,
            window_ticks=20,
            decoded_ticks=int(extended.shape[-1]),
            extra_vae_calls=extra_vae_calls,
            production_normalizer_provably_inactive=inactive,
            context_comparison_valid=True,
            production_crop_comparison_valid=inactive,
            same_suffix_context_comparison=_comparison(baseline_suffix, extended_suffix),
            production_crop_comparison=_comparison(observed, baseline_suffix),
            common_decode_boundary=_comparison(extended_prefix, extended_suffix),
            stage_localizer=stage_localizer,
            interpretation="paired_left_context_intervention_with_identical_suffix_and_right_context",
        )
        LOG.info("H3 Flow audio boundary audit %s", json.dumps(report, sort_keys=True))
    return reports


class H3FlowAudioBoundaryAudit:
    DESCRIPTION = (
        "Optional diagnostic after Audio VAE Decode. Connect the original audio latent list, "
        "decoded audio list, VAE and assembly plan; forward audio to Assemble. Optionally connect "
        "the Flow-patched model to compare the cached low/probe audio suffix against final-high "
        "under identical AudioVAE context. Production audio is never modified."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_vae": ("VAE",),
                "audio_latents": ("LATENT",),
                "audio": ("AUDIO",),
                "assembly_plan": ("H3_CONTINUUM_ASSEMBLY_PLAN",),
            },
            "optional": {
                "flow_model": ("MODEL",),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "report")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "audit"
    CATEGORY = "MiniMax H3/diagnostics"

    def audit(self, audio_vae, audio_latents, audio, assembly_plan, flow_model=None):
        if len(audio_vae) != 1 or len(assembly_plan) != 1:
            raise ValueError("audio audit requires one VAE and one assembly plan")
        stage_witnesses = None
        if flow_model is not None:
            if len(flow_model) != 1:
                raise ValueError("audio audit requires at most one Flow model")
            options = getattr(flow_model[0], "model_options", None) or {}
            binding = options.get(FLOW_BINDING_KEY)
            if isinstance(binding, FlowBinding):
                stage_witnesses = binding.audio_stage_witnesses
        reports = audit_audio_boundaries(
            audio_vae[0],
            audio_latents,
            audio,
            assembly_plan[0],
            stage_witnesses=stage_witnesses,
        )
        return audio, json.dumps(reports, indent=2, sort_keys=True)


NODE_CLASS_MAPPINGS = {"H3FlowAudioBoundaryAudit": H3FlowAudioBoundaryAudit}
NODE_DISPLAY_NAME_MAPPINGS = {"H3FlowAudioBoundaryAudit": "MiniMax H3 Audio Boundary Audit"}
