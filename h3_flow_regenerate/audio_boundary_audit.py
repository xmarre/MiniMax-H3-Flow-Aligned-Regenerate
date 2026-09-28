"""Compare identical generated audio under independently extended left context.

This optional decode audit never replaces production audio. A latent norm is
not a loudness proxy; context attribution requires decoding the same latent
payload while changing only context outside the carried overlap.
"""

from __future__ import annotations

import json
import logging

import torch

from .partitioned_stage import tensor_sha256

LOG = logging.getLogger(__name__)
HOP = 800
RATE = 32000
LOW_PROBE_AUDIO_WITNESS_KIND = "partitioned_low_probe_audio_caller_v1"
PRE_SUCCESSOR_AUDIO_WITNESS_KIND = "partitioned_high_pre_successor_audio_caller_v1"


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


def _validate_stage_witness(witness, right, prefix):
    """Validate one ordered stage witness against the exact final carried prefix."""

    prefix_digest = tensor_sha256(right[..., :prefix])
    fields = getattr(witness, "fields", None)
    payload = getattr(witness, "payload", None)
    if not isinstance(fields, dict) or not torch.is_tensor(payload):
        return None, prefix_digest
    if (
        int(fields.get("audio_prefix_ticks", -1)) != prefix
        or int(fields.get("audio_total_ticks", -1)) != int(right.shape[-1])
        or fields.get("final_prefix_sha256") != prefix_digest
        or tuple(payload.shape) != tuple(right.shape)
    ):
        return None, prefix_digest
    return witness, prefix_digest


def audit_audio_boundaries(
    audio_vae,
    latents,
    audios,
    plan,
    *,
    stage_witnesses=None,
    pre_successor_witnesses=None,
):
    if plan.get("magic") != "H3_CONTINUUM_ASSEMBLY_PLAN" or plan.get("fps") != 24:
        raise ValueError("audio boundary audit requires a native H3 Continuum assembly plan")
    groups = plan.get("decode_groups", plan.get("chunks"))
    if not isinstance(groups, list) or len(groups) != len(latents) or len(audios) != len(latents):
        raise ValueError("audio audit inputs must match physical decode groups")
    reports = []
    witnesses = tuple(stage_witnesses or ())
    successor_witnesses = tuple(pre_successor_witnesses or ())
    boundary_count = max(0, len(latents) - 1)
    ordered_witnesses = witnesses[-boundary_count:] if boundary_count and len(witnesses) >= boundary_count else ()
    ordered_successor_witnesses = (
        successor_witnesses[-boundary_count:]
        if boundary_count and len(successor_witnesses) >= boundary_count
        else ()
    )
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
        extended_suffix = decoded[..., cut : cut + window]
        baseline_suffix = baseline[..., prefix * HOP : prefix * HOP + window]

        def decode_stage_counterfactual(
            ordered,
            *,
            policy,
            source_stage,
            common_key,
            final_key,
        ):
            counterfactual_report = {
                "status": "not_requested" if not ordered else "not_evaluated",
                "reason": "stage_witness_not_supplied" if not ordered else "no_matching_stage_witness",
                "policy": policy,
            }
            if not ordered:
                return counterfactual_report, None
            ordered_witness = ordered[i - 1] if len(ordered) == boundary_count else None
            witness, final_prefix_digest = _validate_stage_witness(
                ordered_witness,
                right,
                prefix,
            )
            if witness is None:
                return counterfactual_report, None
            stage_audio = witness.payload
            if stage_audio.dtype != right.dtype:
                counterfactual_report["reason"] = "stage_witness_dtype_mismatch"
                return counterfactual_report, None
            if stage_audio.shape[-1] < prefix + 52:
                counterfactual_report["reason"] = "stage_witness_insufficient_right_context"
                return counterfactual_report, None

            counterfactual_latent = extended.clone()
            counterfactual_start = 32 + prefix
            counterfactual_latent[..., counterfactual_start : counterfactual_start + 52] = stage_audio[
                ..., prefix : prefix + 52
            ].to(counterfactual_latent)
            stage_decoded = decode(counterfactual_latent)
            stage_suffix = stage_decoded[..., cut : cut + window]
            stage_previous = stage_decoded[..., cut - window : cut]
            fields = witness.fields
            counterfactual_report = {
                "status": "measured",
                "reason": "matched_exact_final_prefix",
                "policy": policy,
                "source_stage": source_stage,
                "source_domain": "caller_output_latent",
                "authoritative_prefix_source": "final_continuation_latent",
                "left_context_source": "same_extended_actual_context_as_final_counterfactual",
                "extra_left_ticks": 32,
                "generated_ticks_replaced": 52,
                "window_ticks": 20,
                "final_prefix_sha256": final_prefix_digest,
                "witness_session_id": fields.get("session_id"),
                "witness_chunk_id": fields.get("chunk_id"),
                common_key: _comparison(stage_previous, stage_suffix),
                final_key: _comparison(stage_suffix, extended_suffix),
            }
            return counterfactual_report, stage_decoded

        low_probe_counterfactual, low_probe_decoded = decode_stage_counterfactual(
            ordered_witnesses,
            policy="low_probe_generated_suffix_counterfactual_v1",
            source_stage="low_probe_caller",
            common_key="low_probe_common_decode_boundary",
            final_key="final_vs_low_probe_suffix",
        )
        pre_successor_counterfactual, pre_successor_decoded = decode_stage_counterfactual(
            ordered_successor_witnesses,
            policy="high_pre_successor_suffix_counterfactual_v1",
            source_stage="high_pre_successor_bridge_caller",
            common_key="pre_successor_common_decode_boundary",
            final_key="final_vs_pre_successor_suffix",
        )
        extra_vae_calls = 2 + int(low_probe_decoded is not None) + int(pre_successor_decoded is not None)

        finite_tensors = [baseline, decoded, observed]
        if low_probe_decoded is not None:
            finite_tensors.append(low_probe_decoded)
        if pre_successor_decoded is not None:
            finite_tensors.append(pre_successor_decoded)
        if not all(bool(torch.isfinite(x).all().item()) for x in finite_tensors):
            report.update(status="not_evaluated", reason="nonfinite_audio", extra_vae_calls=extra_vae_calls)
            continue
        inactive = bool((production.float().std(dim=(1, 2)) < 0.2 - 1e-6).all().item())
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
            common_decode_boundary=_comparison(decoded[..., cut - window : cut], extended_suffix),
            low_probe_counterfactual=low_probe_counterfactual,
            pre_successor_counterfactual=pre_successor_counterfactual,
            interpretation="paired_left_context_intervention_with_identical_suffix_and_right_context",
        )
        LOG.info("H3 Flow audio boundary audit %s", json.dumps(report, sort_keys=True))
    return reports


class H3FlowAudioBoundaryAudit:
    DESCRIPTION = (
        "Optional diagnostic after Audio VAE Decode. Connect the original audio latent list, "
        "decoded audio list, VAE and assembly plan; forward audio to Assemble. "
        "With Flow metrics connected, it also decodes the captured low/probe generated-audio "
        "trajectory as a counterfactual. Production audio is never modified."
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
                "metrics": ("H3_FLOW_METRICS",),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "report")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "audit"
    CATEGORY = "MiniMax H3/diagnostics"

    def audit(self, audio_vae, audio_latents, audio, assembly_plan, metrics=None):
        if len(audio_vae) != 1 or len(assembly_plan) != 1:
            raise ValueError("audio audit requires one VAE and one assembly plan")
        stage_witnesses = ()
        pre_successor_witnesses = ()
        if metrics:
            if len(metrics) != 1:
                raise ValueError("audio audit accepts at most one Flow metrics object")
            getter = getattr(metrics[0], "transient_witnesses", None)
            if not callable(getter):
                raise ValueError("audio audit metrics input does not expose transient witnesses")
            stage_witnesses = getter(LOW_PROBE_AUDIO_WITNESS_KIND)
            pre_successor_witnesses = getter(PRE_SUCCESSOR_AUDIO_WITNESS_KIND)
        reports = audit_audio_boundaries(
            audio_vae[0],
            audio_latents,
            audio,
            assembly_plan[0],
            stage_witnesses=stage_witnesses,
            pre_successor_witnesses=pre_successor_witnesses,
        )
        return audio, json.dumps(reports, indent=2, sort_keys=True)


NODE_CLASS_MAPPINGS = {"H3FlowAudioBoundaryAudit": H3FlowAudioBoundaryAudit}
NODE_DISPLAY_NAME_MAPPINGS = {"H3FlowAudioBoundaryAudit": "MiniMax H3 Audio Boundary Audit"}
