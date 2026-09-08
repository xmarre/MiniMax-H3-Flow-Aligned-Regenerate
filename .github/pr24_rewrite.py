from __future__ import annotations

import subprocess
from pathlib import Path

BASE = "fe0ef8752b92081b5a85bc9b39ad8e2a7037d591"
ROOT = Path(__file__).resolve().parents[1]

PR_CHANGED = [
    "README.md",
    "docs/MIXED_GRID_CONTINUUM.md",
    "docs/geometric-seam-bridge.md",
    "h3_flow_regenerate/comfy_compat.py",
    "h3_flow_regenerate/final_geometry.py",
    "h3_flow_regenerate/geometric_bridge.py",
    "h3_flow_regenerate/geometric_provider.py",
    "h3_flow_regenerate/handoff.py",
    "h3_flow_regenerate/runtime.py",
    "h3_flow_regenerate/target_sparse_node.py",
    "pyproject.toml",
    "tests/test_exact_mask_output.py",
    "tests/test_final_geometry.py",
    "tests/test_geometric_bridge.py",
    "tests/test_geometric_combined_evidence.py",
    "tests/test_geometric_provider.py",
    "tests/test_geometric_runtime.py",
    "tests/test_mixed_grid.py",
]


def base_bytes(path: str) -> bytes | None:
    probe = subprocess.run(
        ["git", "cat-file", "-e", f"{BASE}:{path}"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if probe.returncode:
        return None
    return subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT)


def restore_pr_files() -> None:
    for rel in PR_CHANGED:
        path = ROOT / rel
        content = base_bytes(rel)
        if content is None:
            if path.exists():
                path.unlink()
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def representation_module() -> str:
    return '''"""Exact-prefix representation reconciliation for Mixed-Grid Continuum.

The learned 3D upscaler produces a coherent target-grid prefix and suffix, but
Mixed-Grid must discard that learned prefix and restore Continuum's authoritative
prefix.  The overlap therefore gives us an exact, same-frame measurement of the
representation residual introduced by that replacement.

This module transfers only the zero-spatial-mean part of the last overlap
residual onto the first generated suffix token.  The existing DC bridge owns the
spatial-mean component.  When both bridges are enabled their sum is exactly the
full same-frame residual, so the exact-prefix -> corrected-suffix transition
matches the upscaler's native learned-prefix -> learned-suffix transition before
fresh target-grid refinement.

No prefix value, later suffix token, audio value, noise, mask, conditioning, VAE
state, or model evaluation is changed here.  Disabled and already-matched paths
return the original tensor object.
"""

from __future__ import annotations

import math

import torch


def _rms(value: torch.Tensor) -> float:
    result = float(value.float().square().mean().sqrt().detach().cpu().item())
    if not math.isfinite(result):
        raise RuntimeError("representation bridge produced a non-finite RMS")
    return result


def _validate_pair(learned: torch.Tensor, exact_prefix: torch.Tensor) -> int:
    if not torch.is_tensor(learned) or not torch.is_tensor(exact_prefix):
        raise TypeError("representation bridge expects torch.Tensor inputs")
    if learned.ndim != 5 or exact_prefix.ndim != 5:
        raise ValueError("representation bridge expects BxCxTxHxW tensors")
    if not learned.is_floating_point() or not exact_prefix.is_floating_point():
        raise TypeError("representation bridge expects floating-point tensors")
    if learned.shape[:2] != exact_prefix.shape[:2] or learned.shape[-2:] != exact_prefix.shape[-2:]:
        raise ValueError("representation bridge learned/exact geometry differs")
    prefix_t = int(exact_prefix.shape[2])
    if prefix_t < 1 or prefix_t >= int(learned.shape[2]):
        raise ValueError("representation bridge requires a non-empty prefix and suffix")
    if not bool(torch.isfinite(learned).all().item()) or not bool(torch.isfinite(exact_prefix).all().item()):
        raise RuntimeError("representation bridge inputs contain NaN or Inf")
    return prefix_t


def disabled_suffix_representation_bridge_metrics(*, prefix_t: int, requested: bool = False) -> dict:
    p = int(prefix_t)
    if p < 1:
        raise ValueError("representation bridge prefix length must be positive")
    return {
        "suffix_representation_bridge_version": 1,
        "suffix_representation_bridge_requested": bool(requested),
        "suffix_representation_bridge_enabled": False,
        "suffix_representation_bridge_accepted": False,
        "suffix_representation_bridge_reason": "disabled" if not requested else "not_applied",
        "suffix_representation_bridge_prefix_t": p,
        "suffix_representation_bridge_corrected_tokens": 0,
        "suffix_representation_bridge_delta_rms": 0.0,
        "suffix_representation_bridge_dc_rms": 0.0,
        "suffix_representation_bridge_structural_rms": 0.0,
        "suffix_representation_bridge_centered_error_before": 0.0,
        "suffix_representation_bridge_centered_error_after": 0.0,
        "suffix_representation_bridge_centered_error_ratio": 1.0,
    }


@torch.no_grad()
def apply_suffix_representation_bridge(
    learned_clean_video: torch.Tensor,
    exact_prefix: torch.Tensor,
    *,
    requested: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Transplant the measured overlap representation residual to suffix token 0.

    The correction is not a temporal fade or image-space blend.  Let ``L_p`` be
    the learned upscaler's last prefix token, ``E_p`` the exact token that will
    replace it, and ``L_s`` the first learned suffix token.  The measured overlap
    residual is ``D = E_p - L_p``.  This function applies ``D - mean_spatial(D)``
    to ``L_s``; the independent DC bridge may apply ``mean_spatial(D)``.

    Consequently, with both bridges active, ``(L_s + D) - E_p == L_s - L_p``:
    exact-prefix restoration preserves the upscaler's native boundary transition
    algebraically instead of fitting an affine camera transform after the fact.
    """

    prefix_t = _validate_pair(learned_clean_video, exact_prefix)
    if not isinstance(requested, bool):
        raise TypeError("representation bridge requested flag must be boolean")
    if not requested:
        return learned_clean_video, disabled_suffix_representation_bridge_metrics(prefix_t=prefix_t)

    learned_last = learned_clean_video[:, :, prefix_t - 1].float()
    exact_last = exact_prefix[:, :, -1].to(device=learned_clean_video.device, dtype=torch.float32)
    suffix_first = learned_clean_video[:, :, prefix_t].float()
    delta = exact_last - learned_last
    dc = delta.mean(dim=(-2, -1), keepdim=True)
    structural = delta - dc

    native_transition = suffix_first - learned_last
    hard_transition = suffix_first - exact_last
    native_centered = native_transition - native_transition.mean(dim=(-2, -1), keepdim=True)
    hard_centered = hard_transition - hard_transition.mean(dim=(-2, -1), keepdim=True)
    before_error = _rms(hard_centered - native_centered)

    metrics = disabled_suffix_representation_bridge_metrics(prefix_t=prefix_t, requested=True)
    metrics.update(
        suffix_representation_bridge_delta_rms=_rms(delta),
        suffix_representation_bridge_dc_rms=_rms(dc),
        suffix_representation_bridge_structural_rms=_rms(structural),
        suffix_representation_bridge_centered_error_before=before_error,
    )
    if before_error == 0.0 or not bool(torch.count_nonzero(structural).item()):
        metrics.update(
            suffix_representation_bridge_reason="structural_overlap_already_matched",
            suffix_representation_bridge_centered_error_after=before_error,
            suffix_representation_bridge_centered_error_ratio=1.0,
        )
        return learned_clean_video, metrics

    corrected = learned_clean_video.clone()
    corrected[:, :, prefix_t].add_(structural.to(dtype=corrected.dtype))
    if not bool(torch.isfinite(corrected).all().item()):
        raise RuntimeError("representation bridge produced NaN or Inf")
    if not torch.equal(corrected[:, :, :prefix_t], learned_clean_video[:, :, :prefix_t]):
        raise RuntimeError("representation bridge modified prefix values")
    if not torch.equal(corrected[:, :, prefix_t + 1 :], learned_clean_video[:, :, prefix_t + 1 :]):
        raise RuntimeError("representation bridge modified unmeasured later suffix tokens")

    corrected_transition = corrected[:, :, prefix_t].float() - exact_last
    corrected_centered = corrected_transition - corrected_transition.mean(dim=(-2, -1), keepdim=True)
    after_error = _rms(corrected_centered - native_centered)
    ratio = after_error / max(before_error, 1e-12)
    metrics.update(
        suffix_representation_bridge_centered_error_after=after_error,
        suffix_representation_bridge_centered_error_ratio=ratio,
    )
    if not after_error < before_error:
        metrics.update(suffix_representation_bridge_reason="projection_did_not_reduce_measured_overlap_error")
        return learned_clean_video, metrics

    metrics.update(
        suffix_representation_bridge_enabled=True,
        suffix_representation_bridge_accepted=True,
        suffix_representation_bridge_reason="exact_overlap_structural_residual_transplanted",
        suffix_representation_bridge_corrected_tokens=1,
    )
    return corrected, metrics
'''


def representation_tests() -> str:
    return '''from __future__ import annotations

import pytest
import torch

from h3_flow_regenerate.representation_bridge import (
    apply_suffix_representation_bridge,
    disabled_suffix_representation_bridge_metrics,
)
from h3_flow_regenerate.tone_bridge import apply_suffix_dc_bridge


def _pair(dtype=torch.float32):
    torch.manual_seed(91)
    learned = torch.randn(1, 24, 5, 8, 10, dtype=torch.float32)
    exact = learned[:, :, :2].clone()
    yy = torch.linspace(-1, 1, 8).view(1, 1, 1, 8, 1)
    xx = torch.linspace(-1, 1, 10).view(1, 1, 1, 1, 10)
    exact = exact + 0.12 + 0.08 * yy - 0.05 * xx
    return learned.to(dtype), exact.to(dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_structure_plus_dc_preserves_upscaler_native_boundary_transition(dtype):
    learned, exact = _pair(dtype)
    before = learned.clone()
    structured, report = apply_suffix_representation_bridge(learned, exact, requested=True)
    assert report["suffix_representation_bridge_accepted"] is True
    assert report["suffix_representation_bridge_corrected_tokens"] == 1
    assert torch.equal(structured[:, :, :2], before[:, :, :2])
    assert torch.equal(structured[:, :, 3:], before[:, :, 3:])

    combined, dc = apply_suffix_dc_bridge(structured, exact, weights=(1.0,))
    assert dc["suffix_dc_bridge_corrected_tokens"] == 1
    native = before[:, :, 2].float() - before[:, :, 1].float()
    restored = combined[:, :, 2].float() - exact[:, :, 1].float()
    atol = 3e-3 if dtype == torch.bfloat16 else 8e-4 if dtype == torch.float16 else 2e-6
    assert torch.allclose(restored, native, atol=atol, rtol=0)
    assert torch.equal(combined[:, :, :2], before[:, :, :2])
    assert torch.equal(combined[:, :, 3:], before[:, :, 3:])


def test_structure_bridge_transfers_zero_mean_residual_only():
    learned, exact = _pair()
    corrected, report = apply_suffix_representation_bridge(learned, exact, requested=True)
    change = corrected[:, :, 2].float() - learned[:, :, 2].float()
    assert torch.allclose(change.mean(dim=(-2, -1)), torch.zeros_like(change.mean(dim=(-2, -1))), atol=2e-7)
    assert report["suffix_representation_bridge_centered_error_after"] < report[
        "suffix_representation_bridge_centered_error_before"
    ]
    assert report["suffix_representation_bridge_centered_error_ratio"] < 1e-4


def test_bridge_has_no_unmeasured_temporal_fade():
    learned, exact = _pair()
    corrected, report = apply_suffix_representation_bridge(learned, exact, requested=True)
    assert report["suffix_representation_bridge_corrected_tokens"] == 1
    assert not torch.equal(corrected[:, :, 2], learned[:, :, 2])
    assert torch.equal(corrected[:, :, 3:], learned[:, :, 3:])


def test_disabled_and_already_matched_paths_are_exact_object_noops():
    learned, exact = _pair()
    disabled, report = apply_suffix_representation_bridge(learned, exact, requested=False)
    assert disabled is learned
    assert report == disabled_suffix_representation_bridge_metrics(prefix_t=2)

    exact_match = learned[:, :, :2].clone()
    matched, report = apply_suffix_representation_bridge(learned, exact_match, requested=True)
    assert matched is learned
    assert report["suffix_representation_bridge_accepted"] is False
    assert report["suffix_representation_bridge_reason"] == "structural_overlap_already_matched"


def test_nonfinite_input_is_rejected():
    learned, exact = _pair()
    learned = learned.clone()
    learned[0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="NaN or Inf"):
        apply_suffix_representation_bridge(learned, exact)
'''


def rewrite_handoff() -> None:
    path = ROOT / "h3_flow_regenerate/handoff.py"
    text = path.read_text()
    text = replace_once(
        text,
        '    learned_upscaler: Any | None = field(default=None, repr=False, compare=False)\n',
        '    learned_upscaler: Any | None = field(default=None, repr=False, compare=False)\n    suffix_geometric_bridge: bool = False\n',
        label="handoff field",
    )
    needle = (
        '        if self.suffix_dc_bridge and self.exact_prefix_mode not in {"target_sparse_lifter", "mixed_grid_low_suffix"}:\n'
        '            raise ValueError("suffix_dc_bridge is only supported by Continuum-specific exact-prefix modes")\n'
    )
    addition = needle + (
        '        if not isinstance(self.suffix_geometric_bridge, bool):\n'
        '            raise TypeError("suffix_geometric_bridge must be boolean")\n'
        '        if self.suffix_geometric_bridge and self.exact_prefix_mode != "mixed_grid_low_suffix":\n'
        '            raise ValueError("suffix_geometric_bridge requires mixed-grid Continuum")\n'
    )
    text = replace_once(text, needle, addition, label="handoff validation")
    path.write_text(text)


def rewrite_target_sparse_node() -> None:
    path = ROOT / "h3_flow_regenerate/target_sparse_node.py"
    text = path.read_text()
    text = replace_once(
        text,
        '        suffix_dc_bridge=True,\n    ):\n',
        '        suffix_dc_bridge=True,\n        suffix_geometric_bridge=False,\n    ):\n',
        label="patch signature",
    )
    text = text.replace(
        '                suffix_dc_bridge=bool(suffix_dc_bridge),\n                learned_upscaler=learned_upscaler,\n',
        '                suffix_dc_bridge=bool(suffix_dc_bridge),\n                suffix_geometric_bridge=bool(suffix_geometric_bridge),\n                learned_upscaler=learned_upscaler,\n',
    )
    if text.count("suffix_geometric_bridge=bool(suffix_geometric_bridge)") != 2:
        raise RuntimeError("target sparse config wiring did not update both source modes")

    start = text.index("class H3ProgressiveMixedGridHandoff")
    desc_start = text.index("    DESCRIPTION = (", start)
    input_start = text.index("\n\n    @classmethod\n    def INPUT_TYPES", desc_start)
    new_desc = '''    DESCRIPTION = (\n        "Real low-resolution Continuum suffix generation with original target-grid protected-prefix "\n        "conditioning, learned 3D handoff, exact prefix restoration, and fresh target-grid refinement. "\n        "Requires an H3 latent-upscaler provider and VDN external-sequence API v2 when VDN is enabled. "\n        "The validated one-token DC bridge remains enabled by default. The optional legacy-named "\n        "suffix_geometric_bridge now enables an experimental exact-overlap representation bridge: it "\n        "transplants the measured non-DC learned-prefix replacement residual onto only the first suffix "\n        "token before refinement. It remains off by default pending decoded-media validation."\n    )'''
    text = text[:desc_start] + new_desc + text[input_start:]

    marker = "        return inputs\n\n\nNODE_CLASS_MAPPINGS"
    insert = '''        inputs.setdefault("optional", {})["suffix_geometric_bridge"] = (\n            "BOOLEAN",\n            {\n                "default": False,\n                "tooltip": (\n                    "Experimental Mixed-Grid exact-overlap representation bridge (legacy input name kept "\n                    "for workflow compatibility). It measures the discarded learned-upscaler prefix against "\n                    "the authoritative exact prefix on the same final overlap frame, transfers only that "\n                    "zero-mean structural residual to the first generated suffix token, and leaves the DC "\n                    "component to suffix_dc_bridge. With both enabled, the pre-refinement boundary exactly "\n                    "preserves the upscaler's native learned-prefix→suffix transition. No prefix warp, fade, "\n                    "extra H3 call, audio/noise/mask/conditioning change, or later-suffix extrapolation."\n                ),\n            },\n        )\n        return inputs\n\n\nNODE_CLASS_MAPPINGS'''
    text = replace_once(text, marker, insert, label="mixed UI")
    path.write_text(text)


def rewrite_runtime() -> None:
    path = ROOT / "h3_flow_regenerate/runtime.py"
    text = path.read_text()
    import_anchor = "from .seam_diagnostics import (\n"
    if import_anchor not in text:
        raise RuntimeError("runtime seam diagnostics import anchor missing")
    text = text.replace(
        import_anchor,
        "from .representation_bridge import (\n"
        "    apply_suffix_representation_bridge,\n"
        "    disabled_suffix_representation_bridge_metrics,\n"
        ")\n" + import_anchor,
        1,
    )
    build_pos = text.index("        target_raw, target_shapes = build_handoff_state(")
    start = text.index("        if mixed_plan is not None:\n            target_video, target_audio", build_pos)
    end = text.index('        if config.transfer_mode == "learned_3d":', start)
    block = '''        if mixed_plan is not None:\n            target_video, target_audio = unpack_streams(target_raw, target_shapes)\n            diagnostic_started = time.perf_counter()\n            diagnostic_noise = deterministic_video_noise(\n                tuple(target_video.shape),\n                seed=int(seed or 0) + config.seed_offset,\n                device=target_video.device,\n                dtype=target_video.dtype,\n            )\n            learned_clean = recover_conditional_clean_for_diagnostics(\n                target_video,\n                diagnostic_noise,\n                sigma=sigma,\n            )\n            exact_prefix = mixed_plan.prefix.to(device=learned_clean.device, dtype=learned_clean.dtype)\n            representation_requested = bool(getattr(config, "suffix_geometric_bridge", False))\n            if representation_requested:\n                corrected_clean, representation_metrics = apply_suffix_representation_bridge(\n                    learned_clean,\n                    exact_prefix,\n                    requested=True,\n                )\n            else:\n                corrected_clean = learned_clean\n                representation_metrics = disabled_suffix_representation_bridge_metrics(\n                    prefix_t=mixed_plan.prefix_t,\n                    requested=False,\n                )\n\n            dc_enabled = bool(getattr(config, "suffix_dc_bridge", False))\n            if dc_enabled:\n                corrected_clean, bridge_metrics = apply_suffix_dc_bridge(\n                    corrected_clean,\n                    exact_prefix,\n                    weights=(1.0,),\n                )\n            else:\n                bridge_metrics = disabled_suffix_dc_bridge_metrics(prefix_t=mixed_plan.prefix_t)\n\n            corrected_tokens = max(\n                int(representation_metrics["suffix_representation_bridge_corrected_tokens"]),\n                int(bridge_metrics["suffix_dc_bridge_corrected_tokens"]),\n            )\n            if corrected_tokens:\n                target_video = map_clean_bridge_to_conditional_state(\n                    target_video,\n                    learned_clean,\n                    corrected_clean,\n                    sigma=sigma,\n                    prefix_t=mixed_plan.prefix_t,\n                    corrected_tokens=corrected_tokens,\n                )\n            splice_diagnostics = measure_exact_prefix_splice(\n                learned_clean,\n                exact_prefix,\n                corrected_clean_video=corrected_clean,\n            )\n            splice_diagnostics["splice_diagnostic_elapsed_ms"] = (time.perf_counter() - diagnostic_started) * 1000.0\n            splice_diagnostics["splice_recovery"] = "inverse_conditional_renoise"\n            splice_diagnostics["suffix_dc_bridge_state_mapping"] = (\n                "affine_equivalent_pre_renoise" if dc_enabled else "disabled"\n            )\n            splice_diagnostics["suffix_representation_bridge_state_mapping"] = (\n                "affine_equivalent_pre_renoise"\n                if representation_metrics["suffix_representation_bridge_accepted"]\n                else "disabled_or_noop"\n            )\n            binding.metrics.increment("mixed_grid_splice_diagnostic_runs")\n            binding.metrics.event(\n                "mixed_grid_representation_bridge",\n                legacy_option_name="suffix_geometric_bridge",\n                authoritative_prefix_modified=False,\n                later_suffix_extrapolated=False,\n                **representation_metrics,\n            )\n\n            if not corrected_tokens:\n                target_video = target_video.clone()\n            target_video[:, :, : mixed_plan.prefix_t] = mixed_plan.prefix.to(target_video)\n            target_raw = pack_streams((target_video, target_audio))[0]\n            binding.metrics.event(\n                "mixed_grid_transfer",\n                learned_transfer_performed=True,\n                upscaler_prefix_context_used=True,\n                upscaler_prefix_output_discarded=True,\n                final_original_prefix_restored=True,\n                transfer_mode="learned_3d_suffix",\n                **representation_metrics,\n                **bridge_metrics,\n                **splice_diagnostics,\n            )\n            del diagnostic_noise, learned_clean, corrected_clean\n'''
    text = text[:start] + block + text[end:]
    path.write_text(text)


def rewrite_readme() -> None:
    path = ROOT / "README.md"
    text = path.read_text()
    needle = "- [docs/MIXED_GRID_CONTINUUM.md](docs/MIXED_GRID_CONTINUUM.md) — Mixed-Grid contract and diagnostics\n"
    text = replace_once(
        text,
        needle,
        needle + "- [docs/representation-seam-bridge.md](docs/representation-seam-bridge.md) — experimental exact-overlap representation reconciliation for exact-prefix replacement\n",
        label="README docs link",
    )
    path.write_text(text)


def rewrite_mixed_docs() -> None:
    path = ROOT / "docs/MIXED_GRID_CONTINUUM.md"
    text = path.read_text().rstrip() + '''\n\n## Experimental exact-overlap representation bridge\n\nThe decoded `metrics_00276` validation falsified the affine-framing hypothesis as a sufficient repair. The optional affine correction closed the measured `sy` residual from roughly `-0.01025` to `+0.00236`, yet the visible whole-frame shrink/top-edge reveal remained. The more diagnostic measurement was already present at the learned-transfer splice: replacing the learned 3D upscaler's internally coherent prefix with the authoritative exact prefix amplified the boundary by about 23% in raw RMS, 37% in spatial-low-pass RMS, and 40% in per-channel spatial-mean RMS before the existing DC bridge.\n\nThe experimental `suffix_geometric_bridge` input name is retained for workflow compatibility, but its implementation is now a representation-residual bridge rather than an affine warp. It uses the learned prefix only as private overlap calibration:\n\n```text\nL_p = learned upscaler last prefix token\nE_p = authoritative exact last prefix token\nL_s = learned first suffix token\nD   = E_p - L_p\n```\n\nThe representation bridge transfers `D - mean_spatial(D)` to `L_s`. The existing suffix DC bridge independently transfers `mean_spatial(D)`. With both enabled, the corrected boundary satisfies `L_s + D - E_p == L_s - L_p`: restoring the exact prefix preserves the upscaler's native boundary transition algebraically before re-noise and fresh target-grid refinement.\n\nThis is deliberately a one-boundary-token operation, not a fade. The overlap gives an exact same-frame residual only at the replacement boundary; applying it to later suffix tokens would be unsupported temporal extrapolation. The bridge therefore never modifies the authoritative prefix or suffix token 1+, and it adds no H3 NFE, VAE pass, optical-flow model, image/tensor crossfade, audio change, noise change, mask change, conditioning change, VDN API change, or Spectrum-history change.\n\n`mixed_grid_representation_bridge` reports the measured full/DC/zero-mean residual, centered-transition error before/after transplantation, whether one suffix token was corrected, and the explicit no-op/accept reason. `mixed_grid_transfer` continues to report native, exact-restored and corrected seam metrics. With both experimental representation and validated DC bridges enabled, the decisive pre-high invariant is that corrected/native seam ratios should be approximately 1 across raw, low-pass and spatial-mean metrics. Decoded media remains the release gate; the experimental bridge stays off by default.\n'''
    path.write_text(text)

    doc = ROOT / "docs/representation-seam-bridge.md"
    doc.write_text('''# Experimental exact-overlap representation seam bridge\n\n## Why the affine path was retired\n\nThe previous PR #24 experiment treated the remaining Continuum jump as a camera-like affine residual. Real run `metrics_00276` showed that the implementation did what it was designed to do: only the independently authorized vertical-scale axis was corrected, its final signed residual fell from about `-0.010249` to `+0.002361`, and the protected prefix remained exact. The decoded shrink/top-edge reveal nevertheless remained. That is direct evidence that a good affine fit was a proxy for the discontinuity, not its sufficient cause.\n\nThe stronger signal is the exact-prefix replacement itself. In the same run, the learned 3D upscaler's native learned-prefix→learned-suffix seam measured about `0.545` raw RMS, `0.250` low-pass RMS and `0.184` spatial-mean RMS. Hard-restoring the authoritative prefix raised those to about `0.671`, `0.341` and `0.259` respectively. The exact-prefix splice therefore amplified the seam by roughly `1.23x`, `1.37x` and `1.40x`. The existing DC bridge reduced the mean component, but the corrected splice was still about `1.10x` native raw RMS and `1.15x` native low-pass RMS.\n\nThis identifies a representation-continuity problem at the point where a jointly upscaled sequence is split into an exact native prefix plus learned suffix. It does not require assuming that the learned upscaler has a globally different latent basis, nor does it justify modifying the protected prefix.\n\n## Exact overlap residual transplantation\n\nThe discarded learned prefix is a private calibration overlap. For its final token:\n\n```text\nD = exact_prefix_last - learned_prefix_last\n```\n\n`D` is measured between two target-grid latents representing the same protected frame. It is therefore more direct evidence than registering two different temporal frames and fitting scale/translation.\n\nThe experimental bridge decomposes it into:\n\n```text\nD_dc         = spatial_mean(D)\nD_structural = D - D_dc\n```\n\nThe legacy `suffix_geometric_bridge=true` switch now applies only `D_structural` to the first learned suffix token. `suffix_dc_bridge=true` retains its existing responsibility for `D_dc`. With both enabled:\n\n```text\ncorrected_suffix_0 = learned_suffix_0 + D\n\ncorrected_suffix_0 - exact_prefix_last\n    = learned_suffix_0 - learned_prefix_last\n```\n\nSo the hard exact-prefix replacement preserves the learned upscaler's native boundary transition exactly (modulo output dtype rounding) before the high-resolution sampler resumes. There is no fitted camera model and no post-high affine warp.\n\n## Why only suffix token 0\n\nThe exact overlap proves the residual at the boundary frame. It does not prove that the same full tensor residual should persist into future generated frames. Propagating it over several tokens would require an extrapolation model and risks creating the delayed wobble that the earlier handcrafted three-token fade demonstrated. The current bridge therefore corrects exactly one generated latent token. Fresh target-grid H3 refinement then operates on the coherent boundary state.\n\n## Safety and no-op contract\n\nThe bridge:\n\n- never changes protected-prefix values;\n- never changes suffix token 1 or later;\n- never changes audio, caller noise, masks or conditioning;\n- adds no transformer call/NFE and no VAE call;\n- does not alter VDN API 2 or Spectrum history/forecast accounting;\n- runs in float32 for the residual calculation and returns the original video dtype;\n- returns the original tensor object when disabled or when the measured zero-mean overlap residual is already matched;\n- verifies that transplantation strictly reduces the measured centered boundary error, otherwise it returns the original tensor object.\n\nThe existing one-token DC bridge remains a separate feature and remains enabled by default for Mixed-Grid because it already has decoded-media validation. The representation bridge is Mixed-Grid-only and remains off by default.\n\n## Validation gate\n\nStructural tests can prove the algebraic boundary invariant, exact-prefix preservation, later-suffix preservation, dtype behavior, and unchanged NFE/accounting. They cannot prove perceptual success. The next matched real-media run must keep every workflow input unchanged except enabling both:\n\n```text\nsuffix_dc_bridge = true\nsuffix_geometric_bridge = true\n```\n\nBefore high refinement, `corrected_over_native_seam_rms_ratio`, `corrected_over_native_seam_lowpass_ratio`, and `corrected_over_native_seam_spatial_mean_ratio` should all be approximately `1.0`. After refinement, inspect the final seam metrics and decoded video for the original shrink/top-edge reveal and for any new one-token pulse. Do not enable this bridge by default or mark PR #24 ready until decoded media passes.\n''')


def rewrite_tests() -> None:
    handoff = ROOT / "tests/test_handoff.py"
    handoff.write_text(
        handoff.read_text().rstrip()
        + '''\n\n\ndef test_suffix_geometric_bridge_legacy_flag_is_boolean_and_mixed_grid_only():\n    provider = FakeLearnedProvider()\n    with pytest.raises(ValueError, match="requires mixed-grid Continuum"):\n        ProgressiveTargetInputConfig(\n            source_latent_h=4,\n            source_latent_w=4,\n            exact_prefix_mode="target_sparse_lifter",\n            suffix_geometric_bridge=True,\n        )\n    mixed = ProgressiveTargetInputConfig(\n        source_latent_h=4,\n        source_latent_w=4,\n        transfer_mode="learned_3d",\n        learned_upscaler=provider,\n        exact_prefix_mode="mixed_grid_low_suffix",\n        suffix_geometric_bridge=True,\n    )\n    assert mixed.suffix_geometric_bridge is True\n    with pytest.raises(TypeError, match="must be boolean"):\n        ProgressiveTargetInputConfig(\n            source_latent_h=4,\n            source_latent_w=4,\n            transfer_mode="learned_3d",\n            learned_upscaler=provider,\n            exact_prefix_mode="mixed_grid_low_suffix",\n            suffix_geometric_bridge=1,\n        )\n'''
    )

    mixed = ROOT / "tests/test_mixed_grid.py"
    mixed.write_text(
        mixed.read_text().rstrip()
        + '''\n\n\ndef test_representation_bridge_ui_is_mixed_grid_only():\n    from h3_flow_regenerate.target_sparse_node import (\n        H3ProgressiveMixedGridHandoff,\n        H3ProgressiveTargetSparseHandoff,\n    )\n\n    sparse = H3ProgressiveTargetSparseHandoff.INPUT_TYPES()\n    assert all("suffix_geometric_bridge" not in group for group in sparse.values())\n    mixed = H3ProgressiveMixedGridHandoff.INPUT_TYPES()\n    assert "suffix_geometric_bridge" in mixed.get("optional", {})\n    spec = mixed["optional"]["suffix_geometric_bridge"]\n    assert spec[1]["default"] is False\n'''
    )

    (ROOT / "tests/test_representation_bridge.py").write_text(representation_tests())


def main() -> None:
    restore_pr_files()
    (ROOT / "h3_flow_regenerate/representation_bridge.py").write_text(representation_module())
    rewrite_handoff()
    rewrite_target_sparse_node()
    rewrite_runtime()
    rewrite_readme()
    rewrite_mixed_docs()
    rewrite_tests()


if __name__ == "__main__":
    main()
