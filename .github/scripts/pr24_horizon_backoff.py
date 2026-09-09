from pathlib import Path

source_path = Path("h3_flow_regenerate/source_trajectory_bridge.py")
text = source_path.read_text()

start = text.index("def _profile_with_axis_mask(")
end = text.index("\ndef _verify_source_warp(", start)
replacement = '''def _profile_with_axis_limits(
    profile: dict,
    axis_mask: list[bool],
    active_limits: list[int],
) -> dict:
    """Return a profile view with only selected axes and verified temporal horizons."""
    masked = copy.deepcopy(profile)
    selected = [bool(value) for value in axis_mask]
    limits = [max(0, int(value)) for value in active_limits]
    for axis, keep in enumerate(selected):
        if axis >= len(masked.get("axis", [])):
            selected[axis] = False
            continue
        axis_report = masked["axis"][axis]
        if not keep:
            axis_report["accepted"] = False
            continue
        original_active = int(axis_report.get("active_tokens", 0))
        limit = min(original_active, limits[axis])
        if limit <= 0:
            axis_report["accepted"] = False
            selected[axis] = False
            continue
        axis_report["active_tokens"] = limit
        envelope = axis_report.get("applied_envelope")
        if isinstance(envelope, dict):
            envelope["active_tokens"] = limit
            if envelope.get("kind") == "measured_cumulative":
                envelope["signed_states"] = list(envelope.get("signed_states", []))[:limit]
            elif envelope.get("kind") == "measured_monotonic":
                envelope["weights"] = list(envelope.get("weights", []))[:limit]
        if limit < original_active:
            axis_report["verification_horizon_limited"] = True
            axis_report["verification_horizon_original_tokens"] = original_active
    masked["axis_accepted"] = selected
    return masked

'''
text = text[:start] + replacement + text[end + 1 :]

loop_start = text.index("    attempted_axis_mask = [bool(value) for value in authorized]\n")
loop_end = text.index("    metrics.update(\n        source_trajectory_axis_post_verified", loop_start)
new_loop = '''    initial_axis_limits = [
        int(profile["axis"][axis].get("active_tokens", 0)) if bool(authorized[axis]) else 0
        for axis in range(4)
    ]
    attempted_axis_limits = list(initial_axis_limits)
    attempted_axis_mask = [bool(authorized[axis]) and attempted_axis_limits[axis] > 0 for axis in range(4)]
    verification_rounds = []
    final_verification = None
    final_profile = None
    corrected = video
    warp = {
        "tokens_corrected": 0,
        "effective_active_temporal_length": 0,
        "applied_transforms": [],
        "applied_transform_sequence_length": 0,
        "applied_transforms_compact": False,
        "out_of_bounds_fraction": 0.0,
    }
    accepted = False
    # Every failed finite-horizon axis gets a chance to retreat by one directly
    # observed token and is then re-warped/re-verified from the untouched source.
    # Boundary failures cannot be repaired by shortening and are pruned immediately.
    max_rounds = 1 + sum(initial_axis_limits)
    for round_index in range(max_rounds):
        if not any(attempted_axis_mask):
            break
        attempt_profile = _profile_with_axis_limits(profile, attempted_axis_mask, attempted_axis_limits)
        attempted_axis_mask = [bool(value) for value in attempt_profile.get("axis_accepted", attempted_axis_mask)]
        if not any(attempted_axis_mask):
            break
        base_transform = (
            candidate_transform[0] if attempted_axis_mask[0] else 1.0,
            candidate_transform[1] if attempted_axis_mask[1] else 1.0,
            candidate_transform[2] if attempted_axis_mask[2] else 0.0,
            candidate_transform[3] if attempted_axis_mask[3] else 0.0,
        )
        corrected, warp = _warp_suffix(video, p, base_transform, attempt_profile)
        if corrected is video:
            final_profile = attempt_profile
            break
        verification = _verify_source_warp(
            video,
            corrected,
            p,
            motion,
            candidate,
            attempt_profile,
            attempted_axis_mask,
            warp,
        )
        verification_rounds.append(
            {
                "round": round_index + 1,
                "axis_attempted": list(attempted_axis_mask),
                "axis_active_tokens": list(attempted_axis_limits),
                "axis_verified": list(verification["axis_verified"]),
                "tokens_corrected": int(warp["tokens_corrected"]),
                "residual_reduction_ratio": verification["residual_reduction_ratio"],
                "transition_count": len(verification["transition_verification"]),
            }
        )
        final_verification = verification
        final_profile = attempt_profile
        if verification["ok"]:
            accepted = True
            break

        next_mask = list(attempted_axis_mask)
        next_limits = list(attempted_axis_limits)
        changed = False
        boundary_verified = verification.get("boundary_axis_verified", [None] * 4)
        for axis in range(4):
            if not attempted_axis_mask[axis] or bool(verification["axis_verified"][axis]):
                continue
            if boundary_verified[axis] is True and next_limits[axis] > 1:
                next_limits[axis] -= 1
                changed = True
            else:
                next_mask[axis] = False
                next_limits[axis] = 0
                changed = True
        if not changed:
            break
        attempted_axis_mask = next_mask
        attempted_axis_limits = next_limits

    final_axis_mask = list(attempted_axis_mask) if accepted else [False] * 4
    final_axis_limits = list(attempted_axis_limits) if accepted else [0] * 4
    pruned_axes = [bool(authorized[axis]) and not final_axis_mask[axis] for axis in range(4)]
'''
text = text[:loop_start] + new_loop + text[loop_end:]

old_metrics = '''    metrics.update(
        source_trajectory_axis_post_verified=final_axis_mask,
        source_trajectory_axis_pruned_post_verification=pruned_axes,
        source_trajectory_post_verification_rounds=verification_rounds,
        source_trajectory_base_transform=base_transform,
    )
'''
new_metrics = '''    metrics.update(
        source_trajectory_axis_post_verified=final_axis_mask,
        source_trajectory_axis_pruned_post_verification=pruned_axes,
        source_trajectory_axis_active_tokens_authorized=initial_axis_limits,
        source_trajectory_axis_active_tokens_post_verified=final_axis_limits,
        source_trajectory_post_verification_rounds=verification_rounds,
        source_trajectory_base_transform=base_transform,
    )
'''
assert old_metrics in text
text = text.replace(old_metrics, new_metrics, 1)

old_reject = '''    if not accepted or final_profile is None or corrected is video:
        metrics["source_trajectory_bridge_reason"] = (
            "post_warp_residual_verification_failed"
            if final_verification is not None
            else "no_nonidentity_source_transform"
        )
        return video, metrics
'''
new_reject = '''    if not accepted or final_profile is None or corrected is video:
        if final_verification is not None:
            metrics.update(
                source_trajectory_bridge_attempted_tokens_corrected=int(warp["tokens_corrected"]),
                source_trajectory_bridge_attempted_transforms=warp["applied_transforms"],
                source_trajectory_bridge_attempted_out_of_bounds_fraction=float(warp["out_of_bounds_fraction"]),
                source_trajectory_bridge_tokens_corrected=0,
                source_trajectory_bridge_effective_active_temporal_length=0,
                source_trajectory_bridge_applied_transforms=[],
                source_trajectory_bridge_applied_transform_sequence_length=0,
                source_trajectory_bridge_applied_transforms_compact=False,
                source_trajectory_bridge_out_of_bounds_fraction=0.0,
            )
        metrics["source_trajectory_bridge_reason"] = (
            "post_warp_residual_verification_failed"
            if final_verification is not None
            else "no_nonidentity_source_transform"
        )
        return video, metrics
'''
assert old_reject in text
text = text.replace(old_reject, new_reject, 1)
source_path.write_text(text)


test_path = Path("tests/test_source_trajectory_bridge.py")
tests = test_path.read_text()
addition = r'''


def test_failed_trailing_transition_backs_off_axis_horizon_before_pruning(monkeypatch):
    video = torch.zeros(1, 24, 8, 8, 8)
    prefix_t = 3
    motion = {
        "accepted": True,
        "reason": "robust_recent_motion",
        "expected_transform": (1.0, 1.0, 0.0, 0.0),
    }
    boundary = {"aligned_error": 0.1, "transform": (1.0, 1.0, 0.0, 0.0)}
    candidate = {
        "accepted": True,
        "reason": "provisional_motion_residual",
        "axis_applied": [False, False, True, False],
        "raw_signed_residual": (0.0, 0.0, 0.5, 0.0),
        "transform": (1.0, 1.0, 0.5, 0.0),
        "axis_estimator_floor": (0.005, 0.005, 0.25, 0.25),
    }
    profile = {
        "accepted": True,
        "reason": "source_evidence_and_temporal_state_authorized",
        "axis_accepted": [False, False, True, False],
        "axis_mode": ["inactive", "inactive", "measured_drift", "inactive"],
        "axis": [
            {"accepted": False, "mode": "inactive"},
            {"accepted": False, "mode": "inactive"},
            {
                "accepted": True,
                "mode": "measured_drift",
                "active_tokens": 4,
                "applied_envelope": {
                    "kind": "measured_cumulative",
                    "signed_states": [0.5, 0.6, 0.7, 0.8],
                    "active_tokens": 4,
                },
            },
            {"accepted": False, "mode": "inactive"},
        ],
        "transitions": [],
    }
    monkeypatch.setattr(source_bridge, "_motion_model", lambda *args, **kwargs: motion)
    monkeypatch.setattr(source_bridge, "register_pair", lambda *args, **kwargs: boundary)
    monkeypatch.setattr(source_bridge, "_motion_residual_candidate", lambda *args, **kwargs: candidate)
    monkeypatch.setattr(source_bridge, "_temporal_profile", lambda *args, **kwargs: profile)

    seen_limits = []

    def fake_warp(source, p, base_transform, attempt_profile):
        active = int(attempt_profile["axis"][2]["active_tokens"])
        seen_limits.append(active)
        corrected = source.clone()
        corrected[:, :, p : p + active] += 0.001
        return corrected, {
            "tokens_corrected": active,
            "effective_active_temporal_length": active,
            "applied_transforms": [base_transform] * active,
            "applied_transform_sequence_length": active,
            "applied_transforms_compact": active > 1,
            "out_of_bounds_fraction": 0.0,
        }

    def fake_verify(source, corrected, p, motion_arg, candidate_arg, attempt_profile, axis_mask, warp):
        active = int(attempt_profile["axis"][2]["active_tokens"])
        ok = active <= 2
        return {
            "ok": ok,
            "axis_verified": [False, False, ok, False],
            "boundary_axis_verified": [None, None, True, None],
            "boundary_after": boundary,
            "post_signed_residual": (0.0, 0.0, 0.1, 0.0),
            "residual_reduction_ratio": [None, None, 0.2, None],
            "transition_verification": [],
        }

    monkeypatch.setattr(source_bridge, "_warp_suffix", fake_warp)
    monkeypatch.setattr(source_bridge, "_verify_source_warp", fake_verify)

    corrected, report = source_bridge.apply_source_trajectory_bridge(video, prefix_t, requested=True)
    assert seen_limits == [4, 3, 2]
    assert report["source_trajectory_bridge_accepted"] is True
    assert report["source_trajectory_axis_active_tokens_authorized"] == [0, 0, 4, 0]
    assert report["source_trajectory_axis_active_tokens_post_verified"] == [0, 0, 2, 0]
    assert report["source_trajectory_axis_post_verified"] == [False, False, True, False]
    assert [item["axis_active_tokens"] for item in report["source_trajectory_post_verification_rounds"]] == [
        [0, 0, 4, 0],
        [0, 0, 3, 0],
        [0, 0, 2, 0],
    ]
    assert torch.equal(corrected[:, :, :prefix_t], video[:, :, :prefix_t])
    assert not torch.equal(corrected[:, :, prefix_t : prefix_t + 2], video[:, :, prefix_t : prefix_t + 2])
    assert torch.equal(corrected[:, :, prefix_t + 2 :], video[:, :, prefix_t + 2 :])


def test_rejected_source_attempt_reports_zero_applied_tokens(monkeypatch):
    video = torch.zeros(1, 24, 6, 8, 8)
    prefix_t = 3
    motion = {"accepted": True, "reason": "robust_recent_motion", "expected_transform": (1.0, 1.0, 0.0, 0.0)}
    boundary = {"aligned_error": 0.1, "transform": (1.0, 1.0, 0.0, 0.0)}
    candidate = {
        "accepted": True,
        "reason": "provisional_motion_residual",
        "axis_applied": [True, False, False, False],
        "raw_signed_residual": (0.02, 0.0, 0.0, 0.0),
        "transform": (float(torch.exp(torch.tensor(0.02))), 1.0, 0.0, 0.0),
        "axis_estimator_floor": (0.005, 0.005, 0.25, 0.25),
    }
    profile = {
        "accepted": True,
        "reason": "source_evidence_and_temporal_state_authorized",
        "axis_accepted": [True, False, False, False],
        "axis_mode": ["measured_drift", "inactive", "inactive", "inactive"],
        "axis": [
            {
                "accepted": True,
                "mode": "measured_drift",
                "active_tokens": 1,
                "applied_envelope": {"kind": "measured_cumulative", "signed_states": [0.02], "active_tokens": 1},
            },
            {"accepted": False},
            {"accepted": False},
            {"accepted": False},
        ],
        "transitions": [],
    }
    monkeypatch.setattr(source_bridge, "_motion_model", lambda *args, **kwargs: motion)
    monkeypatch.setattr(source_bridge, "register_pair", lambda *args, **kwargs: boundary)
    monkeypatch.setattr(source_bridge, "_motion_residual_candidate", lambda *args, **kwargs: candidate)
    monkeypatch.setattr(source_bridge, "_temporal_profile", lambda *args, **kwargs: profile)

    def fake_warp(source, p, base_transform, attempt_profile):
        corrected = source.clone()
        corrected[:, :, p : p + 1] += 0.001
        return corrected, {
            "tokens_corrected": 1,
            "effective_active_temporal_length": 1,
            "applied_transforms": [base_transform],
            "applied_transform_sequence_length": 1,
            "applied_transforms_compact": False,
            "out_of_bounds_fraction": 0.01,
        }

    monkeypatch.setattr(source_bridge, "_warp_suffix", fake_warp)
    monkeypatch.setattr(
        source_bridge,
        "_verify_source_warp",
        lambda *args, **kwargs: {
            "ok": False,
            "axis_verified": [False, False, False, False],
            "boundary_axis_verified": [False, None, None, None],
            "boundary_after": boundary,
            "post_signed_residual": (0.03, 0.0, 0.0, 0.0),
            "residual_reduction_ratio": [1.5, None, None, None],
            "transition_verification": [],
        },
    )

    corrected, report = source_bridge.apply_source_trajectory_bridge(video, prefix_t, requested=True)
    assert corrected is video
    assert report["source_trajectory_bridge_accepted"] is False
    assert report["source_trajectory_bridge_attempted_tokens_corrected"] == 1
    assert report["source_trajectory_bridge_tokens_corrected"] == 0
    assert report["source_trajectory_bridge_applied_transforms"] == []
    assert report["source_trajectory_bridge_out_of_bounds_fraction"] == 0.0
'''
if "test_failed_trailing_transition_backs_off_axis_horizon_before_pruning" not in tests:
    test_path.write_text(tests + addition)


doc_path = Path("docs/mixed-grid-seam-repair.md")
doc = doc_path.read_text()
marker = "## Source-trajectory repair\n"
insert = '''## `metrics_00314`: every boundary axis passed, but finite-horizon termination rejected all four\n\nThe matched `00314` run again returned the original source tensor: all four axes were temporally authorized, and all four passed the boundary residual check, but later directly affected transitions caused every axis to fail post verification. The final telemetry was therefore `source_trajectory_bridge_accepted=false` and `learned_upscaler_input_modified=false`.\n\nThe failure pattern is horizon-dependent rather than a boundary failure. `sx` and `sy` were already safety-limited to one token and failed the trailing token-0 -> token-1 transition, so those one-token corrections cannot be shortened and must be pruned. `tx` was authorized for four measured tokens and first failed at its trailing corrected->untouched transition; `ty` was authorized for five measured tokens and likewise failed at the first untouched transition. Rolling those longer axes back completely discards shorter directly measured horizons that may still satisfy the same verification contract.\n\nPost verification therefore now performs **monotonic temporal-horizon backoff per axis**. A boundary failure prunes the axis immediately because shortening cannot change token 0. A later transition failure on an axis with more than one active token shortens that axis by exactly one directly observed token, rebuilds the warp from the untouched source tensor, and re-runs the full boundary plus affected-transition verification. No weights, thresholds, safety limits, or sign-crossing rules change. Backoff stops at the first fully verified combined correction or at zero surviving axes.\n\nThis is not a fade and does not extrapolate: it only removes already-authorized tail tokens. With the current four-axis/four-follow-up measurement window the search is finite and bounded by the sum of the initially authorized active-token counts.\n\nRejected attempts now also report zero `source_trajectory_bridge_tokens_corrected`, because the returned tensor is byte-identical to the input; attempted warp diagnostics are retained separately under `source_trajectory_bridge_attempted_*`.\n\n'''
if "## `metrics_00314`" not in doc:
    assert marker in doc
    doc_path.write_text(doc.replace(marker, insert + marker, 1))
