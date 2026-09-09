from pathlib import Path

path = Path('h3_flow_regenerate/source_trajectory_bridge.py')
text = path.read_text()
old = '''    if all(abs(state - boundary) <= floor for state in states[1:]):
        observed_tokens = min(suffix_length, len(followup_signed_residuals) + 1)
        report.update(
            accepted=True,
            mode="persistent",
            reason="persistent_state_within_estimator_floor_observed_window",
            active_tokens=observed_tokens,
            applied_envelope={
                "kind": "constant",
                "value": 1.0,
                "active_tokens": observed_tokens,
                "measured_followup_transitions": len(followup_signed_residuals),
                "extrapolated_beyond_observation": False,
            },
        )
        return report
'''
new = '''    if all(abs(state - boundary) <= floor for state in states[1:]):
        observed_tokens = min(suffix_length, len(followup_signed_residuals) + 1)
        if observed_tokens < suffix_length:
            report.update(
                mode="persistent",
                reason="persistent_state_without_observed_recovery",
                active_tokens=0,
                applied_envelope={
                    "kind": "none",
                    "active_tokens": 0,
                    "measured_tokens_available": observed_tokens,
                    "measured_followup_transitions": len(followup_signed_residuals),
                    "terminal_recovery_observed": False,
                    "extrapolated_beyond_observation": False,
                },
            )
            return report
        report.update(
            accepted=True,
            mode="persistent",
            reason="persistent_state_observed_through_suffix_end",
            active_tokens=observed_tokens,
            applied_envelope={
                "kind": "constant",
                "value": 1.0,
                "active_tokens": observed_tokens,
                "measured_followup_transitions": len(followup_signed_residuals),
                "terminal_recovery_observed": False,
                "extrapolated_beyond_observation": False,
            },
        )
        return report
'''
if old not in text:
    raise SystemExit('persistent policy anchor missing')
path.write_text(text.replace(old, new, 1))

test = Path('tests/test_source_trajectory_bridge.py')
t = test.read_text()
old_test = '''def test_persistent_source_state_is_limited_to_measured_window():
    frame = _texture()
    prefix = [frame.clone() for _ in range(6)]
    shifted = _warp(frame, sy=1.0 / 0.975)
    suffix = [shifted.clone() for _ in range(7)]
    video = torch.stack([*prefix, *suffix], dim=2)
    before = video.clone()
    corrected, report = apply_source_trajectory_bridge(video, 6, requested=True)
    assert report["source_trajectory_bridge_accepted"] is True
    assert report["source_trajectory_temporal_profile"]["axis_mode"][1] == "persistent"
    envelope = report["source_trajectory_temporal_profile"]["axis"][1]["applied_envelope"]
    assert envelope["measured_followup_transitions"] == 4
    assert envelope["extrapolated_beyond_observation"] is False
    assert report["source_trajectory_bridge_tokens_corrected"] == 5
    assert not torch.equal(corrected[:, :, 6:11], before[:, :, 6:11])
    assert torch.equal(corrected[:, :, 11:], before[:, :, 11:])
'''
new_test = '''def test_persistent_source_state_without_observed_recovery_fails_safe():
    frame = _texture()
    prefix = [frame.clone() for _ in range(6)]
    shifted = _warp(frame, sy=1.0 / 0.975)
    suffix = [shifted.clone() for _ in range(7)]
    video = torch.stack([*prefix, *suffix], dim=2)
    before = video.clone()
    corrected, report = apply_source_trajectory_bridge(video, 6, requested=True)
    assert report["source_trajectory_bridge_accepted"] is False
    assert report["source_trajectory_temporal_profile"]["axis_mode"][1] == "persistent"
    axis = report["source_trajectory_temporal_profile"]["axis"][1]
    assert axis["reason"] == "persistent_state_without_observed_recovery"
    assert axis["applied_envelope"]["measured_followup_transitions"] == 4
    assert axis["applied_envelope"]["extrapolated_beyond_observation"] is False
    assert corrected is video
    assert torch.equal(corrected, before)


def test_persistent_source_state_observed_through_suffix_end_can_apply():
    frame = _texture()
    prefix = [frame.clone() for _ in range(6)]
    shifted = _warp(frame, sy=1.0 / 0.975)
    suffix = [shifted.clone() for _ in range(5)]
    video = torch.stack([*prefix, *suffix], dim=2)
    before = video.clone()
    corrected, report = apply_source_trajectory_bridge(video, 6, requested=True)
    assert report["source_trajectory_bridge_accepted"] is True
    assert report["source_trajectory_temporal_profile"]["axis_mode"][1] == "persistent"
    axis = report["source_trajectory_temporal_profile"]["axis"][1]
    assert axis["reason"] == "persistent_state_observed_through_suffix_end"
    assert report["source_trajectory_bridge_tokens_corrected"] == 5
    assert not torch.equal(corrected[:, :, 6:], before[:, :, 6:])
'''
if old_test not in t:
    raise SystemExit('persistent test anchor missing')
t = t.replace(old_test, new_test, 1)
test.write_text(t)

doc = Path('docs/mixed-grid-seam-repair.md')
d = doc.read_text()
old_doc = '''- **persistent** — the measured cumulative residual remains within one estimator floor of the initial boundary state for at least two follow-up transitions;
'''
new_doc = '''- **persistent** — the measured cumulative residual remains within one estimator floor of the initial boundary state; a persistent correction is applied only when that state is directly observed through the end of the suffix, because truncating an unrecovered persistent correction would merely create a delayed seam at the first uncorrected token;
'''
if old_doc not in d:
    raise SystemExit('persistent doc anchor missing')
d = d.replace(old_doc, new_doc, 1)
old_para = '''Persistent and measured-drift states are deliberately bounded to the directly observed window. Measured-drift additionally reuses the original per-axis geometric safety bound on every cumulative token state and stops at the first token that would leave it. With four measured follow-up transitions, at most suffix token 0 plus those four observed tokens can be corrected, and often fewer. No correction is projected onto unmeasured or safety-rejected later suffix tokens.
'''
new_para = '''Persistent and measured-drift states never extrapolate onto unmeasured tokens. A persistent state that extends beyond the measured window without an observed recovery now fails safe instead of ending a constant correction abruptly and moving the seam to the window boundary; it can be applied only when the directly observed persistent window reaches the suffix end. Measured-drift additionally reuses the original per-axis geometric safety bound on every cumulative token state and stops at the first token that would leave it. With four measured follow-up transitions, at most suffix token 0 plus those four observed tokens can be corrected, and often fewer. No correction is projected onto unmeasured or safety-rejected later suffix tokens.
'''
if old_para not in d:
    raise SystemExit('persistent paragraph anchor missing')
doc.write_text(d.replace(old_para, new_para, 1))
