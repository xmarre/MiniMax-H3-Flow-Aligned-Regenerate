from __future__ import annotations

import torch

import h3_flow_regenerate.final_geometry as final_geometry


def _initial_geometry(*, source_sy=-0.02):
    return {
        "requested": True,
        "accepted": True,
        "cross_grid_corroboration": {
            "axis_authorized": [False, True, False, False],
            "source_signed_residual": [0.0, source_sy, 0.0, 0.0],
        },
        "temporal_profile": {
            "accepted": True,
            "axis_accepted": [False, True, False, False],
            "axis": [
                {"accepted": False, "active_tokens": 0},
                {
                    "accepted": True,
                    "active_tokens": 2,
                    "applied_envelope": {
                        "kind": "measured_monotonic",
                        "active_tokens": 2,
                        "weights": [1.0, 1.0, 0.0],
                    },
                },
                {"accepted": False, "active_tokens": 0},
                {"accepted": False, "active_tokens": 0},
            ],
        },
        "target_natural_motion_model": {
            "accepted": True,
            "expected_transform": [1.0, 1.0, 0.0, 0.0],
            "dispersion": [0.005, 0.007, 0.25, 0.25],
        },
    }


def _candidate(*, sy=0.98, score=2.0):
    return {
        "accepted": True,
        "axis_applied": [False, True, False, False],
        "axis_evidence_score": [0.0, score, 0.0, 0.0],
        "axis_estimator_floor": [0.005, 0.005, 0.25, 0.25],
        "raw_signed_residual": [0.0, float(torch.log(torch.tensor(sy))), 0.0, 0.0],
        "raw_transform": [1.0, sy, 0.0, 0.0],
    }


def _video(prefix_t=3, suffix_t=5):
    generator = torch.Generator().manual_seed(1987)
    return torch.randn(1, 6, prefix_t + suffix_t, 16, 20, generator=generator)


def test_final_closure_reuses_authorized_axis_and_measured_temporal_envelope(monkeypatch):
    video = _video()
    saved = video.clone()
    calls = 0

    def register_pair(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        sy = 0.98 if calls == 1 else 0.998
        return {"aligned_error": 0.1, "transform": [1.0, sy, 0.0, 0.0]}

    monkeypatch.setattr(final_geometry, "register_pair", register_pair)
    monkeypatch.setattr(final_geometry, "_motion_residual_candidate", lambda *_args, **_kwargs: _candidate())

    corrected, report = final_geometry.close_final_mixed_grid_residual(video, 3, _initial_geometry())

    assert report["accepted"], report
    assert report["axis_applied"] == [False, True, False, False]
    assert report["tokens_corrected"] == 2
    assert report["effective_active_temporal_length"] == 2
    assert report["final_target_signed_residual"][1] < 0
    assert abs(report["final_target_signed_residual_after"][1]) < abs(report["final_target_signed_residual"][1])
    assert torch.equal(corrected[:, :, :3], saved[:, :, :3])
    assert not torch.equal(corrected[:, :, 3:5], saved[:, :, 3:5])
    assert torch.equal(corrected[:, :, 5:], saved[:, :, 5:])
    assert torch.equal(video, saved)


def test_final_closure_never_promotes_target_only_nuisance_axes(monkeypatch):
    video = _video()
    candidate = _candidate()
    candidate.update(
        axis_applied=[True, True, True, True],
        axis_evidence_score=[4.0, 2.0, 4.0, 4.0],
        raw_signed_residual=[0.02, -0.02, 0.5, -0.5],
        raw_transform=[1.0202013400267558, 0.98, 0.5, -0.5],
    )
    calls = 0

    def register_pair(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        sy = 0.98 if calls == 1 else 0.998
        return {"aligned_error": 0.1, "transform": [1.0, sy, 0.0, 0.0]}

    monkeypatch.setattr(final_geometry, "register_pair", register_pair)
    monkeypatch.setattr(final_geometry, "_motion_residual_candidate", lambda *_args, **_kwargs: candidate)

    _, report = final_geometry.close_final_mixed_grid_residual(video, 3, _initial_geometry())

    assert report["accepted"], report
    assert report["axis_applied"] == [False, True, False, False]
    assert report["base_target_transform"] == [1.0, 0.98, 0.0, 0.0] or report["base_target_transform"] == (
        1.0,
        0.98,
        0.0,
        0.0,
    )


def test_final_closure_rejects_sign_flip_or_weak_target_evidence(monkeypatch):
    video = _video()
    for candidate in (_candidate(sy=1.02), _candidate(sy=0.98, score=1.24)):
        monkeypatch.setattr(
            final_geometry,
            "register_pair",
            lambda *_args, **_kwargs: {"aligned_error": 0.1, "transform": [1.0, 0.98, 0.0, 0.0]},
        )
        monkeypatch.setattr(
            final_geometry,
            "_motion_residual_candidate",
            lambda *_args, _candidate=candidate, **_kwargs: _candidate,
        )
        result, report = final_geometry.close_final_mixed_grid_residual(video, 3, _initial_geometry())
        assert result is video
        assert not report["accepted"]
        assert report["reason"] == "no_final_residual_axis_survived"


def test_final_closure_reverts_when_projection_does_not_reduce_residual(monkeypatch):
    video = _video()
    calls = 0

    def register_pair(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        sy = 0.98 if calls == 1 else 0.97
        return {"aligned_error": 0.1, "transform": [1.0, sy, 0.0, 0.0]}

    monkeypatch.setattr(final_geometry, "register_pair", register_pair)
    monkeypatch.setattr(final_geometry, "_motion_residual_candidate", lambda *_args, **_kwargs: _candidate())

    result, report = final_geometry.close_final_mixed_grid_residual(video, 3, _initial_geometry())

    assert result is video
    assert not report["accepted"]
    assert report["reason"] == "final_projection_did_not_reduce_authorized_residual"
    assert report["tokens_corrected"] == 0


def test_final_closure_is_exact_noop_without_initial_authorization():
    video = _video()
    initial = _initial_geometry()
    initial["accepted"] = False

    result, report = final_geometry.close_final_mixed_grid_residual(video, 3, initial)

    assert result is video
    assert not report["accepted"]
    assert report["reason"] == "initial_geometry_not_authorized"
