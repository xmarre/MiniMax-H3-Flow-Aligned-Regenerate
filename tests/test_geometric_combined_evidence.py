from __future__ import annotations

import math

from h3_flow_regenerate.geometric_bridge import (
    _corroborate_target_candidate,
    _motion_residual_candidate,
)


def test_actual_like_source_residual_is_provisional_when_old_standalone_gate_is_impossible():
    motion = {
        "accepted": True,
        "expected_transform": (0.9892341420727254, 1.0025614248754695, -0.9375, 0.0),
        "dispersion": (0.013517017851171641, 0.012961403605976161, 0.2779875, 0.0926625),
    }
    boundary = {
        "aligned_error": 0.09217923249054762,
        "comparison_error": 0.09441049517024853,
        "comparison_improvement": 0.02363362966879188,
        "transform": (0.9849999999999999, 0.9775, -0.75, 0.125),
        "residual_axis_gains": (-0.0002744265, 0.0015844459, 0.0000444795, 0.0001887799),
    }

    report = _motion_residual_candidate(boundary, motion, (40, 54))

    assert report["accepted"], report
    assert report["axis_applied"] == [False, True, False, False]
    assert not report["axis_standalone_significant"][1]
    assert report["axis_standalone_2p5_dispersion_threshold"][1] > math.log(1.03)
    assert 1.8 < report["axis_evidence_score"][1] < 2.1


def test_combined_evidence_still_rejects_two_weak_domains():
    source = {
        "accepted": True,
        "axis_applied": [False, True, False, False],
        "transform": (1.0, 0.99, 0.0, 0.0),
        "axis_evidence_score": [0.0, 1.4, 0.0, 0.0],
    }
    target = {
        "accepted": True,
        "axis_applied": [False, True, False, False],
        "transform": (1.0, 0.99, 0.0, 0.0),
        "axis_evidence_score": [0.0, 1.4, 0.0, 0.0],
    }
    persistence = {"accepted": True, "axis_persistent": [False, True, False, False]}

    report = _corroborate_target_candidate(source, target, persistence, (40, 54), (56, 76))

    assert not report["accepted"]
    assert report["reason"] == "combined_evidence_insufficient"


def test_persistence_is_axis_specific_during_cross_grid_authorization():
    source = {
        "accepted": True,
        "axis_applied": [False, True, True, False],
        "transform": (1.0, 0.98, 0.75, 0.0),
        "axis_evidence_score": [0.0, 3.0, 3.0, 0.0],
    }
    target = {
        "accepted": True,
        "axis_applied": [False, True, True, False],
        "transform": (1.0, 0.98, 0.75 * 76 / 54, 0.0),
        "axis_evidence_score": [0.0, 3.0, 3.0, 0.0],
    }
    persistence = {"accepted": True, "axis_persistent": [False, True, False, False]}

    report = _corroborate_target_candidate(source, target, persistence, (40, 54), (56, 76))

    assert report["accepted"], report
    assert report["axis_applied"] == [False, True, False, False]
    assert report["axis_reason"][2] == "axis_not_shared_or_persistent"
