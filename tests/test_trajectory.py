from dataclasses import replace

import pytest
import torch

from h3_flow_regenerate.contracts import H3FlowTrajectory, TrajectorySample
from h3_flow_regenerate.nodes import H3FlowTrajectoryNode
from h3_flow_regenerate.geometry import geometry_from_video


def sample(value=1.0, provenance="actual", phase="corrected"):
    return TrajectorySample(
        coordinate=0.5,
        video_sigma=0.9,
        audio_sigma=0.7,
        outer_step=1,
        call_index=1,
        phase=phase,
        provenance=provenance,
        video_x0=torch.full((1, 24, 1, 4, 4), value),
    )


def begin(trajectory, session="s", chunk="0"):
    return trajectory.begin(
        session_id=session,
        chunk_id=chunk,
        sampler="sa_solver_pece",
        scheduler="abc",
        geometry=geometry_from_video(torch.zeros(1, 24, 1, 4, 4)),
        audio_shape=(1, 32, 2, 8),
        layout_signature="layout",
        conditioning_signature="cond",
    )


def test_trajectory_node_forces_fresh_prompt_local_handle():
    changed = H3FlowTrajectoryNode.IS_CHANGED("system_ram", 16)
    assert changed != changed  # NaN is ComfyUI's always-changed cache fingerprint.
    first = H3FlowTrajectoryNode().create("system_ram", 16)[0]
    second = H3FlowTrajectoryNode().create("system_ram", 16)[0]
    assert first is not second


def test_commit_publishes_exact_provenance_and_copies_storage():
    trajectory = H3FlowTrajectory(storage="system_ram")
    run_id = begin(trajectory)
    original = sample()
    trajectory.append(run_id, original)
    run = trajectory.commit(run_id)
    original.video_x0.zero_()
    assert run.complete
    assert run.exact_samples()[0].provenance == "actual"
    assert torch.all(run.samples[0].video_x0 == 1)


def test_abort_preserves_diagnostics_but_never_becomes_guidance_state():
    trajectory = H3FlowTrajectory()
    run_id = begin(trajectory)
    trajectory.append(run_id, sample())
    aborted = trajectory.abort(run_id, "cancelled")
    assert not aborted.complete
    assert aborted.abort_reason == "cancelled"
    assert len(trajectory.runs) == 1
    with pytest.raises(RuntimeError, match="latest matching trajectory is incomplete"):
        trajectory.select()

    retry = begin(trajectory)
    trajectory.append(retry, sample(2))
    completed = trajectory.commit(retry)
    assert completed.samples[0].video_x0.mean() == 2
    assert trajectory.select().run_id == completed.run_id
    assert trajectory.latest.run_id == completed.run_id


def test_committed_run_can_be_invalidated_after_downstream_failure():
    trajectory = H3FlowTrajectory(max_runs=2)
    run_id = begin(trajectory)
    trajectory.append(run_id, sample())
    committed = trajectory.commit(run_id)
    assert committed.complete
    invalid = trajectory.invalidate(run_id, "high stage failed")
    assert not invalid.complete
    assert invalid.abort_reason == "high stage failed"
    with pytest.raises(RuntimeError, match="latest trajectory run is incomplete"):
        _ = trajectory.latest


def test_newer_incomplete_run_blocks_stale_complete_fallback():
    trajectory = H3FlowTrajectory()
    old = begin(trajectory, "session", "chunk")
    trajectory.append(old, sample(1))
    trajectory.commit(old)

    failed = begin(trajectory, "session", "chunk")
    trajectory.append(failed, sample(2))
    trajectory.abort(failed, "new attempt failed")

    with pytest.raises(RuntimeError, match="latest matching trajectory is incomplete"):
        trajectory.select(session_id="session", chunk_id="chunk")
    with pytest.raises(RuntimeError, match="latest trajectory run is incomplete"):
        _ = trajectory.latest


def test_completed_forecast_only_run_is_not_selectable_for_guidance():
    trajectory = H3FlowTrajectory()
    run_id = trajectory.begin(
        session_id="s",
        chunk_id="0",
        sampler="euler",
        scheduler="schedule",
        geometry=geometry_from_video(torch.zeros(1, 24, 1, 4, 4)),
        audio_shape=(1, 32, 2, 8),
        layout_signature="layout",
        conditioning_signature="cond",
    )
    forecast = sample(1)
    trajectory.append(run_id, replace(forecast, provenance="forecast"))
    trajectory.commit(run_id)
    assert trajectory.runs[-1].complete
    with pytest.raises(RuntimeError, match="no exact anchors"):
        trajectory.select(session_id="s", chunk_id="0", conditioning_signature="cond")


def test_conditioning_signature_isolates_interleaved_chunk_runs():
    trajectory = H3FlowTrajectory(max_runs=8)

    first = trajectory.begin(
        session_id="continuum",
        chunk_id="0",
        sampler="euler",
        scheduler="schedule",
        geometry=geometry_from_video(torch.zeros(1, 24, 1, 4, 4)),
        audio_shape=(1, 32, 2, 8),
        layout_signature="layout",
        conditioning_signature="cond-a",
    )
    trajectory.append(first, sample(1))
    trajectory.commit(first)

    second = trajectory.begin(
        session_id="continuum",
        chunk_id="0",
        sampler="euler",
        scheduler="schedule",
        geometry=geometry_from_video(torch.zeros(1, 24, 1, 4, 4)),
        audio_shape=(1, 32, 2, 8),
        layout_signature="layout",
        conditioning_signature="cond-b",
    )
    trajectory.append(second, sample(2))
    trajectory.commit(second)

    assert (
        trajectory.select(
            session_id="continuum",
            chunk_id="0",
            conditioning_signature="cond-a",
        )
        .samples[0]
        .video_x0.mean()
        == 1
    )
    assert (
        trajectory.select(
            session_id="continuum",
            chunk_id="0",
            conditioning_signature="cond-b",
        )
        .samples[0]
        .video_x0.mean()
        == 2
    )


def test_chunk_and_session_isolation():
    trajectory = H3FlowTrajectory()
    for session, chunk, value in [("a", "0", 1), ("a", "1", 2), ("b", "0", 3)]:
        run_id = begin(trajectory, session, chunk)
        trajectory.append(run_id, sample(value))
        trajectory.commit(run_id)
    assert trajectory.select(session_id="a", chunk_id="0").samples[0].video_x0.mean() == 1
    assert trajectory.select(session_id="b", chunk_id="0").samples[0].video_x0.mean() == 3


def test_stale_transaction_and_nested_begin_rejected():
    trajectory = H3FlowTrajectory()
    run_id = begin(trajectory)
    with pytest.raises(RuntimeError, match="active transaction"):
        begin(trajectory)
    with pytest.raises(RuntimeError, match="stale"):
        trajectory.append("wrong", sample())
    trajectory.abort(run_id, "test")
