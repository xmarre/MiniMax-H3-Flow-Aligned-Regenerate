import pytest
import torch

from h3_flow_regenerate.contracts import H3FlowTrajectory, TrajectorySample
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


def test_abort_never_publishes_partial_run_and_retry_works():
    trajectory = H3FlowTrajectory()
    run_id = begin(trajectory)
    trajectory.append(run_id, sample())
    trajectory.abort(run_id, "cancelled")
    assert trajectory.runs == ()
    retry = begin(trajectory)
    trajectory.append(retry, sample(2))
    assert trajectory.commit(retry).samples[0].video_x0.mean() == 2


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
