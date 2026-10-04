import pytest
import torch

from h3_flow_regenerate.geometry import pack_streams
from h3_flow_regenerate.high_stage_boundary import HighStageBoundaryTrace
from h3_flow_regenerate.metrics import H3FlowMetrics


def test_trace_detects_motion_and_flow_change_in_fifth_decoder_window_token():
    generator = torch.Generator().manual_seed(958)
    reference = torch.randn(1, 24, 1, 32, 32, generator=generator)
    video = reference.repeat(1, 1, 18, 1, 1)
    video[:, :, 16] = torch.roll(video[:, :, 16], shifts=1, dims=-2)
    # The sixth suffix token is outside this bounded observation.
    video[:, :, 17] = torch.roll(video[:, :, 17], shifts=3, dims=-1)
    audio = torch.randn(1, 32, 2, 24, generator=generator)
    packed, shapes = pack_streams((video, audio))
    original = packed.clone()
    metrics = H3FlowMetrics()
    trace = HighStageBoundaryTrace(metrics, video[:, :, :12], shapes)

    trace.observe(packed, point="before_flow", call_index=0, sigma=0.8, actual=True)

    before = metrics.events[-1].fields
    assert before["suffix_tokens"] == 5
    for trajectory in before["trajectories"].values():
        assert trajectory["trajectory_forward_steps"] == 5
        assert len(trajectory["pairwise_dy"]) == 5
        assert trajectory["pairwise_dy"][:4] == pytest.approx([0.0] * 4, abs=1e-5)
    assert before["trajectories"]["full"]["pairwise_dy"][4] == pytest.approx(1.0, abs=0.07)

    changed_video = video.clone()
    changed_video[:, :, 16].add_(0.5)
    changed, _ = pack_streams((changed_video, audio))
    changed_original = changed.clone()
    trace.observe(changed, point="after_flow", call_index=0, sigma=0.8, actual=True)

    after = metrics.events[-1].fields
    assert after["flow_suffix_delta_rms"] == pytest.approx(0.5 / 5**0.5, abs=1e-6)
    assert trace.previous_prediction is None
    assert torch.equal(packed, original)
    assert torch.equal(changed, changed_original)
    assert torch.equal(video[:, :, :12], reference.repeat(1, 1, 12, 1, 1))


@pytest.mark.parametrize("suffix_tokens", [1, 3])
def test_trace_bounds_decoder_window_to_available_suffix(suffix_tokens):
    video = torch.zeros(1, 24, 12 + suffix_tokens, 16, 16)
    audio = torch.zeros(1, 32, 2, 24)
    packed, shapes = pack_streams((video, audio))
    original = packed.clone()
    metrics = H3FlowMetrics()
    trace = HighStageBoundaryTrace(metrics, video[:, :, :12], shapes)

    trace.observe(packed, point="before_flow", call_index=0, sigma=0.8, actual=True)

    fields = metrics.events[-1].fields
    assert fields["suffix_tokens"] == suffix_tokens
    for trajectory in fields["trajectories"].values():
        assert trajectory["trajectory_forward_steps"] == suffix_tokens
        assert len(trajectory["pairwise_dy"]) == suffix_tokens
    assert torch.equal(packed, original)
