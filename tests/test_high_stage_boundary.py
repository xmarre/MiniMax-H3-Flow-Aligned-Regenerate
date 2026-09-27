from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate.geometry import pack_streams
from h3_flow_regenerate.high_stage_boundary import high_boundary_contract
from h3_flow_regenerate.metrics import H3FlowMetrics


def test_boundary_trace_is_bounded_nonmutating_and_cleans_up_on_exception():
    video = torch.randn(1, 24, 12, 16, 16)
    audio = torch.randn(1, 32, 2, 40)
    packed, shapes = pack_streams((video, audio))
    original = packed.clone()
    binding = SimpleNamespace(guidance_protected_prefix_t=0, high_boundary_trace=None, metrics=H3FlowMetrics())
    with (
        pytest.raises(RuntimeError, match="sampler failure"),
        high_boundary_contract(binding, video[:, :, :3], shapes, measure=True),
    ):
        assert binding.guidance_protected_prefix_t == 3
        trace = binding.high_boundary_trace
        for point in ("before_flow", "after_flow"):
            trace.observe(packed, point=point, call_index=0, sigma=0.5, actual=False)
        assert trace.previous_prediction is None
        trace.observe(packed, point="before_flow", call_index=16, sigma=0.1, actual=True)
        assert len(binding.metrics.events) == 2
        assert binding.metrics.events[-1].fields["flow_suffix_delta_rms"] == 0.0
        assert binding.metrics.events[-1].fields["suffix_tokens"] == 4
        assert binding.metrics.events[-1].fields["actual"] is False
        assert torch.equal(packed, original)
        raise RuntimeError("sampler failure")
    assert binding.guidance_protected_prefix_t == 0
    assert binding.high_boundary_trace is None


def test_ownership_without_measurement_and_nested_rejection():
    binding = SimpleNamespace(guidance_protected_prefix_t=0, high_boundary_trace=None, metrics=H3FlowMetrics())
    prefix = torch.zeros(1, 24, 3, 4, 4)
    with high_boundary_contract(binding, prefix, [], measure=False):
        assert binding.high_boundary_trace is None
        with (
            pytest.raises(RuntimeError, match="nested"),
            high_boundary_contract(binding, prefix, [], measure=False),
        ):
            pass
        assert binding.guidance_protected_prefix_t == 3
    assert binding.guidance_protected_prefix_t == 0
