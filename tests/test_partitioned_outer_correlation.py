from types import SimpleNamespace

import pytest

from h3_flow_regenerate import partitioned_outer
from h3_flow_regenerate.handoff import ProgressiveTargetInputConfig
from h3_flow_regenerate.partitioned_scheduler import PARTITIONED_PROGRESSIVE_KEY
from h3_flow_regenerate.runtime import FLOW_BINDING_KEY, FLOW_REQUEST_ID_KEY, FlowBinding


class _Executor:
    def __init__(self, guider):
        self.class_obj = guider


def _fixture():
    binding = FlowBinding()
    progressive = ProgressiveTargetInputConfig(source_scale=0.5)
    guider = SimpleNamespace(
        model_options={
            FLOW_BINDING_KEY: binding,
            PARTITIONED_PROGRESSIVE_KEY: progressive,
            "transformer_options": {},
        }
    )
    return binding, guider, _Executor(guider)


def test_partitioned_outer_establishes_and_restores_flow_request_correlation(monkeypatch):
    binding, guider, executor = _fixture()
    seen = {}

    monkeypatch.setattr(partitioned_outer, "_has_exact_video_protection", lambda *_args: True)
    monkeypatch.setattr(partitioned_outer, "configured_audio_guided_overlap_ticks", lambda: 0)
    monkeypatch.setattr(
        partitioned_outer,
        "_ProgressiveExactMaskExecutor",
        lambda wrapped, **_kwargs: wrapped,
    )

    sentinel = object()

    def fake_run(_adapted, _guider, active_binding, *_args):
        transformer = _guider.model_options["transformer_options"]
        seen["request_id"] = active_binding.active_request_id
        seen["published_request_id"] = transformer.get(FLOW_REQUEST_ID_KEY)
        seen["evaluation_serial"] = active_binding.evaluation_serial
        return sentinel

    monkeypatch.setattr(partitioned_outer, "run_partitioned_progressive", fake_run)

    result = partitioned_outer.partitioned_outer_wrapper(
        executor,
        object(),
        object(),
        object(),
        object(),
        denoise_mask=object(),
        latent_shapes=[(1, 24, 2, 4, 4), (1, 32, 2, 2)],
    )

    assert result is sentinel
    assert isinstance(seen["request_id"], str) and seen["request_id"].startswith("flow-")
    assert seen["published_request_id"] == seen["request_id"]
    assert seen["evaluation_serial"] == 0
    assert binding.active_request_id is None
    assert binding.evaluation_serial == 0
    assert FLOW_REQUEST_ID_KEY not in guider.model_options["transformer_options"]
    sampler_wall = next(event for event in binding.metrics.events if event.kind == "sampler_wall")
    assert sampler_wall.fields["request_id"] == seen["request_id"]
    assert sampler_wall.fields["partitioned_exact_prefix"] is True


def test_partitioned_outer_restores_request_correlation_after_candidate_failure(monkeypatch):
    binding, guider, executor = _fixture()

    monkeypatch.setattr(partitioned_outer, "_has_exact_video_protection", lambda *_args: True)
    monkeypatch.setattr(partitioned_outer, "configured_audio_guided_overlap_ticks", lambda: 0)
    monkeypatch.setattr(
        partitioned_outer,
        "_ProgressiveExactMaskExecutor",
        lambda wrapped, **_kwargs: wrapped,
    )

    def fail_run(*_args, **_kwargs):
        assert binding.active_request_id is not None
        assert guider.model_options["transformer_options"][FLOW_REQUEST_ID_KEY] == binding.active_request_id
        raise RuntimeError("candidate failure")

    monkeypatch.setattr(partitioned_outer, "run_partitioned_progressive", fail_run)

    with pytest.raises(RuntimeError, match="candidate failure"):
        partitioned_outer.partitioned_outer_wrapper(
            executor,
            object(),
            object(),
            object(),
            object(),
            denoise_mask=object(),
            latent_shapes=[(1, 24, 2, 4, 4), (1, 32, 2, 2)],
        )

    assert binding.active_request_id is None
    assert binding.evaluation_serial == 0
    assert FLOW_REQUEST_ID_KEY not in guider.model_options["transformer_options"]
