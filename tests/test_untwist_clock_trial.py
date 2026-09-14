from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate import execution_contract_diagnostics as diag
from h3_flow_regenerate import runtime
from h3_flow_regenerate.untwist_clock_trial import (
    FLOW_SAMPLING_CONTEXT_KEY,
    _build_sampling_context,
    _trial_stage_wrapper,
)

CONTROLLED_SIGMAS = (
    1.0,
    0.9882352948188782,
    0.9729729890823364,
    0.9523809552192688,
    0.9230769276618958,
    0.8780487775802612,
    0.800000011920929,
    0.6315789222717285,
    0.0,
)


def _record():
    state = diag._State(max_bytes=16 * 1024 * 1024, strict_provenance=False, manifest={})
    return diag._Record(
        state=state,
        original_sigmas=CONTROLLED_SIGMAS,
        original_schedule_digest="controlled-digest",
    )


def test_context_maps_split_children_back_to_original_trajectory():
    record = _record()

    low = _build_sampling_context(record, stage="low", child_sigmas=torch.tensor(CONTROLLED_SIGMAS[:6]))
    probe = _build_sampling_context(
        record,
        stage="probe",
        child_sigmas=torch.tensor([CONTROLLED_SIGMAS[5]]),
    )
    high = _build_sampling_context(record, stage="high", child_sigmas=torch.tensor(CONTROLLED_SIGMAS[5:]))

    assert low["original_stage_start_index"] == 0
    assert low["original_stage_start_progress"] == 0.0
    assert probe["original_stage_start_index"] == 5
    assert probe["original_stage_start_progress"] == pytest.approx(5 / 7)
    assert high["original_stage_start_index"] == 5
    assert high["original_stage_start_progress"] == pytest.approx(5 / 7)
    assert high["original_nonzero_sigmas"] == list(CONTROLLED_SIGMAS[:-1])
    assert high["sample_sigmas_ownership"] == "sampler_unchanged"


def test_context_rejects_child_coordinate_outside_original_schedule():
    record = _record()
    with pytest.raises(RuntimeError, match="not a subset"):
        _build_sampling_context(record, stage="high", child_sigmas=torch.tensor([0.777, 0.0]))


def test_stage_wrapper_publishes_context_without_mutating_child_sigmas():
    record = _record()
    original = torch.tensor(CONTROLLED_SIGMAS[5:])
    guider = SimpleNamespace(model_options={"transformer_options": {runtime.FLOW_STAGE_KEY: "high"}})
    calls = []

    class Executor:
        class_obj = guider

        def __call__(self, noise, latent_image, sampler, sigmas, *args, **kwargs):
            calls.append(sigmas.detach().clone())
            context = guider.model_options["transformer_options"][FLOW_SAMPLING_CONTEXT_KEY]
            assert context["original_stage_start_index"] == 5
            assert context["original_nonzero_sigmas"] == list(CONTROLLED_SIGMAS[:-1])
            assert torch.equal(sigmas, original)
            return "ok"

    def sample_euler():
        pass

    sampler = SimpleNamespace(sampler_function=sample_euler)
    token = diag._ACTIVE.set(record)
    try:
        result = _trial_stage_wrapper(
            Executor(),
            torch.zeros(1),
            torch.zeros(1),
            sampler,
            original,
            None,
            None,
            False,
            123,
            [(1, 24, 52, 64, 64), (1, 32, 2, 292)],
        )
    finally:
        diag._ACTIVE.reset(token)

    assert result == "ok"
    assert len(calls) == 1
    assert torch.equal(calls[0], original)
    assert FLOW_SAMPLING_CONTEXT_KEY not in guider.model_options["transformer_options"]


def test_stage_wrapper_restores_context_when_child_raises():
    record = _record()
    guider = SimpleNamespace(model_options={"transformer_options": {runtime.FLOW_STAGE_KEY: "high"}})

    class Executor:
        class_obj = guider

        def __call__(self, *args, **kwargs):
            assert FLOW_SAMPLING_CONTEXT_KEY in guider.model_options["transformer_options"]
            raise ValueError("boom")

    def sample_euler():
        pass

    token = diag._ACTIVE.set(record)
    try:
        with pytest.raises(ValueError, match="boom"):
            _trial_stage_wrapper(
                Executor(),
                torch.zeros(1),
                torch.zeros(1),
                SimpleNamespace(sampler_function=sample_euler),
                torch.tensor(CONTROLLED_SIGMAS[5:]),
                None,
                None,
                False,
                123,
                [(1, 24, 52, 64, 64), (1, 32, 2, 292)],
            )
    finally:
        diag._ACTIVE.reset(token)

    assert FLOW_SAMPLING_CONTEXT_KEY not in guider.model_options["transformer_options"]
