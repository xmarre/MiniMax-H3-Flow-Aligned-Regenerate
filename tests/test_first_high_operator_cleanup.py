from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate import first_high_operator_comparison as w


def _state():
    return w._State(replay=SimpleNamespace(), mode="native_window", source_gate={})


def _sampler_args(state):
    return (
        object(),
        torch.tensor([0.8, 0.0]),
        {"model_options": {w.STATE_KEY: state}},
        None,
        torch.zeros(1),
        torch.zeros(1),
        None,
        True,
    )


def test_sampler_boundary_catches_only_owned_completion_and_returns_raw_x0(monkeypatch):
    state = _state()
    call = w._Call(state=state)
    x0 = torch.randn(2, 3)
    monkeypatch.setattr(w._diag, "_stage_name", lambda _options: "high")
    monkeypatch.setattr(w._replay, "_validate_replay_sampler_entry", lambda _replay, _record: None)

    record_token = w._diag._ACTIVE.set(SimpleNamespace())
    call_token = w._ACTIVE_CALL.set(call)
    try:

        def executor(*_args, **_kwargs):
            raise w._FirstCallComplete(call.token, x0)

        result = w._sampler_entry_wrapper(executor, *_sampler_args(state))
    finally:
        w._ACTIVE_CALL.reset(call_token)
        w._diag._ACTIVE.reset(record_token)

    assert result is x0
    assert call.completed is True
    assert call.raw_x0 is x0


def test_sampler_boundary_rethrows_foreign_completion(monkeypatch):
    state = _state()
    call = w._Call(state=state)
    x0 = torch.randn(2, 3)
    monkeypatch.setattr(w._diag, "_stage_name", lambda _options: "high")
    monkeypatch.setattr(w._replay, "_validate_replay_sampler_entry", lambda _replay, _record: None)

    record_token = w._diag._ACTIVE.set(SimpleNamespace())
    call_token = w._ACTIVE_CALL.set(call)
    try:

        def executor(*_args, **_kwargs):
            raise w._FirstCallComplete(object(), x0)

        with pytest.raises(w._FirstCallComplete):
            w._sampler_entry_wrapper(executor, *_sampler_args(state))
    finally:
        w._ACTIVE_CALL.reset(call_token)
        w._diag._ACTIVE.reset(record_token)

    assert call.completed is False
    assert call.raw_x0 is None


def test_sampler_boundary_rejects_missing_or_duplicate_owner(monkeypatch):
    state = _state()
    x0 = torch.randn(2, 3)
    monkeypatch.setattr(w._diag, "_stage_name", lambda _options: "high")
    monkeypatch.setattr(w._replay, "_validate_replay_sampler_entry", lambda _replay, _record: None)
    record_token = w._diag._ACTIVE.set(SimpleNamespace())
    try:
        with pytest.raises(RuntimeError, match="no matching execution-local completion owner"):
            w._sampler_entry_wrapper(lambda *_args, **_kwargs: x0, *_sampler_args(state))

        call = w._Call(state=state, completed=True, raw_x0=x0)
        call_token = w._ACTIVE_CALL.set(call)
        try:

            def executor(*_args, **_kwargs):
                raise w._FirstCallComplete(call.token, x0)

            with pytest.raises(RuntimeError, match="completed more than once"):
                w._sampler_entry_wrapper(executor, *_sampler_args(state))
        finally:
            w._ACTIVE_CALL.reset(call_token)
    finally:
        w._diag._ACTIVE.reset(record_token)


def test_export_contract_proves_actual_core_externalization_not_identity_assumption():
    class Base:
        @staticmethod
        def process_latent_out(value):
            return value * 2.0

    guider = SimpleNamespace(model_patcher=SimpleNamespace(model=Base()))
    raw = torch.tensor([[1.0, -2.0]], dtype=torch.float16)
    expected = raw.to(torch.float32) * 2.0

    report = w._export_contract(guider, raw, expected)
    assert report["exact_core_callback_x0_externalization"] is True
    assert report["process_latent_out_identity_for_this_x0"] is False
    assert report["raw_x0_dtype"] == "torch.float16"
    assert report["returned_dtype"] == "torch.float32"

    changed = w._export_contract(guider, raw, expected + 1.0)
    assert changed["exact_core_callback_x0_externalization"] is False


def test_core_cleanup_contract_requires_outer_sample_owned_state_to_be_released():
    guider = SimpleNamespace()
    options = {}
    report = w._core_cleanup_contract(guider, options)
    assert report == {
        "inner_model_absent": True,
        "loaded_models_absent": True,
        "multigpu_thread_pool_absent": True,
        "complete": True,
    }

    guider.inner_model = object()
    assert w._core_cleanup_contract(guider, options)["complete"] is False
    del guider.inner_model
    options["multigpu_thread_pool"] = object()
    assert w._core_cleanup_contract(guider, options)["complete"] is False
