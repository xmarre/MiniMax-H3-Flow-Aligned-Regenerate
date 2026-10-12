"""Bounded real Core H3 token-admission diagnostic contracts."""

import sys
from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate.native_token_admission import (
    CoreTokenAdmissionObserver,
    capture_token_admission,
)


def _fixture(*, count=6, tail=(991, 997)):
    # Two query blocks form one token group. Use a fake CPU workspace to test
    # logical slicing without requiring CUDA; the production entrypoint requires it.
    ws = torch.zeros(4096, dtype=torch.uint8)
    plan = {"NQ": 2, "tokCnt": 0, "tokHist": 64, "tokIdx": 1024}
    ws[:4] = torch.tensor([count], dtype=torch.int32).view(torch.uint8)
    hist = torch.zeros((1, 1, 128), dtype=torch.uint32)
    hist[..., -1] = 6
    ws[64 : 64 + hist.numel() * 4] = hist.reshape(-1).view(torch.uint8)
    ids = torch.tensor([[[13, 4, 2, 9, *tail]]], dtype=torch.int32)
    ws[1024 : 1024 + ids.numel() * 4] = ids.reshape(-1).view(torch.uint8)
    return {"ws": ws, "p": plan, "t": 66, "h": 1, "token_aug": 6}


def test_actual_workspace_counts_and_token_set():
    value = capture_token_admission(_fixture(), require_cuda=False)
    assert value["active"] and value["complete"]
    assert value["max_reservations"] == 6
    assert value["at_budget_groups"] == 1
    assert value["overflow_groups"] == 0
    assert value["max_window_candidates"] == 6
    assert value["max_top_bin"] == 6
    assert len(value["retained_ids_sha256"]) == 64


def test_ignored_unwritten_id_tails_and_changed_retained_ids():
    a = capture_token_admission(_fixture(count=4, tail=(991, 997)), require_cuda=False)
    b = capture_token_admission(_fixture(count=4, tail=(1, 2)), require_cuda=False)
    assert a["retained_ids_sha256"] == b["retained_ids_sha256"]
    c = _fixture(count=4)
    c["ws"][1024:1028] = torch.tensor([99], dtype=torch.int32).view(torch.uint8)
    assert capture_token_admission(c, require_cuda=False)["retained_ids_sha256"] != a["retained_ids_sha256"]
    assert a["overflow_groups"] == 0
    overflowing = capture_token_admission(_fixture(count=7), require_cuda=False)
    assert overflowing["overflow_groups"] == 1


def test_cpu_is_rejected_in_actual_cuda_observation():
    with pytest.raises(ValueError, match="unqualified Core sparse"):
        capture_token_admission(_fixture())


def test_bad_workspace_bounds_are_not_silently_read():
    frame = _fixture()
    frame["p"]["tokIdx"] = 4096
    with pytest.raises(ValueError, match="workspace slot"):
        capture_token_admission(frame, require_cuda=False)


def test_profile_event_requires_installed_core_call(monkeypatch):
    from h3_flow_regenerate import native_token_admission as token

    observer = CoreTokenAdmissionObserver()
    record = {"calls": 0, "complete": False}
    frame = SimpleNamespace(
        f_code=SimpleNamespace(
            co_name="sol_attn_chunked",
            co_filename="/opt/site-packages/comfy_kitchen/backends/cuda/__init__.py",
        ),
        f_locals={},
    )
    operation = SimpleNamespace(__name__="sol_attn_core")
    monkeypatch.setattr(token.importlib.metadata, "version", lambda name: "0.2.37")
    monkeypatch.setattr(token, "capture_token_admission", lambda _locals: {"complete": True, "token_aug": 256})
    observer.record_event(record, frame, "c_return", SimpleNamespace(__name__="other"))
    assert record["calls"] == 0
    observer.record_event(record, frame, "c_return", operation)
    assert record["calls"] == 1 and record["complete"]
    observer.record_event(record, frame, "c_return", operation)
    assert record["calls"] == 2 and record["unobserved"] == "multiple_core_calls"


def test_thread_profile_restores_on_failure_and_missing_call():
    observer = CoreTokenAdmissionObserver()
    old = sys.getprofile()
    try:
        with pytest.raises(ZeroDivisionError), observer.observe("A"):
            raise ZeroDivisionError()
        assert sys.getprofile() is old
        assert observer.records["A"]["complete"] is False
        assert observer.records["A"]["unobserved"] == "missing_or_multiple_core_calls"
    finally:
        sys.setprofile(old)


def test_comparisons_require_matched_true_complete_calls():
    observer = CoreTokenAdmissionObserver()
    base = {
        "calls": 1,
        "complete": True,
        "active": True,
        "t": 66,
        "h": 1,
        "token_aug": 256,
        "reservations_sha256": "a",
        "histogram_sha256": "b",
        "retained_ids_sha256": "c",
    }
    observer.records = {
        "A": base,
        "B_target_projection": {**base, "retained_ids_sha256": "d"},
        "A_replay": base.copy(),
    }
    summary = observer.summary()
    assert summary["comparisons"]["A_replay"]["comparable"]
    assert summary["comparisons"]["A_replay"]["retained_ids_equal"]
    assert not summary["comparisons"]["B_target_projection"]["retained_ids_equal"]
    observer.records["A_replay"]["complete"] = False
    assert observer.summary()["comparisons"]["A_replay"]["retained_ids_equal"] is None
