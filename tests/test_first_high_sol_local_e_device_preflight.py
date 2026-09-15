from __future__ import annotations

import importlib

import pytest


root = importlib.import_module("__init__")
torch = root._FIRST_HIGH_SOL_LOCAL_MODULE.torch


def _install_cuda_stubs(monkeypatch, *, count: int, capability: tuple[int, int], current: int = 0):
    cuda = torch.cuda
    monkeypatch.setattr(cuda, "is_available", lambda: True)
    monkeypatch.setattr(cuda, "device_count", lambda: count)
    monkeypatch.setattr(cuda, "current_device", lambda: current)
    monkeypatch.setattr(cuda, "get_device_capability", lambda _device: capability)


def test_e_preflight_uses_single_runtime_cuda_when_replay_tensor_is_cpu(monkeypatch):
    _install_cuda_stubs(monkeypatch, count=1, capability=(12, 0))
    seen = []

    def fake_preflight(device):
        seen.append(str(device))
        return {"ok": True}

    monkeypatch.setattr(root, "_ORIGINAL_E_MEMORY_PREFLIGHT", fake_preflight)
    result = root._memory_preflight_on_runtime_cuda(torch.device("cpu"))

    assert seen == ["cuda:0"]
    assert result["ok"] is True
    assert result["replay_tensor_device"] == "cpu"
    assert result["cuda_preflight_device"] == "cuda:0"
    assert result["cuda_compute_capability"] == [12, 0]


def test_e_preflight_keeps_explicit_cuda_device(monkeypatch):
    _install_cuda_stubs(monkeypatch, count=2, capability=(12, 0), current=0)
    seen = []
    monkeypatch.setattr(root, "_ORIGINAL_E_MEMORY_PREFLIGHT", lambda device: seen.append(str(device)) or {})

    result = root._memory_preflight_on_runtime_cuda(torch.device("cuda:1"))

    assert seen == ["cuda:1"]
    assert result["cuda_preflight_device"] == "cuda:1"


def test_e_preflight_rejects_ambiguous_multigpu_cpu_replay(monkeypatch):
    _install_cuda_stubs(monkeypatch, count=2, capability=(12, 0))
    monkeypatch.setattr(root, "_ORIGINAL_E_MEMORY_PREFLIGHT", lambda _device: pytest.fail("must not run"))

    with pytest.raises(RuntimeError, match="runtime CUDA device is ambiguous"):
        root._memory_preflight_on_runtime_cuda(torch.device("cpu"))


def test_e_preflight_rejects_non_sm120_runtime_device(monkeypatch):
    _install_cuda_stubs(monkeypatch, count=1, capability=(8, 9))
    monkeypatch.setattr(root, "_ORIGINAL_E_MEMORY_PREFLIGHT", lambda _device: pytest.fail("must not run"))

    with pytest.raises(RuntimeError, match=r"requires CUDA SM120.*SM89"):
        root._memory_preflight_on_runtime_cuda(torch.device("cpu"))
