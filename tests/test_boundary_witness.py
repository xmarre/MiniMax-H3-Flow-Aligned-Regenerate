import json

import pytest
import torch

from h3_flow_regenerate.boundary_witness import BoundaryWitness, configured_boundary_witness
from h3_flow_regenerate.metrics import H3FlowMetrics


def test_witness_default_off_and_stage_claim_is_single_use(monkeypatch, tmp_path):
    monkeypatch.delenv("H3_FLOW_BOUNDARY_WITNESS_DIR", raising=False)
    metrics = H3FlowMetrics()
    assert configured_boundary_witness(metrics) is None
    sink = BoundaryWitness(tmp_path, metrics)
    assert not sink.claim({"stage": "probe"})
    assert sink.claim({"stage": "low"})
    assert not sink.claim({"stage": "low"})


def test_witness_owns_cpu_copy_and_persists_without_mutating_input(tmp_path):
    metrics = H3FlowMetrics()
    sink = BoundaryWitness(tmp_path, metrics)
    assert sink.claim({"stage": "low"})
    original = torch.arange(12).reshape(4, 3)
    expected = original.clone()
    sink.add("raw", original)
    original.zero_()  # Models retained scratch reuse after the synchronous copy.
    sink.finish({"stage": "low"})
    receipt_path = next(tmp_path.glob("*.json"))
    receipt = json.loads(receipt_path.read_text())
    saved = torch.load(tmp_path / receipt["tensor_file"], weights_only=True)
    assert torch.equal(saved["raw"], expected)
    assert saved["raw"].device.type == "cpu"
    assert receipt["extra_h3_nfe"] == receipt["extra_provider_calls"] == receipt["extra_vae_calls"] == 0
    assert not receipt["output_mutated"]
    assert not sink._tensors
    assert metrics.events[-1].kind == "partitioned_boundary_witness"


def test_witness_budget_rejects_before_copy(tmp_path):
    sink = BoundaryWitness(tmp_path, H3FlowMetrics(), max_bytes=1)
    with pytest.raises(RuntimeError, match="byte budget"):
        sink.add("raw", torch.ones(2))
    assert not sink._tensors
