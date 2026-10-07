import json

import pytest
import torch

from h3_flow_regenerate.boundary_witness import (
    WITNESS_DIRECTORY_OPTION,
    BoundaryWitness,
    configured_boundary_witness,
)
from h3_flow_regenerate.metrics import H3FlowMetrics


def test_witness_default_off_and_stage_claim_is_single_use(monkeypatch, tmp_path):
    monkeypatch.delenv("H3_FLOW_BOUNDARY_WITNESS_DIR", raising=False)
    metrics = H3FlowMetrics()
    assert configured_boundary_witness(metrics) is None
    assert configured_boundary_witness(metrics, directory="") is None
    explicit = configured_boundary_witness(metrics, directory=str(tmp_path))
    assert explicit is not None
    assert explicit.directory == tmp_path.resolve()
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


def test_requested_witness_missing_capability_fails_before_sampling(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from h3_flow_regenerate.partitioned_scheduler import _validate_partitioned_vdn_compat

    monkeypatch.setenv("H3_FLOW_BOUNDARY_WITNESS_DIR", str(tmp_path / "stale-env"))
    owner = SimpleNamespace(
        _vdn_forward=True,
        _vdn_external_sequence_api=4,
        _vdn_partitioned_boundary_query_api=1,
        _vdn_partitioned_boundary_query_policy="boundary_suffix_local_group_dense_v1",
    )
    patcher = SimpleNamespace(
        object_patches={"diffusion_model.blocks.0.attn.forward": owner},
        model_options={WITNESS_DIRECTORY_OPTION: ""},
    )
    # An explicit node OFF must override a stale process environment.
    _validate_partitioned_vdn_compat(patcher)
    patcher.model_options[WITNESS_DIRECTORY_OPTION] = str(tmp_path)
    with pytest.raises(RuntimeError, match="before sampling"):
        _validate_partitioned_vdn_compat(patcher)
    owner._vdn_partitioned_boundary_witness_api = 1
    _validate_partitioned_vdn_compat(patcher)


def test_requested_low_witness_must_complete_and_owner_is_removed(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from h3_flow_regenerate.partitioned_scheduler import _partitioned_stage_contract
    from h3_flow_regenerate.partitioned_stage import PARTITIONED_STAGE_KEY, PartitionedStagePlan

    monkeypatch.delenv("H3_FLOW_BOUNDARY_WITNESS_DIR", raising=False)
    prefix = torch.zeros(1, 24, 2, 4, 4)
    plan = PartitionedStagePlan(prefix=prefix, temporal=4, source_h=2, source_w=2, prefix_noise=prefix.clone())
    guider = SimpleNamespace(
        model_options={
            WITNESS_DIRECTORY_OPTION: str(tmp_path),
            "transformer_options": {"h3_flow_stage": "low"},
        }
    )
    with (
        pytest.raises(RuntimeError, match="not completed"),
        _partitioned_stage_contract(guider, plan, H3FlowMetrics()),
    ):
        owner = guider.model_options["transformer_options"][PARTITIONED_STAGE_KEY]
        assert owner.boundary_witness is not None
    assert PARTITIONED_STAGE_KEY not in guider.model_options["transformer_options"]


def test_observation_selection_survives_options_copy_and_has_stage_lifetime(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from h3_flow_regenerate.partitioned_scheduler import _partitioned_stage_contract
    from h3_flow_regenerate.partitioned_stage import PARTITIONED_STAGE_KEY, PartitionedStagePlan

    monkeypatch.delenv("H3_FLOW_BOUNDARY_WITNESS_DIR", raising=False)
    prefix = torch.zeros(1, 24, 2, 4, 4)
    plan = PartitionedStagePlan(prefix=prefix, temporal=4, source_h=2, source_w=2, prefix_noise=prefix.clone())
    guider = SimpleNamespace(
        model_options={
            WITNESS_DIRECTORY_OPTION: str(tmp_path),
            "transformer_options": {"h3_flow_stage": "probe"},
        }
    )
    with _partitioned_stage_contract(guider, plan, H3FlowMetrics()):
        owner = guider.model_options["transformer_options"][PARTITIONED_STAGE_KEY]
        copied = guider.model_options["transformer_options"].copy()
        assert copied[PARTITIONED_STAGE_KEY].boundary_witness is owner.boundary_witness
        guider.model_options[WITNESS_DIRECTORY_OPTION] = ""
        assert owner.boundary_witness is not None
    with _partitioned_stage_contract(guider, plan, H3FlowMetrics()):
        assert guider.model_options["transformer_options"][PARTITIONED_STAGE_KEY].boundary_witness is None


def test_target_band_capture_does_not_install_feature_sink_and_cleans_up_on_failure(tmp_path):
    from types import SimpleNamespace

    from h3_flow_regenerate.partitioned_scheduler import _partitioned_stage_contract
    from h3_flow_regenerate.partitioned_stage import (
        PARTITIONED_STAGE_KEY,
        PartitionedStagePlan,
        PartitionedTargetBandGeometry,
    )

    prefix = torch.zeros(1, 24, 2, 4, 4)
    plan = PartitionedStagePlan(prefix=prefix, temporal=7, source_h=2, source_w=2, prefix_noise=prefix.clone())
    band = PartitionedTargetBandGeometry(
        protected_t=2, band_t=2, temporal=7, source_h=2, source_w=2, target_h=4, target_w=4
    )
    guider = SimpleNamespace(
        model_options={
            WITNESS_DIRECTORY_OPTION: str(tmp_path),
            "transformer_options": {"h3_flow_stage": "low"},
        }
    )
    with (
        pytest.raises(RuntimeError, match="sampler failure"),
        _partitioned_stage_contract(guider, plan, H3FlowMetrics(), target_band=band),
    ):
        owner = guider.model_options["transformer_options"][PARTITIONED_STAGE_KEY]
        assert owner.boundary_witness is None
        assert owner.target_band is band
        raise RuntimeError("sampler failure")
    assert PARTITIONED_STAGE_KEY not in guider.model_options["transformer_options"]
    assert guider.model_options[WITNESS_DIRECTORY_OPTION] == str(tmp_path)
    assert not list(tmp_path.iterdir())
