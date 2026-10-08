"""Spectrum history identity for non-Sol partitioned continuation stages."""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate.metrics import H3FlowMetrics
from h3_flow_regenerate.partitioned_attention import (
    ATTENTION_BACKEND_HISTORY_KEY,
    ATTENTION_BACKEND_RECEIPTS_KEY,
    PARTITIONED_NATIVE_HISTORY_NAME,
    SOL_RUNTIME_KEY,
    PartitionedNativeHistoryPolicy,
    call_partitioned_attention,
)
from h3_flow_regenerate.partitioned_scheduler import _partitioned_stage_contract
from h3_flow_regenerate.partitioned_stage import PARTITIONED_STAGE_KEY, PartitionedStagePlan
from h3_flow_regenerate.runtime import _flow_stage_contract


def _plan():
    prefix = torch.zeros(1, 24, 2, 4, 6)
    return PartitionedStagePlan(prefix, 5, 2, 4, prefix.clone())


def _layout():
    return SimpleNamespace(signature=("layout", 49), seq_len=49, segments=(("text", 7), ("video", 42)))


def test_non_sol_stage_publishes_and_restores_flow_history_provider():
    other = object()
    guider = SimpleNamespace(model_options={"transformer_options": {ATTENTION_BACKEND_HISTORY_KEY: {"other": other}}})
    options = guider.model_options["transformer_options"]
    with _flow_stage_contract(guider, "low"), _partitioned_stage_contract(guider, _plan(), H3FlowMetrics()):
        history = options[ATTENTION_BACKEND_HISTORY_KEY]
        policy = history[PARTITIONED_NATIVE_HISTORY_NAME]
        assert history["other"] is other
        assert isinstance(policy, PartitionedNativeHistoryPolicy)
        first = policy(layout=_layout(), options=options, model=SimpleNamespace(dtype=torch.bfloat16))
        assert first is not None
        # Nested option dictionaries are copied between model calls; the identity
        # depends only on the stage owner leaf, so copies keep the same identity.
        copied = {**options, ATTENTION_BACKEND_HISTORY_KEY: dict(history)}
        assert policy(layout=_layout(), options=copied, model=SimpleNamespace(dtype=torch.bfloat16)) == first
        assert policy(layout=_layout(), options={}, model=SimpleNamespace(dtype=torch.bfloat16)) is None
    assert options == {ATTENTION_BACKEND_HISTORY_KEY: {"other": other}}

    bare = SimpleNamespace(model_options={"transformer_options": {}})
    with _flow_stage_contract(bare, "low"), _partitioned_stage_contract(bare, _plan(), H3FlowMetrics()):
        bare_options = bare.model_options["transformer_options"]
        assert PARTITIONED_NATIVE_HISTORY_NAME in bare_options[ATTENTION_BACKEND_HISTORY_KEY]
    assert ATTENTION_BACKEND_HISTORY_KEY not in bare.model_options["transformer_options"]


def test_sol_stage_keeps_sol_history_only():
    sol = {"backend": "sol"}
    guider = SimpleNamespace(model_options={"transformer_options": {SOL_RUNTIME_KEY: sol}})
    options = guider.model_options["transformer_options"]
    with _flow_stage_contract(guider, "low"), _partitioned_stage_contract(guider, _plan(), H3FlowMetrics()):
        assert ATTENTION_BACKEND_HISTORY_KEY not in options
    assert options == {SOL_RUNTIME_KEY: sol}


def test_native_partitioned_attention_reports_one_stage_receipt_per_call():
    cli = pytest.importorskip("comfy.cli_args")
    cli.args.cpu = True
    guider = SimpleNamespace(model_options={"transformer_options": {}})
    options = guider.model_options["transformer_options"]
    generator = torch.Generator().manual_seed(5)
    q, k, v = (torch.randn(4, 2, 8, generator=generator) for _ in range(3))
    with _flow_stage_contract(guider, "low"), _partitioned_stage_contract(guider, _plan(), H3FlowMetrics()):
        policy = options[ATTENTION_BACKEND_HISTORY_KEY][PARTITIONED_NATIVE_HISTORY_NAME]
        receipts = []
        call_options = {**options, ATTENTION_BACKEND_RECEIPTS_KEY: receipts}
        for _ in range(3):
            call_partitioned_attention(
                q,
                k,
                v,
                terminal=None,
                metrics=H3FlowMetrics(),
                transformer_options=call_options,
                scale=8**-0.5,
                prefix_k_range=None,
                prefix_log_key_measure=0.0,
            )
        assert receipts == [policy.receipt()]
        assert policy.accept_receipts(receipts)
        assert not policy.accept_receipts([])
        assert not policy.accept_receipts([*receipts, ("other_route",)])

        # A receipt from a different stage owner is never accepted.
        foreign = {**call_options, PARTITIONED_STAGE_KEY: object(), ATTENTION_BACKEND_RECEIPTS_KEY: []}
        call_partitioned_attention(
            q,
            k,
            v,
            terminal=None,
            metrics=H3FlowMetrics(),
            transformer_options=foreign,
            scale=8**-0.5,
            prefix_k_range=None,
            prefix_log_key_measure=0.0,
        )
        assert not policy.accept_receipts(foreign[ATTENTION_BACKEND_RECEIPTS_KEY])


@pytest.mark.parametrize("stage", ["low", "probe", "high"])
@pytest.mark.parametrize("uniform", [False, True])
def test_spectrum_uses_flow_history_instead_of_core_bsa_audit(stage, uniform):
    root = os.environ.get("SPECTRUM_PATH")
    if root and root not in sys.path:
        sys.path.insert(0, root)
    backend_history = pytest.importorskip("comfyui_spectrum_h3.backend_history")
    guider = SimpleNamespace(model_options={"transformer_options": {}})
    options = guider.model_options["transformer_options"]
    plan = _plan()
    if uniform:
        plan = PartitionedStagePlan(plan.prefix, plan.temporal, *plan.target_hw, plan.prefix_noise)
    with _flow_stage_contract(guider, stage), _partitioned_stage_contract(guider, plan, H3FlowMetrics()):
        identity, safe = backend_history.preflight(options, _layout(), SimpleNamespace(dtype=torch.bfloat16))
        assert safe is True
        assert dict(identity)[PARTITIONED_NATIVE_HISTORY_NAME] is not None


def test_domain_stream_views_report_the_owning_stage_receipt():
    cli = pytest.importorskip("comfy.cli_args")
    cli.args.cpu = True
    from h3_flow_regenerate.partitioned_stage import PartitionedStageStreamView

    guider = SimpleNamespace(model_options={"transformer_options": {}})
    options = guider.model_options["transformer_options"]
    generator = torch.Generator().manual_seed(6)
    q, k, v = (torch.randn(4, 2, 8, generator=generator) for _ in range(3))
    with _flow_stage_contract(guider, "low"), _partitioned_stage_contract(guider, _plan(), H3FlowMetrics()):
        policy = options[ATTENTION_BACKEND_HISTORY_KEY][PARTITIONED_NATIVE_HISTORY_NAME]
        receipts = []
        for stream, head in (("target", 1), ("source", 2)):
            view = PartitionedStageStreamView(options[PARTITIONED_STAGE_KEY], stream=stream, attention_head_t=head)
            call_partitioned_attention(
                q,
                k,
                v,
                terminal=None,
                metrics=H3FlowMetrics(),
                transformer_options={**options, PARTITIONED_STAGE_KEY: view, ATTENTION_BACKEND_RECEIPTS_KEY: receipts},
                scale=8**-0.5,
                prefix_k_range=None,
                prefix_log_key_measure=0.0,
            )
        assert receipts == [policy.receipt()]
        assert policy.accept_receipts(receipts)
