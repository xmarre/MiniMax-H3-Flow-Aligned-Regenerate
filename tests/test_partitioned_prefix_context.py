from __future__ import annotations

from types import SimpleNamespace

import torch

from h3_flow_regenerate.partitioned_diagnostics import PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE
from h3_flow_regenerate.partitioned_prefix import PARTITIONED_PREFIX_KEY
from h3_flow_regenerate.partitioned_stage import (
    PARTITIONED_STAGE_KEY,
    PartitionedStagePlan,
    PartitionedStageRuntime,
)
from h3_flow_regenerate.partitioned_transformer import partitioned_diffusion_wrapper


class _Metrics:
    def __init__(self):
        self.counters = {}
        self.events = []

    def increment(self, name, value=1):
        self.counters[name] = self.counters.get(name, 0) + value

    def event(self, kind, **fields):
        self.events.append((kind, fields))


class _Executor:
    def __init__(self):
        self.class_obj = SimpleNamespace(blocks=[object()])
        self.calls = []
        self.sentinel = object()

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.sentinel


def test_source_carrier_context_delegates_unchanged_uniform_grid_without_partition_contract():
    prefix = torch.randn(1, 24, 1, 4, 4)
    plan = PartitionedStagePlan(
        prefix=prefix,
        temporal=2,
        source_h=2,
        source_w=2,
        prefix_noise=torch.randn_like(prefix),
    )
    metrics = _Metrics()
    runtime = PartitionedStageRuntime(
        plan=plan,
        metrics=metrics,
        prefix_transformer_context=PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE,
    )
    options = {PARTITIONED_STAGE_KEY: runtime, "preserve": "value"}
    video = torch.randn(1, 24, 2, 2, 2)
    audio = torch.randn(1, 32, 2, 5)
    x = (video, audio)
    timestep = torch.tensor([0.5])
    context = torch.randn(1, 3, 8)
    payload = {"keep": "payload"}
    executor = _Executor()

    result = partitioned_diffusion_wrapper(
        executor,
        x,
        timestep,
        context,
        transformer_options=options,
        minimax_payload=payload,
        extra_marker="unchanged",
    )

    assert result is executor.sentinel
    assert len(executor.calls) == 1
    args, kwargs = executor.calls[0]
    assert args[0] is x
    assert args[1] is timestep
    assert args[2] is context
    assert args[3] is options
    assert kwargs["minimax_payload"] is payload
    assert kwargs["extra_marker"] == "unchanged"
    assert PARTITIONED_PREFIX_KEY not in args[3]
    assert metrics.counters["partitioned_source_carrier_uniform_transformer_calls"] == 1
    assert metrics.counters["partitioned_source_carrier_uniform_prefix_frames"] == 1
    assert metrics.events == [
        (
            "partitioned_prefix_transformer_context",
            {
                "mode": PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE,
                "temporal": 2,
                "prefix_t": 1,
                "source_hw": (2, 2),
                "target_hw": (4, 4),
                "heterogeneous_partition_contract_published": False,
                "exact_target_prefix_injected_into_transformer": False,
                "native_source_carrier_preserved": True,
                "external_exact_prefix_owner_unchanged": True,
                "diagnostic_only": True,
            },
        )
    ]
