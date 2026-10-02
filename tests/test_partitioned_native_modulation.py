"""Exercise the actual Core block loop and Flow's per-forward label lifetime."""

import pytest
import torch

from h3_flow_regenerate import partitioned_transformer as transform
from h3_flow_regenerate.metrics import H3FlowMetrics
from h3_flow_regenerate.partitioned_stage import PARTITIONED_STAGE_KEY, PartitionedStagePlan, PartitionedStageRuntime


@pytest.mark.parametrize("source_hw", [(4, 4), (8, 8)])
def test_native_modulation_validates_once_and_isolates_replacement_writes(monkeypatch, source_hw):
    cli = pytest.importorskip("comfy.cli_args")
    cli.args.cpu = True
    import comfy.ops
    from comfy.ldm.minimax.model import MiniMaxH3Model
    from comfy.patcher_extension import WrapperExecutor

    dm = MiniMaxH3Model(
        hidden_size=8,
        num_layers=50,
        token_refiner_num_layers=0,
        num_attention_heads=1,
        attention_head_dim=128,
        ffn_hidden_size=16,
        text_dim=8,
        time_embed_dim=2,
        adaln_curve_grid=4,
        rope_inv_freq_len=2,
        dtype=torch.float32,
        device="cpu",
        operations=comfy.ops.disable_weight_init,
    )
    with torch.no_grad():
        for param in dm.parameters():
            param.fill_(0.02)
        dm.adaln_t_table.fill_(0.1)
        dm.rope.inv_freq.fill_(0.2)
    for block in dm.blocks:
        block.forward = lambda img, *_args, **_kw: img.clone()

    prefix = torch.randn(1, 24, 2, 8, 8)
    plan = PartitionedStagePlan(
        prefix=prefix,
        prefix_noise=torch.randn_like(prefix),
        temporal=5,
        source_h=source_hw[0],
        source_w=source_hw[1],
    )
    metrics = H3FlowMetrics()
    runtime = PartitionedStageRuntime(plan=plan, metrics=metrics, audio_position_domain="source_carrier")
    labels = []

    def previous(args, extra):
        row = args["mod_segments"][-1][2]
        labels.append(row.clone())
        result = extra["original_block"](args)
        # A downstream replacement can write its forwarded index tensor without
        # corrupting either native Core labels or another block's expansion.
        row.fill_(777)
        return result

    options = {
        PARTITIONED_STAGE_KEY: runtime,
        "patches_replace": {"dit": {("double_block", i): previous for i in range(50)}},
    }
    video = torch.randn(1, 24, 5, *source_hw)
    audio = torch.randn(1, 32, 2, 5)
    mask = torch.ones(1, 1, 5, *source_hw)
    mask[:, :, :2] = 0
    context = torch.randn(1, 3, 8)
    calls = []
    original = transform.partitioned_mod_segments

    def observed(*args):
        calls.append(args[0])
        return original(*args)

    monkeypatch.setattr(transform, "partitioned_mod_segments", observed)
    with torch.no_grad():
        first = WrapperExecutor.new_class_executor(
            dm._forward,
            dm,
            [transform.partitioned_diffusion_wrapper],
        ).execute(
            [video, audio],
            torch.tensor([750.0]),
            context,
            transformer_options=options,
            denoise_mask=mask,
        )
    assert len(calls) == 1
    assert len(labels) == 50
    assert all(torch.equal(row, labels[0]) for row in labels)
    assert labels[0].numel() == plan.partitioned_rows
    assert labels[0][: plan.prefix_rows].unique().numel() == 1
    assert labels[0][plan.prefix_rows :].unique().numel() == 1
    assert labels[0][0] != labels[0][-1]
    assert torch.count_nonzero(first[0][:, :, :2]) == 0
    assert metrics.counters["partitioned_modulation_validations"] == 1
    labels.clear()
    with torch.no_grad():
        second = WrapperExecutor.new_class_executor(
            dm._forward,
            dm,
            [transform.partitioned_diffusion_wrapper],
        ).execute(
            [video, audio],
            torch.tensor([750.0]),
            context,
            transformer_options=options,
            denoise_mask=mask,
        )
    assert len(calls) == 2
    assert metrics.counters["partitioned_modulation_validations"] == 2
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
