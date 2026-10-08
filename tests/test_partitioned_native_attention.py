"""Native partitioned attention preserves weighted domains without Sol runtime."""

import pytest
import torch

from h3_flow_regenerate.metrics import H3FlowMetrics
from h3_flow_regenerate.partitioned_attention import (
    call_partitioned_attention,
    native_partitioned_attention,
    partitioned_block_extra,
)


@pytest.fixture(autouse=True)
def cpu_core():
    cli = pytest.importorskip("comfy.cli_args")
    cli.args.cpu = True


@pytest.mark.parametrize("q_rows", [3, 9])
@pytest.mark.parametrize("measure_range", [None, (2, 5), (0, 5)])
def test_selected_backend_matches_explicit_weighted_softmax(q_rows, measure_range):
    generator = torch.Generator().manual_seed(91)
    q = torch.randn(q_rows, 2, 8, generator=generator)
    k = torch.randn(9, 2, 8, generator=generator)
    v = torch.randn(9, 2, 8, generator=generator)
    measure = -0.7 if measure_range is not None else 0.0
    seen = []

    def selected(original, qh, kh, vh, heads, **kwargs):
        seen.append(kwargs)
        return original(qh, kh, vh, heads, **kwargs)

    sentinel = object()
    options = {"optimized_attention_override": sentinel, "keep": sentinel}
    result = native_partitioned_attention(
        q,
        k,
        v,
        options=options,
        terminal=selected,
        scale=8**-0.5,
        prefix_k_range=measure_range,
        prefix_log_key_measure=measure,
    )
    logits = torch.einsum("qhd,khd->hqk", q, k) * 8**-0.5
    if measure_range is not None:
        logits[..., measure_range[0] : measure_range[1]] += measure
    expected = torch.einsum("hqk,khd->qhd", logits.softmax(-1), v)
    torch.testing.assert_close(result, expected)
    assert len(seen) == 1
    assert "optimized_attention_override" not in seen[0]["transformer_options"]
    assert options == {"optimized_attention_override": sentinel, "keep": sentinel}
    if measure_range is not None or q_rows != 9:
        assert seen[0]["mask"].shape == (1, 1, 1, 9)
    else:
        assert seen[0]["mask"] is None


@pytest.mark.parametrize("control", ["force_dense", "query_position_map"])
def test_exact_or_mapped_square_queries_request_native_masked_route(control):
    q = torch.randn(4, 1, 8)
    masks = []

    def selected(original, *args, **kwargs):
        masks.append(kwargs["mask"])
        return original(*args, **kwargs)

    native_partitioned_attention(
        q,
        q,
        q,
        options={},
        terminal=selected,
        scale=8**-0.5,
        prefix_k_range=None,
        prefix_log_key_measure=0.0,
        **{control: True if control == "force_dense" else {"requested_q_rows": 4}},
    )
    assert masks[0] is not None and torch.count_nonzero(masks[0]) == 0


def test_sol_selected_retains_request_dispatch(monkeypatch):
    request = pytest.importorskip("sol_h3.partitioned_request")
    sentinel = object()
    seen = []
    monkeypatch.setattr(request, "partitioned_request_attention", lambda *a, **kw: seen.append(kw) or sentinel)
    metrics = H3FlowMetrics()
    options = {"sol_h3_runtime_v1": {"backend": "sol"}}
    assert (
        call_partitioned_attention(None, None, None, terminal=None, metrics=metrics, transformer_options=options)
        is sentinel
    )
    assert seen == [{"transformer_options": options}]
    assert metrics.counters["partitioned_sol_kernel_calls"] == 1


def test_core_bsa_producer_cannot_bypass_partitioned_override():
    def producer(*_args, **_kwargs):
        raise AssertionError("the chunked producer must not bypass partition key weighting")

    producer.__module__ = "comfy_extras.nodes_sparse_attention"
    producer.__qualname__ = "make_h3_block_patch.<locals>.attention"
    seen = []

    def original(args):
        seen.append(args)
        return {"img": args["img"]}

    args = {"img": object(), "attention": producer}
    extra = {"original_block": original, "keep": object()}
    adapted = partitioned_block_extra(extra)
    assert adapted["original_block"](args) == {"img": args["img"]}
    assert "attention" not in seen[0]
    assert args["attention"] is producer and extra["original_block"] is original
    assert adapted["keep"] is extra["keep"]


def test_other_block_attention_replacements_are_preserved():
    def attention(*_a, **_kw):
        return None

    args = {"attention": attention}
    assert partitioned_block_extra({"original_block": lambda value: value})["original_block"](args) is args


def test_real_core_bsa_block_replacement_reaches_weighted_native_attention(monkeypatch):
    from types import SimpleNamespace

    import test_partitioned_target_band_domain as fixture

    bsa = pytest.importorskip("comfy_extras.nodes_sparse_attention")
    case = fixture._Case()
    monkeypatch.setattr(fixture, "_forward", _native_forward)
    monkeypatch.setattr(bsa, "h3_eligible", lambda *_a, **_kw: True)

    def producer(*_a, **_kw):
        raise AssertionError("Core producer bypassed partitioned weighting")

    monkeypatch.setattr(bsa, "h3_sparse_attention", producer)
    patch = SimpleNamespace(vsa=False, dense_reason=lambda *_a: "CPU oracle", log_once=lambda *_a: None)
    replacements = {
        ("double_block", i): bsa.make_h3_block_patch(block, i, patch) for i, block in enumerate(case.dm.blocks)
    }
    options = {
        "patches_replace": {"dit": replacements},
        "optimized_attention_override": bsa.make_attention_override(patch, None),
    }
    baseline = case.domain_call()
    composed = case.domain_call(extra_options=options)
    for actual, expected in zip(composed, baseline, strict=True):
        torch.testing.assert_close(actual, expected)


def _native_forward(dm, wrappers, video, audio, context, options, mask, audio_mask, payload=None):
    import test_partitioned_target_band_domain as fixture
    from comfy.patcher_extension import WrapperExecutor

    with torch.no_grad():
        return WrapperExecutor.new_class_executor(dm._forward, dm, wrappers).execute(
            [video, audio],
            torch.tensor([fixture.TIMESTEP]),
            context,
            transformer_options=options,
            denoise_mask=mask,
            audio_denoise_mask=audio_mask,
            minimax_payload=payload,
        )


def _without_sol(monkeypatch):
    import builtins

    import test_partitioned_target_band_domain as fixture

    original_import = builtins.__import__

    def import_without_sol(name, *args, **kwargs):
        if name == "sol_h3" or name.startswith("sol_h3."):
            raise AssertionError(f"native attention imported {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_sol)
    monkeypatch.setattr(fixture, "_forward", _native_forward)
    monkeypatch.setattr(fixture, "_sol", lambda _monkeypatch: None)
    return fixture


def test_real_core_domain_streams_run_without_importing_sol(monkeypatch):
    fixture = _without_sol(monkeypatch)
    case = fixture._Case()
    result = case.domain_call()
    expected_target = case.target_reference(case.video)
    expected_source = case.source_reference(case.video)
    torch.testing.assert_close(result[0][:, :, : case.geometry.head_t], expected_target[0])
    torch.testing.assert_close(
        fixture.target_band_tail(result[0], case.geometry), expected_source[0][:, :, case.geometry.head_t :]
    )
    torch.testing.assert_close(result[1], expected_source[1])


@pytest.mark.parametrize("retain", [False, True])
@pytest.mark.parametrize("anchors", ["none", "rows"])
@pytest.mark.parametrize("same_grid", [False, True])
def test_real_vdn_domain_streams_without_sol_keep_learned_branch_and_isolation(monkeypatch, retain, anchors, same_grid):
    fixture = _without_sol(monkeypatch)
    fixture.test_real_vdn_readout_keeps_uniform_stream_outputs_and_isolation(monkeypatch, retain, anchors, same_grid)


@pytest.mark.parametrize("core_outer", [False, True])
def test_vdn_preprocess_runs_once_on_full_domain_before_gather_without_sol(monkeypatch, core_outer):
    from types import SimpleNamespace

    fixture = _without_sol(monkeypatch)
    bsa = pytest.importorskip("comfy_extras.nodes_sparse_attention")
    seen = []

    def preprocess(q, k, v, heads, **kwargs):
        options = kwargs["transformer_options"]
        assert q.shape[2] == options["minimax_h3_layout"].seq_len
        assert q.shape[1] == heads
        seen.append(int(q.shape[2]))
        return q + 0.02, k, v

    def transform(original, q, k, v, heads, **kwargs):
        q, k, v = preprocess(q, k, v, heads, **kwargs)
        return original(q, k, v, heads, **kwargs)

    transform.attention_preprocess_v1 = (preprocess, None)
    terminal = transform
    if core_outer:
        patch = SimpleNamespace(vsa=False, dense_reason=lambda *_a: "CPU oracle", log_once=lambda *_a: None)
        terminal = bsa.make_attention_override(patch, transform)

    def forward(dm, wrappers, video, audio, context, options, mask, audio_mask, payload=None):
        return _native_forward(
            dm,
            wrappers,
            video,
            audio,
            context,
            {**options, "optimized_attention_override": terminal},
            mask,
            audio_mask,
            payload,
        )

    monkeypatch.setattr(fixture, "_forward", forward)
    fixture.test_real_vdn_readout_keeps_uniform_stream_outputs_and_isolation(
        monkeypatch, retain=True, anchors="rows", same_grid=False
    )
    assert seen


@pytest.mark.parametrize("context", ["mixed_grid", "domain_uniform_v1"])
def test_continuation_scheduler_completes_low_probe_high_without_sol_backend(monkeypatch, context):
    import test_partitioned_target_band_e2e as fixture

    from h3_flow_regenerate.partitioned_diagnostics import (
        PARTITIONED_TARGET_BAND_CONTEXT_KEY,
        PARTITIONED_TARGET_BAND_TOKENS_KEY,
    )
    from h3_flow_regenerate.partitioned_stage import PARTITIONED_STAGE_KEY

    owner = fixture._vdn_owner

    def native_owner():
        patched = owner()
        patched._vdn_partitioned_attention_provider_api = 1
        return patched

    monkeypatch.setattr(fixture, "_vdn_owner", native_owner)
    if context != "mixed_grid":
        from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE

        frames = sum(int(FRAME_PER_TOKEN[i % len(FRAME_PER_TOKEN)]) for i in range(fixture.TEMPORAL))
        monkeypatch.setattr(fixture, "AUDIO_T", round(frames * float(FRAME_RESCALE)))
    run = fixture._harness(
        monkeypatch,
        spatial_stage_control=fixture.PARTITIONED_SPATIAL_STAGE_TARGET_BAND,
        extra_transformer_options={
            fixture.SOL_RUNTIME_KEY: None,
            PARTITIONED_TARGET_BAND_CONTEXT_KEY: context,
            PARTITIONED_TARGET_BAND_TOKENS_KEY: 4,
        },
    )
    assert run.metrics.counters["partitioned_native_attention_calls"] > 0
    assert run.metrics.counters.get("partitioned_sol_kernel_calls", 0) == 0
    assert {call["stage"] for call in run.calls} == {"low", "probe", "high"}
    assert torch.isfinite(run.result).all()
    assert PARTITIONED_STAGE_KEY not in run.guider.model_options["transformer_options"]


def test_node_installation_does_not_require_sol_package(monkeypatch):
    from types import SimpleNamespace

    from test_partitioned_target_band_e2e import _Upscaler
    from vdn_h3 import partitioned_runtime

    from h3_flow_regenerate import partitioned_node

    _without_sol(monkeypatch)
    model = SimpleNamespace(model_options={"transformer_options": {}}, remove_wrappers_with_key=lambda *_a: None)
    metrics = H3FlowMetrics()
    installed = []
    monkeypatch.setattr(partitioned_node, "patch_flow_model", lambda *a, **kw: (model, metrics))
    monkeypatch.setattr(partitioned_node, "_put_wrapper_first", lambda *_a: None)
    monkeypatch.setattr(partitioned_runtime, "install_partitioned_external_sequence_bridge", installed.append)
    result, returned_metrics = partitioned_node.H3PartitionedExactPrefixHandoff().patch(
        model=model,
        trajectory=object(),
        source_mode="scale",
        source_scale=0.7,
        source_width=864,
        source_height=640,
        handoff_coordinate=0.35,
        handoff_selection="fixed",
        guidance_mode="off",
        direction_weight=0.25,
        acceleration_weight=0.25,
        consistency_weight=0.25,
        low_frequency_cutoff=0.25,
        learned_upscaler=_Upscaler(),
    )
    assert result is model and installed == [model]
    receipt = next(e.fields for e in returned_metrics.events if e.kind == "partitioned_exact_prefix_installed")
    assert receipt["sol_abi"] is None


def test_native_preflight_rejects_stale_sol_only_vdn_owner():
    from types import SimpleNamespace

    import test_partitioned_target_band_e2e as fixture

    from h3_flow_regenerate.partitioned_scheduler import (
        PartitionedPreflightUnsupported,
        _validate_partitioned_vdn_compat,
    )

    patcher = SimpleNamespace(object_patches={"diffusion_model.blocks.0.attn.forward": fixture._vdn_owner()})
    with pytest.raises(PartitionedPreflightUnsupported, match="VDN partitioned attention provider API 1"):
        _validate_partitioned_vdn_compat(patcher, required_attention_provider=True)
