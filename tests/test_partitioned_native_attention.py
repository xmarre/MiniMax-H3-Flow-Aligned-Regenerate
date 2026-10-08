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
    original = lambda args: seen.append(args) or {"img": args["img"]}
    args = {"img": object(), "attention": producer}
    extra = {"original_block": original, "keep": object()}
    adapted = partitioned_block_extra(extra)
    assert adapted["original_block"](args) == {"img": args["img"]}
    assert "attention" not in seen[0]
    assert args["attention"] is producer and extra["original_block"] is original
    assert adapted["keep"] is extra["keep"]


def test_other_block_attention_replacements_are_preserved():
    attention = lambda *_a, **_kw: None
    args = {"attention": attention}
    assert partitioned_block_extra({"original_block": lambda value: value})["original_block"](args) is args


def test_real_core_domain_streams_run_without_importing_sol(monkeypatch):
    import builtins

    import test_partitioned_target_band_domain as fixture

    from h3_flow_regenerate.partitioned_transformer import partitioned_diffusion_wrapper

    case = fixture._Case()
    original_import = builtins.__import__

    def import_without_sol(name, *args, **kwargs):
        if name == "sol_h3" or name.startswith("sol_h3."):
            raise AssertionError(f"native attention imported {name}")
        return original_import(name, *args, **kwargs)

    def native_forward(dm, wrappers, video, audio, context, options, mask, audio_mask, payload=None):
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

    monkeypatch.setattr(builtins, "__import__", import_without_sol)
    monkeypatch.setattr(fixture, "_forward", native_forward)
    result = case.domain_call()
    expected_target = case.target_reference(case.video)
    expected_source = case.source_reference(case.video)
    torch.testing.assert_close(result[0][:, :, : case.geometry.head_t], expected_target[0])
    torch.testing.assert_close(
        fixture.target_band_tail(result[0], case.geometry), expected_source[0][:, :, case.geometry.head_t :]
    )
    torch.testing.assert_close(result[1], expected_source[1])
    assert partitioned_diffusion_wrapper is not None
