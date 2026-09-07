from types import SimpleNamespace

from h3_flow_regenerate.attention import make_layout_block_wrapper
from h3_flow_regenerate.metrics import H3FlowMetrics


def _layout():
    return SimpleNamespace(
        segments=((0, 2, "text"), (2, 4, "audio"), (4, 8, "video")),
        signature=(2, 1, 4, 4, 2),
        seq_len=8,
    )


def test_layout_wrapper_is_output_neutral_when_old_comfy_block_args_have_no_layout():
    metrics = H3FlowMetrics()
    transformer = {}
    seen = []

    def original_block(args):
        seen.append(args)
        return {"img": "native-result"}

    wrapper = make_layout_block_wrapper(0, metrics)
    args = {"transformer_options": transformer, "img": "native-input"}

    assert wrapper(args, {"original_block": original_block}) == {"img": "native-result"}
    assert seen == [args]
    assert transformer == {}
    assert metrics.counters == {"packed_layout_unavailable_calls": 1}
    assert metrics.events == ()


def test_layout_wrapper_uses_modern_transformer_options_layout_fallback():
    metrics = H3FlowMetrics()
    layout = _layout()
    transformer = {"minimax_h3_layout": layout}

    def original_block(args):
        context = transformer["h3_flow_attention_context"]
        assert context["layout"] is layout
        assert context["layer"] == 0
        return {"img": args["img"]}

    wrapper = make_layout_block_wrapper(0, metrics)
    args = {"transformer_options": transformer, "img": "native-input"}

    assert wrapper(args, {"original_block": original_block}) == {"img": "native-input"}
    assert "h3_flow_attention_context" not in transformer
    assert metrics.counters == {}
    assert [event.kind for event in metrics.events] == ["packed_layout"]
    assert metrics.events[0].fields["sequence_rows"] == layout.seq_len


def test_layout_wrapper_prefers_direct_layout_and_restores_existing_context():
    metrics = H3FlowMetrics()
    direct_layout = _layout()
    fallback_layout = SimpleNamespace(segments=(), signature=(0, 0, 0, 0, 0), seq_len=0)
    old_context = {"layout": "outer-layout", "layer": 99}
    transformer = {
        "minimax_h3_layout": fallback_layout,
        "h3_flow_attention_context": old_context,
    }

    def previous(args, extra):
        context = transformer["h3_flow_attention_context"]
        assert context["layout"] is direct_layout
        assert context["layer"] == 3
        return extra["original_block"](args)

    wrapper = make_layout_block_wrapper(3, metrics, previous=previous, record_layout=False)
    args = {
        "layout": direct_layout,
        "transformer_options": transformer,
        "img": "native-input",
    }

    result = wrapper(args, {"original_block": lambda inner: {"img": inner["img"]}})
    assert result == {"img": "native-input"}
    assert transformer["h3_flow_attention_context"] is old_context
    assert metrics.counters == {}
    assert metrics.events == ()
