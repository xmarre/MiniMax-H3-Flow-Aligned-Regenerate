from types import SimpleNamespace

from h3_flow_regenerate.attention import layout_summary, make_layout_block_wrapper
from h3_flow_regenerate.partitioned_prefix import PARTITIONED_PREFIX_KEY


def _layout(signature):
    return SimpleNamespace(
        signature=signature,
        segments=[(0, 3, "text"), (3, 8, "video")],
        seq_len=8,
    )


def test_layout_summary_preserves_partitioned_version_tag():
    signature = (PARTITIONED_PREFIX_KEY, 3468, 62, 46, 44, 348, 2, 66, 64)
    summary = layout_summary(_layout(signature))
    assert summary["signature"] == signature
    assert summary["text_rows"] == 3
    assert summary["video_rows"] == 5
    assert summary["sequence_rows"] == 8


def test_layout_summary_preserves_mixed_grid_version_tag():
    signature = ("h3_flow_mixed_grid_v1", 3468, 62, 46, 44, 348, 2, 66, 64)
    assert layout_summary(_layout(signature))["signature"] == signature


def test_layout_summary_keeps_native_signature_numeric():
    signature = (3468, 62, 46, 44, 348)
    assert layout_summary(_layout(signature))["signature"] == signature


def test_layout_block_wrapper_records_partitioned_signature_without_coercing_tag():
    signature = (PARTITIONED_PREFIX_KEY, 3468, 62, 46, 44, 348, 2, 66, 64)
    events = []

    class Metrics:
        def event(self, name, **payload):
            events.append((name, payload))

        def increment(self, *_args, **_kwargs):
            raise AssertionError("unexpected unavailable-layout path")

    wrapper = make_layout_block_wrapper(0, Metrics())
    args = {"layout": _layout(signature), "transformer_options": {}}
    marker = object()
    result = wrapper(args, {"original_block": lambda _args: marker})

    assert result is marker
    assert events == [("packed_layout", {**layout_summary(_layout(signature))})]
    assert "h3_flow_attention_context" not in args["transformer_options"]
