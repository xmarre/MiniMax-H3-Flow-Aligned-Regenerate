from types import SimpleNamespace

from h3_flow_regenerate.mixed_grid import _mixed_transformer_options


def test_mixed_transformer_options_publish_current_layout_without_mutating_carrier():
    carrier = SimpleNamespace(seq_len=100, signature=("carrier",))
    mixed = SimpleNamespace(seq_len=140, signature=("h3_flow_mixed_grid_v1",))
    marker = object()
    original = {
        "minimax_h3_layout": carrier,
        "optimized_attention_override": marker,
        "uuids": ("cond",),
    }

    forwarded = _mixed_transformer_options(original, mixed)

    assert forwarded is not original
    assert forwarded["minimax_h3_layout"] is mixed
    assert forwarded["optimized_attention_override"] is marker
    assert forwarded["uuids"] == ("cond",)
    assert original["minimax_h3_layout"] is carrier
