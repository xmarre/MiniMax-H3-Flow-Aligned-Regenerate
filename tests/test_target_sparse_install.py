from __future__ import annotations

from types import SimpleNamespace

from h3_flow_regenerate.comfy_compat import _contains_target_sparse_wrapper, _install_target_sparse
from h3_flow_regenerate.metrics import H3FlowMetrics


class _PatchModel:
    def __init__(self, existing):
        self.model_options = {
            "transformer_options": {
                "patches_replace": {
                    "dit": dict(existing),
                }
            }
        }

    def set_model_patch_replace(self, wrapper, namespace, block_kind, layer):
        assert namespace == "dit"
        patches = self.model_options["transformer_options"].setdefault("patches_replace", {}).setdefault("dit", {})
        patches[(block_kind, layer)] = wrapper


def _existing_wrapper(args, _extra):
    return {"img": args["img"]}


def test_target_sparse_install_composes_with_existing_block_replacements(monkeypatch):
    model = _PatchModel({("double_block", 0): _existing_wrapper})
    monkeypatch.setattr(
        "h3_flow_regenerate.comfy_compat.validate_h3_model",
        lambda _model: SimpleNamespace(blocks=[object(), object(), object()]),
    )
    metrics = H3FlowMetrics()

    _install_target_sparse(model, metrics)

    dit = model.model_options["transformer_options"]["patches_replace"]["dit"]
    assert set(dit) == {("double_block", 0), ("double_block", 1), ("double_block", 2)}
    assert _contains_target_sparse_wrapper(dit[("double_block", 0)])
    assert dit[("double_block", 0)]._h3_flow_target_sparse_previous is _existing_wrapper
    assert dit[("double_block", 1)]._h3_flow_target_sparse_previous is None
    assert dit[("double_block", 2)]._h3_flow_target_sparse_previous is None


def test_target_sparse_install_is_idempotent_through_flow_layout_wrapper(monkeypatch):
    model = _PatchModel({})
    monkeypatch.setattr(
        "h3_flow_regenerate.comfy_compat.validate_h3_model",
        lambda _model: SimpleNamespace(blocks=[object()]),
    )
    metrics = H3FlowMetrics()
    _install_target_sparse(model, metrics)
    dit = model.model_options["transformer_options"]["patches_replace"]["dit"]
    sparse = dit[("double_block", 0)]

    def layout_wrapper(args, extra):
        return sparse(args, extra)

    layout_wrapper._h3_flow_layout_wrapper = True
    layout_wrapper._h3_flow_previous = sparse
    dit[("double_block", 0)] = layout_wrapper

    _install_target_sparse(model, metrics)

    assert dit[("double_block", 0)] is layout_wrapper
    assert _contains_target_sparse_wrapper(layout_wrapper)
