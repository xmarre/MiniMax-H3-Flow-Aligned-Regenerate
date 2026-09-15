from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

import __init__ as plugin_root
from h3_flow_regenerate import first_high_operator_comparison as w


def _deny_bare_import(name: str):
    raise ModuleNotFoundError(name)


def test_w_source_resolver_uses_loaded_comfyui_namespaced_module(monkeypatch):
    bare = "h3_flow_regenerate.first_high_operator_comparison"
    synthetic = "custom_nodes.synthetic_flow.h3_flow_regenerate.first_high_operator_comparison"
    original = sys.modules.pop(bare, None)
    monkeypatch.setitem(sys.modules, synthetic, w)
    monkeypatch.setattr(plugin_root.importlib, "import_module", _deny_bare_import)
    try:
        resolved = plugin_root._resolve_w_source_entry({"module": bare, "relative_path": "."})
    finally:
        sys.modules.pop(synthetic, None)
        if original is not None:
            sys.modules[bare] = original

    assert resolved == Path(w.__file__).resolve(strict=True)


def test_w_source_resolver_fails_closed_on_distinct_suffix_sources(monkeypatch, tmp_path):
    bare = "h3_flow_regenerate.first_high_operator_comparison"
    synthetic = "custom_nodes.synthetic_flow.h3_flow_regenerate.first_high_operator_comparison"
    foreign_name = "custom_nodes.foreign_flow.h3_flow_regenerate.first_high_operator_comparison"
    foreign_path = tmp_path / "first_high_operator_comparison.py"
    foreign_path.write_text("# foreign\n", encoding="utf-8")
    foreign = ModuleType(foreign_name)
    foreign.__file__ = str(foreign_path)

    original = sys.modules.pop(bare, None)
    monkeypatch.setitem(sys.modules, synthetic, w)
    monkeypatch.setitem(sys.modules, foreign_name, foreign)
    monkeypatch.setattr(plugin_root.importlib, "import_module", _deny_bare_import)
    try:
        with pytest.raises(RuntimeError, match="resolves ambiguously"):
            plugin_root._resolve_w_source_entry({"module": bare, "relative_path": "."})
    finally:
        sys.modules.pop(synthetic, None)
        sys.modules.pop(foreign_name, None)
        if original is not None:
            sys.modules[bare] = original
