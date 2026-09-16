from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

import __init__ as plugin_root


def _module(name: str, path: Path, *, package: bool = False) -> ModuleType:
    value = ModuleType(name)
    value.__file__ = str(path)
    if package:
        value.__path__ = [str(path.parent)]
    return value


def _remove_suffix_modules(monkeypatch, module_name: str) -> None:
    suffix = f".{module_name}"
    for loaded_name in tuple(sys.modules):
        if loaded_name == module_name or loaded_name.endswith(suffix):
            monkeypatch.delitem(sys.modules, loaded_name, raising=False)


def test_candidate_source_resolver_uses_loaded_comfy_namespaced_module(monkeypatch, tmp_path):
    _remove_suffix_modules(monkeypatch, "sol_h3")
    _remove_suffix_modules(monkeypatch, "sol_h3.runtime")
    package_path = tmp_path / "sol_h3" / "__init__.py"
    runtime_path = tmp_path / "sol_h3" / "runtime.py"
    runtime_path.parent.mkdir(parents=True)
    package_path.write_text("# package\n", encoding="utf-8")
    runtime_path.write_text("# runtime\n", encoding="utf-8")
    package = _module("custom_nodes.synthetic.sol_h3", package_path, package=True)
    runtime = _module("custom_nodes.synthetic.sol_h3.runtime", runtime_path)
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)

    resolved = plugin_root._resolve_production_candidate_source_entry(
        {"owner": "sol", "module": "sol_h3.runtime", "relative_path": "."}
    )

    assert resolved == runtime_path.resolve(strict=True)


def test_candidate_source_resolver_keeps_unloaded_sm120_module_lazy(monkeypatch, tmp_path):
    _remove_suffix_modules(monkeypatch, "sol_h3")
    _remove_suffix_modules(monkeypatch, "sol_h3._vendor.sol_attn.sm120.mainloop")
    package_path = tmp_path / "sol_h3" / "__init__.py"
    mainloop_path = tmp_path / "sol_h3" / "_vendor" / "sol_attn" / "sm120" / "mainloop.py"
    mainloop_path.parent.mkdir(parents=True)
    package_path.parent.mkdir(parents=True, exist_ok=True)
    package_path.write_text("# package\n", encoding="utf-8")
    mainloop_path.write_text("# mainloop\n", encoding="utf-8")
    package = _module("custom_nodes.synthetic.sol_h3", package_path, package=True)
    monkeypatch.setitem(sys.modules, package.__name__, package)

    target_module = "sol_h3._vendor.sol_attn.sm120.mainloop"
    assert target_module not in sys.modules
    resolved = plugin_root._resolve_production_candidate_source_entry(
        {
            "owner": "sol",
            "module": "sol_h3._vendor.sol_attn.sm120.mainloop",
            "relative_path": ".",
        }
    )

    assert resolved == mainloop_path.resolve(strict=True)
    assert target_module not in sys.modules


def test_candidate_source_resolver_fails_closed_on_distinct_suffix_sources(monkeypatch, tmp_path):
    _remove_suffix_modules(monkeypatch, "sol_h3.runtime")
    one = tmp_path / "one" / "runtime.py"
    two = tmp_path / "two" / "runtime.py"
    one.parent.mkdir(parents=True)
    two.parent.mkdir(parents=True)
    one.write_text("# one\n", encoding="utf-8")
    two.write_text("# two\n", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "custom_nodes.one.sol_h3.runtime", _module("one", one))
    monkeypatch.setitem(sys.modules, "custom_nodes.two.sol_h3.runtime", _module("two", two))

    with pytest.raises(RuntimeError, match="resolves ambiguously"):
        plugin_root._resolve_production_candidate_source_entry(
            {"owner": "sol", "module": "sol_h3.runtime", "relative_path": "."}
        )


def test_candidate_contract_aliases_are_temporary_and_source_exact(monkeypatch, tmp_path):
    for name in ("sol_h3", "sol_h3.provenance", "vdn_h3", "vdn_h3.softmax_provider"):
        _remove_suffix_modules(monkeypatch, name)

    sol_pkg_path = tmp_path / "sol" / "sol_h3" / "__init__.py"
    sol_child_path = tmp_path / "sol" / "sol_h3" / "provenance.py"
    vdn_pkg_path = tmp_path / "vdn" / "vdn_h3" / "__init__.py"
    vdn_child_path = tmp_path / "vdn" / "vdn_h3" / "softmax_provider.py"
    for path in (sol_pkg_path, sol_child_path, vdn_pkg_path, vdn_child_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# source\n", encoding="utf-8")

    sol_pkg = _module("custom_nodes.synthetic_sol.sol_h3", sol_pkg_path, package=True)
    sol_child = _module("custom_nodes.synthetic_sol.sol_h3.provenance", sol_child_path)
    vdn_pkg = _module("custom_nodes.synthetic_vdn.vdn_h3", vdn_pkg_path, package=True)
    vdn_child = _module("custom_nodes.synthetic_vdn.vdn_h3.softmax_provider", vdn_child_path)
    sol_pkg.provenance = sol_child
    vdn_pkg.softmax_provider = vdn_child
    for module in (sol_pkg, sol_child, vdn_pkg, vdn_child):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    def fake_verify():
        from sol_h3 import provenance
        from vdn_h3 import softmax_provider

        assert provenance is sol_child
        assert softmax_provider is vdn_child
        return {"verified": True}

    monkeypatch.setattr(plugin_root, "_ORIGINAL_PRODUCTION_CANDIDATE_SOURCE_VERIFY", fake_verify)
    assert plugin_root._verify_production_candidate_sources() == {"verified": True}
    assert "sol_h3" not in sys.modules
    assert "sol_h3.provenance" not in sys.modules
    assert "vdn_h3" not in sys.modules
    assert "vdn_h3.softmax_provider" not in sys.modules


def test_candidate_contract_children_remain_lazy_until_original_verifier(monkeypatch, tmp_path):
    for name in ("sol_h3", "sol_h3.provenance", "vdn_h3", "vdn_h3.softmax_provider"):
        _remove_suffix_modules(monkeypatch, name)

    sol_pkg_path = tmp_path / "sol" / "sol_h3" / "__init__.py"
    vdn_pkg_path = tmp_path / "vdn" / "vdn_h3" / "__init__.py"
    for path in (sol_pkg_path, vdn_pkg_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# package\n", encoding="utf-8")

    sol_pkg = _module("custom_nodes.synthetic_sol.sol_h3", sol_pkg_path, package=True)
    vdn_pkg = _module("custom_nodes.synthetic_vdn.vdn_h3", vdn_pkg_path, package=True)
    monkeypatch.setitem(sys.modules, sol_pkg.__name__, sol_pkg)
    monkeypatch.setitem(sys.modules, vdn_pkg.__name__, vdn_pkg)

    def fake_verify():
        assert sys.modules["sol_h3"] is sol_pkg
        assert sys.modules["vdn_h3"] is vdn_pkg
        assert "sol_h3.provenance" not in sys.modules
        assert "vdn_h3.softmax_provider" not in sys.modules
        return {"verified": True}

    monkeypatch.setattr(plugin_root, "_ORIGINAL_PRODUCTION_CANDIDATE_SOURCE_VERIFY", fake_verify)
    assert plugin_root._verify_production_candidate_sources() == {"verified": True}
    assert "sol_h3" not in sys.modules
    assert "vdn_h3" not in sys.modules
