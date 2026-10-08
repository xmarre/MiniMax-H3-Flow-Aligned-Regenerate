"""Exercise prompt pruning and post-prompt GC with native Core model owners."""

import asyncio
import gc
import json
import weakref
from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate import local_boundary_audit as audit


@pytest.fixture
def core(monkeypatch):
    pytest.importorskip("comfy.cli_args").args.cpu = True
    import execution
    from comfy import model_management, model_patcher
    from comfy_execution import cache_provider

    class ModelNode:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {}}

    monkeypatch.setitem(execution.nodes.NODE_CLASS_MAPPINGS, "LifetimeModel", ModelNode)
    monkeypatch.setitem(execution.nodes.NODE_CLASS_MAPPINGS, "H3FlowLocalBoundaryAudit", audit.H3FlowLocalBoundaryAudit)
    monkeypatch.setattr(model_management, "current_loaded_models", [])
    detached = []
    monkeypatch.setattr(model_patcher.ModelPatcher, "detach", lambda self, **kwargs: detached.append(id(self)))
    owners = getattr(audit, "_AUDIT_MODEL_OWNERS", None)
    if owners is not None:
        owners = type(owners)()
        monkeypatch.setattr(audit, "_AUDIT_MODEL_OWNERS", owners)
    yield SimpleNamespace(
        execution=execution, management=model_management, patcher=model_patcher, owners=owners, detached=detached
    )
    if owners is not None and owners.registered:
        cache_provider.unregister_cache_provider(owners)
        owners.models = ()
    model_management.current_loaded_models.clear()
    gc.collect()


async def prepare(core, caches, prompt, prompt_id):
    # Same order as PromptExecutor.execute_async: notify, fingerprint, prune.
    notifier = core.execution.PromptExecutor._notify_prompt_lifecycle
    notifier(None, "start", prompt_id)
    dynamic = core.execution.DynamicPrompt(prompt)
    changed = core.execution.IsChangedCache(prompt_id, dynamic, caches.outputs)
    for cache in caches.all:
        await cache.set_prompt(dynamic, prompt.keys(), changed)
        cache.clean_unused()


async def seed_models(core, caches):
    await prepare(core, caches, {"model": {"class_type": "LifetimeModel", "inputs": {}}}, "generation")
    refs = []
    patchers = []
    for _ in range(3):
        patcher = core.patcher.ModelPatcher(torch.nn.Linear(2, 2), torch.device("cpu"), torch.device("cpu"))
        loaded = core.management.LoadedModel(patcher)
        loaded.real_model = weakref.ref(patcher.model)
        core.management.current_loaded_models.append(loaded)
        refs.append(weakref.ref(patcher))
        patchers.append(patcher)
    await caches.outputs.set("model", core.execution.CacheEntry({}, [patchers]))
    caches.objects.set_local("model", SimpleNamespace(models=patchers))
    return refs


def audit_prompt():
    return {"audit": {"class_type": "H3FlowLocalBoundaryAudit", "inputs": {"bundle_path": "bundle"}}}


@pytest.mark.parametrize("guarded", [False, True])
@pytest.mark.parametrize("cache_type", ["CLASSIC", "LRU", "RAM_PRESSURE"])
def test_native_prompt_pruning_and_post_audit_gc(core, monkeypatch, guarded, cache_type):
    if not guarded:
        monkeypatch.setattr(
            audit.H3FlowLocalBoundaryAudit, "IS_CHANGED", classmethod(lambda cls, **kwargs: float("nan"))
        )

    async def run():
        caches = core.execution.CacheSet(getattr(core.execution.CacheType, cache_type), {"lru": 0})
        refs = await seed_models(core, caches)
        prompt = audit_prompt()
        await prepare(core, caches, prompt, "audit-1")
        if cache_type == "RAM_PRESSURE":
            from comfy_execution import caching

            monkeypatch.setattr(caching, "virtual_memory_available", lambda: 0)
            caches.outputs.ram_release(1, min_entry_size=0)
        core.execution.PromptExecutor._notify_prompt_lifecycle(None, "end", "audit-1")
        gc.collect()
        assert all(ref() is not None for ref in refs) is guarded
        assert len(core.detached) == (0 if guarded else 3)
        # ComfyUI saves this same prompt in history. It must contain no models.
        json.dumps(prompt)
        if guarded:
            for index in range(3):
                await prepare(core, caches, audit_prompt(), f"repeat-{index}")
                core.execution.PromptExecutor._notify_prompt_lifecycle(None, "end", f"repeat-{index}")
                gc.collect()
                assert all(ref() is not None for ref in refs)
                assert len(core.owners.models) == 3
            await prepare(core, caches, {"model": {"class_type": "LifetimeModel", "inputs": {}}}, "next-generation")
            core.execution.PromptExecutor._notify_prompt_lifecycle(None, "end", "next-generation")
            gc.collect()
            assert all(ref() is None for ref in refs)
            assert len(core.detached) == 3

    asyncio.run(run())


def test_failed_audit_still_preserves_models_and_explicit_unload_remains_available(core, monkeypatch):
    async def run():
        caches = core.execution.CacheSet()
        refs = await seed_models(core, caches)
        await prepare(core, caches, audit_prompt(), "failed-audit")
        # Execution raises before producing an output, but Core invokes end in finally.
        core.execution.PromptExecutor._notify_prompt_lifecycle(None, "end", "failed-audit")
        gc.collect()
        assert all(ref() is not None for ref in refs)
        unloaded = []

        def unload(self, *args, **kwargs):
            unloaded.append(id(self.model))
            return True

        monkeypatch.setattr(core.management.LoadedModel, "model_unload", unload)
        core.management.unload_all_models()
        assert len(unloaded) == 3
        assert core.management.current_loaded_models == []
        assert core.management.loaded_models() == []

    asyncio.run(run())
