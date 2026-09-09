from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


runtime_path = Path("h3_flow_regenerate/runtime.py")
runtime = runtime_path.read_text()
runtime = replace_once(
    runtime,
    "from .source_trajectory_bridge import apply_source_trajectory_bridge\n",
    "from .source_trajectory_bridge import disabled_source_trajectory_bridge_metrics\n",
    "source bridge import",
)
runtime = replace_once(
    runtime,
    """                source_h=source_h,\n                source_w=source_w,\n            )\n""",
    """                source_h=source_h,\n                source_w=source_w,\n                attention_measure=bool(getattr(config, \"suffix_geometric_bridge\", False)),\n            )\n""",
    "mixed plan attention measure",
)
runtime = replace_once(
    runtime,
    """                low_suffix_real_latent=True,\n            )\n""",
    """                low_suffix_real_latent=True,\n                attention_measure_requested=bool(mixed_plan.attention_measure),\n                attention_measure_contract=(\n                    \"h3_flow_mixed_grid_attention_measure_v1\"\n                    if mixed_plan.attention_measure\n                    else None\n                ),\n            )\n""",
    "mixed plan telemetry",
)
old_source = """            source_bridge_requested = bool(getattr(config, \"suffix_geometric_bridge\", False))\n            clean_video, source_trajectory_metrics = apply_source_trajectory_bridge(\n                clean_video,\n                mixed_plan.prefix_t,\n                requested=source_bridge_requested,\n            )\n            binding.metrics.event(\n                \"mixed_grid_source_trajectory_bridge\",\n                legacy_option_name=\"suffix_geometric_bridge\",\n                authoritative_source_prefix_modified=False,\n                later_suffix_extrapolated=False,\n                learned_upscaler_input_modified=bool(source_trajectory_metrics[\"source_trajectory_bridge_accepted\"]),\n                **source_trajectory_metrics,\n            )\n"""
new_source = """            source_bridge_requested = bool(getattr(config, \"suffix_geometric_bridge\", False))\n            # 00318 exhausted every finite verified source-warp horizon. Shortening\n            # a non-zero warp only moved the compensating corrected->untouched\n            # transition earlier, so this repair family is retired rather than\n            # weakened with a fade or unmeasured extrapolation. The experimental\n            # option now owns only the upstream attention-measure and target-side\n            # exact-overlap reconciliation paths.\n            source_trajectory_metrics = disabled_source_trajectory_bridge_metrics(\n                prefix_t=mixed_plan.prefix_t,\n                requested=source_bridge_requested,\n            )\n            if source_bridge_requested:\n                source_trajectory_metrics[\"source_trajectory_bridge_reason\"] = \"retired_source_warp_family\"\n            binding.metrics.event(\n                \"mixed_grid_source_trajectory_bridge\",\n                legacy_option_name=\"suffix_geometric_bridge\",\n                authoritative_source_prefix_modified=False,\n                later_suffix_extrapolated=False,\n                learned_upscaler_input_modified=False,\n                **source_trajectory_metrics,\n            )\n"""
runtime = replace_once(runtime, old_source, new_source, "retire source warp")
runtime_path.write_text(runtime)

mixed_path = Path("h3_flow_regenerate/mixed_grid.py")
mixed = mixed_path.read_text()
if not mixed.endswith("\n"):
    mixed_path.write_text(mixed + "\n")


test_path = Path("tests/test_mixed_grid.py")
test = test_path.read_text()
test = replace_once(
    test,
    """    build_mixed_grid_plan,\n    carrier_layout,\n    mixed_mod_segments,\n    mixed_positions,\n)\n""",
    """    build_mixed_grid_plan,\n    carrier_layout,\n    mixed_attention_measure_contract,\n    mixed_mod_segments,\n    mixed_positions,\n)\n""",
    "mixed test import",
)
start = test.index("def test_source_trajectory_bridge_runs_before_learned_upscaler(monkeypatch):")
end = test.index("\ndef test_native_forward_uses_authoritative_prefix_and_real_suffix", start)
replacement = '''def test_retired_source_warp_does_not_modify_learned_upscaler_input(monkeypatch):\n    import sys\n    from types import ModuleType, SimpleNamespace\n\n    from test_handoff import FakeLearnedProvider\n\n    from h3_flow_regenerate.geometry import resize_spatial_5d, unpack_streams\n    from h3_flow_regenerate.handoff import ProgressiveTargetInputConfig\n    from h3_flow_regenerate.runtime import FlowBinding, _run_progressive\n\n    fake = ModuleType("comfy")\n    fake.samplers = ModuleType("comfy.samplers")\n    fake.samplers.KSAMPLER = lambda function, **kw: SimpleNamespace(sampler_function=function, extra_options={})\n    monkeypatch.setitem(sys.modules, "comfy", fake)\n    monkeypatch.setitem(sys.modules, "comfy.samplers", fake.samplers)\n\n    packed, shapes, mask = inputs(t=9, prefix=6)\n    base = SimpleNamespace(process_latent_in=lambda value: value, diffusion_model=SimpleNamespace(blocks=[]))\n    guider = SimpleNamespace(\n        model_options={"transformer_options": {}},\n        model_patcher=SimpleNamespace(model=base),\n        conds={"positive": []},\n    )\n    binding = FlowBinding()\n    provider = FakeLearnedProvider()\n    config = ProgressiveTargetInputConfig(\n        source_latent_h=4,\n        source_latent_w=6,\n        exact_prefix_mode="mixed_grid_low_suffix",\n        transfer_mode="learned_3d",\n        learned_upscaler=provider,\n        suffix_geometric_bridge=True,\n    )\n\n    def execute(noise, latent, sampler, sigmas, call_mask, *args, latent_shapes):\n        stage = guider.model_options["transformer_options"]["h3_flow_stage"]\n        contract = guider.model_options["transformer_options"].get("h3_flow_mixed_grid_v1")\n        if stage != "high":\n            assert contract is not None\n            assert contract["plan"].attention_measure is True\n        if stage == "high":\n            binding.metrics.event("model_call", actual=True)\n            return packed.clone()\n        if stage == "probe":\n            return latent.clone()\n        return latent / (1 - sigmas[-1])\n\n    sampler = SimpleNamespace(sampler_function=lambda: None, extra_options={})\n    _run_progressive(\n        execute,\n        guider,\n        binding,\n        config,\n        torch.randn_like(packed),\n        packed,\n        sampler,\n        torch.tensor([1.0, 0.9, 0.7, 0.4, 0.0]),\n        mask,\n        None,\n        True,\n        7,\n        list(shapes),\n    )\n    provider_input = provider.calls[0][0]\n    original_video, _ = unpack_streams(packed, shapes)\n    expected_video = resize_spatial_5d(original_video, 4, 6, mode="bicubic")\n    assert torch.equal(provider_input, expected_video)\n    source_events = [event for event in binding.metrics.events if event.kind == "mixed_grid_source_trajectory_bridge"]\n    assert len(source_events) == 1\n    assert source_events[0].fields["source_trajectory_bridge_reason"] == "retired_source_warp_family"\n    assert source_events[0].fields["source_trajectory_bridge_tokens_corrected"] == 0\n    assert source_events[0].fields["learned_upscaler_input_modified"] is False\n\n\ndef test_mixed_attention_measure_contract_00318_geometry():\n    plan = MixedGridPlan(\n        torch.randn(1, 24, 12, 56, 76),\n        62,\n        40,\n        54,\n        attention_measure=True,\n    )\n    contract = mixed_attention_measure_contract(plan, video_start=16261, sequence_rows=56029)\n    assert contract is not None\n    assert plan.target_grid == (28, 38)\n    assert plan.source_grid == (20, 27)\n    assert plan.target_rows == 1064\n    assert plan.source_rows == 540\n    assert contract["sequence_rows"] == 56029\n    assert contract["expected_kv_rows"] == 49741\n    assert contract["prefix_rows_per_frame"] / contract["source_rows_per_frame"] == pytest.approx(1064 / 540)\n    assert contract["exact_prefix_queries_preserved"] is True\n    assert contract["suffix_kv_unchanged"] is True\n\n\ndef test_mixed_attention_measure_is_off_by_default():\n    plan = MixedGridPlan(torch.randn(1, 24, 2, 8, 12), 7, 4, 6)\n    assert mixed_attention_measure_contract(plan, video_start=5, sequence_rows=83) is None\n\n'''
test = test[:start] + replacement + test[end + 1:]
test_path.write_text(test)
'''
# strip accidental terminator from replacement authoring
replacement_script = replacement
