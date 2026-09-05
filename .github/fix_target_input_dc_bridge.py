from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one match, found {count}: {old[:80]!r}")
    p.write_text(text.replace(old, new, 1))


def replace_all(path: str, old: str, new: str, expected: int) -> None:
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != expected:
        raise RuntimeError(f"{path}: expected {expected} matches, found {count}: {old[:80]!r}")
    p.write_text(text.replace(old, new))


# The generic Target Input node is not a Continuum-specific node. Remove the
# Continuum-only seam control from its public schema and patch signature.
replace_once(
    "h3_flow_regenerate/nodes.py",
    '''                "suffix_dc_bridge": (\n                    "BOOLEAN",\n                    {\n                        "default": True,\n                        "tooltip": (\n                            "One-token per-channel DC correction at canonical Continuum exact-prefix "\n                            "boundaries. It preserves the exact prefix and changes only the first "\n                            "generated suffix latent token. Enabled by default after the boundary "\n                            "flash was removed in decoded-media validation."\n                        ),\n                    },\n                ),\n''',
    "",
)
replace_once(
    "h3_flow_regenerate/nodes.py",
    '''        learned_upscaler=None,\n        suffix_dc_bridge=True,\n    ):\n''',
    '''        learned_upscaler=None,\n    ):\n''',
)
replace_all(
    "h3_flow_regenerate/nodes.py",
    '''                suffix_dc_bridge=bool(suffix_dc_bridge),\n''',
    "",
    2,
)

# Keep the internal config field because the dedicated Continuum modes use it,
# but make the generic fallback impossible to opt into accidentally.
replace_once(
    "h3_flow_regenerate/handoff.py",
    '''    suffix_dc_bridge: bool = True\n''',
    '''    suffix_dc_bridge: bool = False\n''',
)
replace_once(
    "h3_flow_regenerate/handoff.py",
    '''        if not isinstance(self.suffix_dc_bridge, bool):\n            raise TypeError("suffix_dc_bridge must be boolean")\n        if self.min_high_steps < 1:\n''',
    '''        if not isinstance(self.suffix_dc_bridge, bool):\n            raise TypeError("suffix_dc_bridge must be boolean")\n        if self.suffix_dc_bridge and self.exact_prefix_mode not in {"target_sparse_lifter", "mixed_grid_low_suffix"}:\n            raise ValueError("suffix_dc_bridge is only supported by Continuum-specific exact-prefix modes")\n        if self.min_high_steps < 1:\n''',
)

# Target-Sparse is Continuum-specific, so it gets its own control instead of
# inheriting one from the generic Target Input node. Mixed-Grid inherits this
# schema and keeps its learned-only override.
replace_once(
    "h3_flow_regenerate/target_sparse_node.py",
    '''    DESCRIPTION = (\n        "Experimental Continuum continuation path. Exact Native Masked video prefixes stay on the "\n        "target grid; early H3 transformer work retains every protected video row plus a coarse "\n        "target-grid anchor lattice for generated rows, then restores the full hidden grid before "\n        "H3's native final layer. source_scale/source_width/source_height control anchor density on "\n        "exact-prefix chunks, not sampler latent geometry. The ordinary Target Input node remains "\n        "the conservative full-target fallback. The shared one-token suffix DC bridge is "\n        "enabled by default on canonical Continuum exact-prefix boundaries."\n    )\n\n    def patch(\n''',
    '''    DESCRIPTION = (\n        "Experimental Continuum continuation path. Exact Native Masked video prefixes stay on the "\n        "target grid; early H3 transformer work retains every protected video row plus a coarse "\n        "target-grid anchor lattice for generated rows, then restores the full hidden grid before "\n        "H3's native final layer. source_scale/source_width/source_height control anchor density on "\n        "exact-prefix chunks, not sampler latent geometry. The one-token suffix DC bridge is "\n        "Continuum-specific and enabled by default on canonical exact-prefix boundaries."\n    )\n\n    @classmethod\n    def INPUT_TYPES(cls):\n        import copy\n\n        inputs = copy.deepcopy(super().INPUT_TYPES())\n        inputs["required"]["suffix_dc_bridge"] = (\n            "BOOLEAN",\n            {\n                "default": True,\n                "tooltip": (\n                    "Continuum-only one-token per-channel DC seam correction. It calibrates from the first "\n                    "actual full-grid H3 predicted-clean boundary, preserves the authoritative prefix, and "\n                    "changes only the first generated suffix latent token."\n                ),\n            },\n        )\n        return inputs\n\n    def patch(\n''',
)

# Remove the bridge from the generic target-input exact-mask fallback entirely.
replace_once(
    "h3_flow_regenerate/runtime.py",
    '''        bridge_prefix = None\n        if config.suffix_dc_bridge:\n            bridge_prefix = _contiguous_exact_video_prefix(\n                guider.model_patcher.model,\n                latent_image,\n                denoise_mask,\n                input_shapes,\n            )\n            if bridge_prefix is None:\n                binding.metrics.event(\n                    "exact_prefix_suffix_dc_bridge_skipped",\n                    source="target_input_fallback",\n                    reason="noncanonical_exact_mask",\n                )\n        bridge_context = (\n            _exact_prefix_suffix_bridge_contract(\n                guider,\n                bridge_prefix,\n                source="target_input_fallback",\n            )\n            if bridge_prefix is not None\n            else contextlib.nullcontext(None)\n        )\n        _begin_capture(binding, guider, sampler, sigmas, input_shapes)\n        error: BaseException | None = None\n        try:\n            with bridge_context as bridge_contract:\n                result = executor(\n                    noise,\n                    latent_image,\n                    sampler,\n                    sigmas,\n                    denoise_mask,\n                    callback,\n                    disable_pbar,\n                    seed,\n                    latent_shapes=latent_shapes,\n                )\n            if isinstance(bridge_contract, dict) and not bool(bridge_contract.get("applied")):\n                binding.metrics.event(\n                    "exact_prefix_suffix_dc_bridge_skipped",\n                    source="target_input_fallback",\n                    reason="no_actual_model_output",\n                )\n            return result\n        except BaseException as exc:\n            error = exc\n            raise\n        finally:\n            _finish_capture(binding, error=error)\n''',
    '''        _begin_capture(binding, guider, sampler, sigmas, input_shapes)\n        error: BaseException | None = None\n        try:\n            return executor(\n                noise,\n                latent_image,\n                sampler,\n                sigmas,\n                denoise_mask,\n                callback,\n                disable_pbar,\n                seed,\n                latent_shapes=latent_shapes,\n            )\n        except BaseException as exc:\n            error = exc\n            raise\n        finally:\n            _finish_capture(binding, error=error)\n''',
)

# Regression expectations: generic Target Input has no Continuum seam option;
# only the two dedicated Continuum nodes expose it.
replace_once(
    "tests/test_target_sparse_node.py",
    '''def test_suffix_dc_bridge_is_exposed_on_all_continuum_progressive_nodes_and_defaults_on():\n    target_inputs = H3ProgressiveTargetInputHandoff.INPUT_TYPES()\n    sparse_inputs = H3ProgressiveTargetSparseHandoff.INPUT_TYPES()\n    mixed_inputs = H3ProgressiveMixedGridHandoff.INPUT_TYPES()\n    for inputs in (target_inputs, sparse_inputs, mixed_inputs):\n        bridge = inputs["required"]["suffix_dc_bridge"]\n        assert bridge[0] == "BOOLEAN"\n        assert bridge[1]["default"] is True\n    assert mixed_inputs["required"]["handoff_transfer"][0] == ["learned_3d"]\n    assert sparse_inputs["required"]["handoff_transfer"][0] == ["bicubic", "learned_3d"]\n''',
    '''def test_suffix_dc_bridge_is_exposed_only_on_continuum_specific_progressive_nodes():\n    target_inputs = H3ProgressiveTargetInputHandoff.INPUT_TYPES()\n    sparse_inputs = H3ProgressiveTargetSparseHandoff.INPUT_TYPES()\n    mixed_inputs = H3ProgressiveMixedGridHandoff.INPUT_TYPES()\n    assert "suffix_dc_bridge" not in target_inputs["required"]\n    for inputs in (sparse_inputs, mixed_inputs):\n        bridge = inputs["required"]["suffix_dc_bridge"]\n        assert bridge[0] == "BOOLEAN"\n        assert bridge[1]["default"] is True\n    assert mixed_inputs["required"]["handoff_transfer"][0] == ["learned_3d"]\n    assert sparse_inputs["required"]["handoff_transfer"][0] == ["bicubic", "learned_3d"]\n''',
)
replace_once(
    "tests/test_handoff.py",
    '''def test_suffix_dc_bridge_config_is_shared_across_continuum_modes_and_boolean():\n    provider = FakeLearnedProvider()\n    for exact_prefix_mode in ("fallback", "target_sparse_lifter"):\n        config = ProgressiveTargetInputConfig(\n            source_latent_h=4,\n            source_latent_w=4,\n            exact_prefix_mode=exact_prefix_mode,\n        )\n        assert config.suffix_dc_bridge is True\n    mixed = ProgressiveTargetInputConfig(\n        source_latent_h=4,\n        source_latent_w=4,\n        transfer_mode="learned_3d",\n        learned_upscaler=provider,\n        exact_prefix_mode="mixed_grid_low_suffix",\n    )\n    assert mixed.suffix_dc_bridge is True\n    with pytest.raises(TypeError, match="must be boolean"):\n        ProgressiveTargetInputConfig(\n            source_latent_h=4,\n            source_latent_w=4,\n            suffix_dc_bridge=1,\n        )\n''',
    '''def test_suffix_dc_bridge_config_is_restricted_to_continuum_specific_modes_and_boolean():\n    provider = FakeLearnedProvider()\n    fallback = ProgressiveTargetInputConfig(source_latent_h=4, source_latent_w=4)\n    assert fallback.suffix_dc_bridge is False\n    with pytest.raises(ValueError, match="Continuum-specific"):\n        ProgressiveTargetInputConfig(\n            source_latent_h=4,\n            source_latent_w=4,\n            suffix_dc_bridge=True,\n        )\n    sparse = ProgressiveTargetInputConfig(\n        source_latent_h=4,\n        source_latent_w=4,\n        exact_prefix_mode="target_sparse_lifter",\n        suffix_dc_bridge=True,\n    )\n    assert sparse.suffix_dc_bridge is True\n    mixed = ProgressiveTargetInputConfig(\n        source_latent_h=4,\n        source_latent_w=4,\n        transfer_mode="learned_3d",\n        learned_upscaler=provider,\n        exact_prefix_mode="mixed_grid_low_suffix",\n        suffix_dc_bridge=True,\n    )\n    assert mixed.suffix_dc_bridge is True\n    with pytest.raises(TypeError, match="must be boolean"):\n        ProgressiveTargetInputConfig(\n            source_latent_h=4,\n            source_latent_w=4,\n            suffix_dc_bridge=1,\n        )\n''',
)

# Documentation must not present the generic Target Input node as owning a
# Continuum-only seam correction.
replace_once(
    "README.md",
    '''All three Continuum progressive nodes expose `suffix_dc_bridge`, enabled by default on canonical whole-frame exact-prefix boundaries. The bridge changes only the first generated suffix latent token with a per-channel DC offset; it never edits the authoritative prefix or later suffix tokens. Mixed-Grid derives the offset from its discarded learned-upscaler prefix, while Target Input fallback and Target-Sparse derive it from the first actual H3 predicted-clean boundary before native mask restoration. Disable it only for matched seam A/B testing.\n''',
    '''The two dedicated exact-prefix Continuum nodes, **Progressive Target-Sparse Continuum** and **Progressive Mixed-Grid Continuum**, expose `suffix_dc_bridge`, enabled by default on canonical whole-frame exact-prefix boundaries. The bridge changes only the first generated suffix latent token with a per-channel DC offset; it never edits the authoritative prefix or later suffix tokens. Mixed-Grid derives the offset from its discarded learned-upscaler prefix, while Target-Sparse derives it from the first actual full-grid H3 predicted-clean boundary before native mask restoration. The generic **Progressive Handoff (Target Input)** node does not expose or apply this Continuum-specific seam correction.\n''',
)
replace_once(
    "README.md",
    '''| **MiniMax H3 Progressive Handoff (Target Input)** | Target-sized/Continuum-safe variant. Unprotected calls run the early H3 stage privately on a smaller grid; exact protected video prefixes conservatively fall back to one ordinary target-grid sampler lifetime. Supports optional `learned_3d` transfer on the normal handoff path. |\n''',
    '''| **MiniMax H3 Progressive Handoff (Target Input)** | Generic target-sized variant. Unprotected calls run the early H3 stage privately on a smaller grid; exact protected video prefixes conservatively fall back to one ordinary target-grid sampler lifetime. Supports optional `learned_3d` transfer on the normal handoff path. It does not own Continuum-specific seam correction. |\n''',
)
replace_once(
    "docs/USAGE.md",
    '''### Exact-prefix suffix DC bridge\n\nThe Continuum-facing **Progressive Handoff (Target Input)**, **Progressive Target-Sparse Continuum**, and **Progressive Mixed-Grid Continuum** nodes expose `suffix_dc_bridge`. It defaults to **on** and is only defined for a canonical whole-frame exact prefix followed by a whole-frame generated suffix. Partial/fractional/noncontiguous masks are skipped rather than guessed.\n\nThe correction is intentionally one latent token wide. Flow computes a per-batch/per-channel spatial-mean offset and applies it only to the first generated suffix token. The authoritative prefix and every later suffix token remain unchanged at bridge application. There is no video-space crossfade and no additional H3 transformer evaluation.\n\nThe calibration source depends on the progressive topology:\n\n- **Mixed-Grid:** use the discarded learned-upscaler prefix to preserve the learned transfer's native DC boundary relation when the exact target-grid prefix is restored.\n- **Target Input fallback / Target-Sparse high stage:** use the first **actual** H3 predicted-clean output at the exact boundary before native inpaint masking restores the protected prefix. Spectrum forecasts are not used as calibration; the bridge waits for the first actual model output. Target-Sparse already requires its first high-stage call to be actual. The conservative fallback reports `exact_prefix_suffix_dc_bridge_skipped` with `reason=no_actual_model_output` if a surrounding policy never produces an actual H3 evaluation.\n\nThe mixed-grid placement passed the matched decoded-media seam test that motivated enabling the control by default. The generalized Target Input and Target-Sparse placements are structurally covered by CI but should still be included in matched decoded-media validation for those topologies.\n''',
    '''### Exact-prefix suffix DC bridge\n\nOnly the dedicated **Progressive Target-Sparse Continuum** and **Progressive Mixed-Grid Continuum** nodes expose `suffix_dc_bridge`. It defaults to **on** and is only defined for a canonical whole-frame exact prefix followed by a whole-frame generated suffix. The generic **Progressive Handoff (Target Input)** node does not expose or apply this Continuum-specific seam correction. Partial/fractional/noncontiguous masks are skipped rather than guessed.\n\nThe correction is intentionally one latent token wide. Flow computes a per-batch/per-channel spatial-mean offset and applies it only to the first generated suffix token. The authoritative prefix and every later suffix token remain unchanged at bridge application. There is no video-space crossfade and no additional H3 transformer evaluation.\n\nThe calibration source depends on the Continuum topology:\n\n- **Mixed-Grid:** use the discarded learned-upscaler prefix to preserve the learned transfer's native DC boundary relation when the exact target-grid prefix is restored.\n- **Target-Sparse high stage:** use the first **actual** full-grid H3 predicted-clean output at the exact boundary before native inpaint masking restores the protected prefix. Spectrum forecasts are not used as calibration; the bridge waits for the first actual model output. Target-Sparse already requires its first high-stage call to be actual.\n\nThe Mixed-Grid placement passed the matched decoded-media seam test that motivated enabling the control by default. The Target-Sparse placement is structurally covered by CI but still needs its own matched decoded-media validation.\n''',
)

print("Removed Continuum suffix bridge from generic Target Input path")
