from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one match, found {count}\n--- old ---\n{old}")
    p.write_text(text.replace(old, new, 1))


replace_once(
    "h3_flow_regenerate/handoff.py",
    '    suffix_dc_bridge: bool = False\n',
    '    suffix_dc_bridge: bool = True\n',
)
replace_once(
    "h3_flow_regenerate/handoff.py",
    '        if self.suffix_dc_bridge and self.exact_prefix_mode != "mixed_grid_low_suffix":\n'
    '            raise ValueError("suffix_dc_bridge is only supported by mixed-grid continuation")\n',
    '',
)

replace_once(
    "h3_flow_regenerate/nodes.py",
    '                "handoff_transfer": (\n'
    '                    ["bicubic", "learned_3d"],\n'
    '                    {\n'
    '                        "default": "bicubic",\n'
    '                        "tooltip": (\n'
    '                            "bicubic preserves the released handoff. learned_3d applies one connected "\n'
    '                            "H3 latent-upscaler provider to the exact-probe clean video state."\n'
    '                        ),\n'
    '                    },\n'
    '                ),\n',
    '                "handoff_transfer": (\n'
    '                    ["bicubic", "learned_3d"],\n'
    '                    {\n'
    '                        "default": "bicubic",\n'
    '                        "tooltip": (\n'
    '                            "bicubic preserves the released handoff. learned_3d applies one connected "\n'
    '                            "H3 latent-upscaler provider to the exact-probe clean video state."\n'
    '                        ),\n'
    '                    },\n'
    '                ),\n'
    '                "suffix_dc_bridge": (\n'
    '                    "BOOLEAN",\n'
    '                    {\n'
    '                        "default": True,\n'
    '                        "tooltip": (\n'
    '                            "One-token per-channel DC correction at canonical Continuum exact-prefix "\n'
    '                            "boundaries. It preserves the exact prefix and changes only the first "\n'
    '                            "generated suffix latent token. Enabled by default after the boundary "\n'
    '                            "flash was removed in decoded-media validation."\n'
    '                        ),\n'
    '                    },\n'
    '                ),\n',
)
replace_once(
    "h3_flow_regenerate/nodes.py",
    '        handoff_transfer="bicubic",\n        learned_upscaler=None,\n    ):\n',
    '        handoff_transfer="bicubic",\n        learned_upscaler=None,\n        suffix_dc_bridge=True,\n    ):\n',
)
for _ in range(2):
    replace_once(
        "h3_flow_regenerate/nodes.py",
        '                transfer_mode=handoff_transfer,\n                learned_upscaler=learned_upscaler,\n            )\n',
        '                transfer_mode=handoff_transfer,\n                suffix_dc_bridge=bool(suffix_dc_bridge),\n                learned_upscaler=learned_upscaler,\n            )\n',
    )

replace_once(
    "h3_flow_regenerate/target_sparse_node.py",
    '        suffix_dc_bridge=False,\n',
    '        suffix_dc_bridge=True,\n',
)
replace_once(
    "h3_flow_regenerate/target_sparse_node.py",
    '        "the conservative full-target fallback until decoded-media validation is complete."\n',
    '        "the conservative full-target fallback. The shared one-token suffix DC bridge is "\n'
    '        "enabled by default on canonical Continuum exact-prefix boundaries."\n',
)

replace_once(
    "h3_flow_regenerate/runtime.py",
    'FLOW_STAGE_KEY = "h3_flow_stage"\n',
    'FLOW_STAGE_KEY = "h3_flow_stage"\nEXACT_PREFIX_BRIDGE_KEY = "h3_flow_exact_prefix_bridge_v1"\n',
)

bridge_helpers = '''

def _contiguous_exact_video_prefix(base_model, latent_image, denoise_mask, shapes):
    """Return a canonical whole-frame Continuum prefix in model-domain latents.

    Arbitrary masks keep their existing behavior; the bridge is only defined for
    a contiguous all-zero video prefix followed by an all-one generated suffix.
    """
    if denoise_mask is None:
        return None
    if len(shapes) != 2:
        raise ValueError("exact-prefix suffix bridge requires native video/audio shapes")
    expected = (shapes[0][0], 1, sum(math.prod(shape[1:]) for shape in shapes))
    if tuple(denoise_mask.shape) != expected:
        raise ValueError("prepared H3 denoise mask does not match packed AV geometry")
    video_mask, _audio_mask = unpack_streams(denoise_mask, shapes)
    if not bool(torch.isfinite(video_mask).all().item()):
        raise ValueError("exact-prefix suffix bridge requires a finite video mask")
    temporal = int(shapes[0][2])
    frames = video_mask.permute(2, 0, 1, 3, 4).reshape(temporal, -1)
    protected = (frames == 0).all(1)
    generated = (frames == 1).all(1)
    prefix_t = int(protected.sum().item())
    if not 0 < prefix_t < temporal:
        return None
    if not bool(protected[:prefix_t].all().item() and generated[prefix_t:].all().item()):
        return None
    internal = _process_latent_in(base_model, latent_image, shapes)
    video, _audio = unpack_streams(internal, shapes)
    return video[:, :, :prefix_t].detach().clone()


@contextlib.contextmanager
def _exact_prefix_suffix_bridge_contract(guider, exact_prefix, *, source):
    options = getattr(guider, "model_options", None)
    if not isinstance(options, dict):
        raise RuntimeError("exact-prefix suffix bridge requires mutable model options")
    transformer = options.setdefault("transformer_options", {})
    if not isinstance(transformer, dict):
        raise RuntimeError("exact-prefix suffix bridge requires mutable transformer options")
    if EXACT_PREFIX_BRIDGE_KEY in transformer:
        raise RuntimeError("nested exact-prefix suffix bridge contract is unsupported")
    contract = {
        "exact_prefix": exact_prefix,
        "source": str(source),
        "applied": False,
    }
    transformer[EXACT_PREFIX_BRIDGE_KEY] = contract
    try:
        yield contract
    finally:
        transformer.pop(EXACT_PREFIX_BRIDGE_KEY, None)
'''
replace_once(
    "h3_flow_regenerate/runtime.py",
    '\n\n@contextlib.contextmanager\ndef _high_stage_contract(guider: Any):\n',
    bridge_helpers + '\n\n@contextlib.contextmanager\ndef _high_stage_contract(guider: Any):\n',
)

bridge_apply = '''
    exact_bridge = transformer.get(EXACT_PREFIX_BRIDGE_KEY)
    if isinstance(exact_bridge, dict) and not bool(exact_bridge.get("applied")) and actual:
        exact_prefix = exact_bridge.get("exact_prefix")
        base_model = getattr(guider, "inner_model", None)
        bridge_shapes = getattr(base_model, "latent_shapes", None)
        if not isinstance(bridge_shapes, list) or len(bridge_shapes) != 2:
            raise RuntimeError("exact-prefix suffix bridge could not resolve H3 AV latent shapes")
        video_x0, audio_x0 = unpack_streams(result, bridge_shapes)
        bridged_video, bridge_metrics = apply_suffix_dc_bridge(
            video_x0,
            exact_prefix,
            weights=(1.0,),
        )
        result, _ = pack_streams((bridged_video, audio_x0))
        exact_bridge["applied"] = True
        binding.metrics.event(
            "exact_prefix_suffix_dc_bridge",
            source=str(exact_bridge.get("source", "unknown")),
            stage=stage,
            sigma=sigma,
            coordinate=coordinate,
            actual=True,
            suffix_dc_bridge_state_mapping="first_actual_model_x0",
            **bridge_metrics,
        )

'''
replace_once(
    "h3_flow_regenerate/runtime.py",
    '    if binding.guidance is not None and binding.guidance.mode != "off":\n',
    bridge_apply + '    if binding.guidance is not None and binding.guidance.mode != "off":\n',
)

fallback_old = '''        _begin_capture(binding, guider, sampler, sigmas, input_shapes)
        error: BaseException | None = None
        try:
            return executor(
                noise,
                latent_image,
                sampler,
                sigmas,
                denoise_mask,
                callback,
                disable_pbar,
                seed,
                latent_shapes=latent_shapes,
            )
        except BaseException as exc:
            error = exc
            raise
        finally:
            _finish_capture(binding, error=error)
'''
fallback_new = '''        bridge_prefix = None
        if config.suffix_dc_bridge:
            bridge_prefix = _contiguous_exact_video_prefix(
                guider.model_patcher.model,
                latent_image,
                denoise_mask,
                input_shapes,
            )
            if bridge_prefix is None:
                binding.metrics.event(
                    "exact_prefix_suffix_dc_bridge_skipped",
                    source="target_input_fallback",
                    reason="noncanonical_exact_mask",
                )
        bridge_context = (
            _exact_prefix_suffix_bridge_contract(
                guider,
                bridge_prefix,
                source="target_input_fallback",
            )
            if bridge_prefix is not None
            else contextlib.nullcontext(None)
        )
        _begin_capture(binding, guider, sampler, sigmas, input_shapes)
        error: BaseException | None = None
        try:
            with bridge_context:
                return executor(
                    noise,
                    latent_image,
                    sampler,
                    sigmas,
                    denoise_mask,
                    callback,
                    disable_pbar,
                    seed,
                    latent_shapes=latent_shapes,
                )
        except BaseException as exc:
            error = exc
            raise
        finally:
            _finish_capture(binding, error=error)
'''
replace_once("h3_flow_regenerate/runtime.py", fallback_old, fallback_new)

sparse_old = '''        _reset_guider_conds(guider, template=conditioning_template)
        high_started = time.perf_counter()
        high_event_start = len(binding.metrics.events)
        sampler_invocation_count += 1
        history_boundary_count += 1
        binding.metrics.increment("progressive_sampler_invocations")
        binding.metrics.increment("progressive_history_boundaries")
        with _flow_stage_contract(guider, "high"), _high_stage_contract(guider):
            result = executor(
'''
sparse_new = '''        bridge_prefix = None
        if config.suffix_dc_bridge:
            bridge_prefix = _contiguous_exact_video_prefix(
                base_model,
                latent_image,
                denoise_mask,
                target_shapes,
            )
            if bridge_prefix is None:
                binding.metrics.event(
                    "exact_prefix_suffix_dc_bridge_skipped",
                    source="target_sparse_high",
                    reason="noncanonical_exact_mask",
                )
        bridge_context = (
            _exact_prefix_suffix_bridge_contract(
                guider,
                bridge_prefix,
                source="target_sparse_high",
            )
            if bridge_prefix is not None
            else contextlib.nullcontext(None)
        )

        _reset_guider_conds(guider, template=conditioning_template)
        high_started = time.perf_counter()
        high_event_start = len(binding.metrics.events)
        sampler_invocation_count += 1
        history_boundary_count += 1
        binding.metrics.increment("progressive_sampler_invocations")
        binding.metrics.increment("progressive_history_boundaries")
        with _flow_stage_contract(guider, "high"), _high_stage_contract(guider), bridge_context:
            result = executor(
'''
replace_once("h3_flow_regenerate/runtime.py", sparse_old, sparse_new)

replace_once(
    "tests/test_handoff.py",
    '''def test_suffix_dc_bridge_config_is_mixed_only_and_boolean():
    provider = FakeLearnedProvider()
    with pytest.raises(ValueError, match="only supported by mixed-grid"):
        ProgressiveTargetInputConfig(
            source_latent_h=4,
            source_latent_w=4,
            transfer_mode="learned_3d",
            learned_upscaler=provider,
            suffix_dc_bridge=True,
        )
    with pytest.raises(TypeError, match="must be boolean"):
        ProgressiveTargetInputConfig(
            source_latent_h=4,
            source_latent_w=4,
            transfer_mode="learned_3d",
            learned_upscaler=provider,
            exact_prefix_mode="mixed_grid_low_suffix",
            suffix_dc_bridge=1,
        )
    config = ProgressiveTargetInputConfig(
        source_latent_h=4,
        source_latent_w=4,
        transfer_mode="learned_3d",
        learned_upscaler=provider,
        exact_prefix_mode="mixed_grid_low_suffix",
        suffix_dc_bridge=True,
    )
    assert config.suffix_dc_bridge is True
''',
    '''def test_suffix_dc_bridge_config_is_shared_across_continuum_modes_and_boolean():
    provider = FakeLearnedProvider()
    for exact_prefix_mode in ("fallback", "target_sparse_lifter"):
        config = ProgressiveTargetInputConfig(
            source_latent_h=4,
            source_latent_w=4,
            exact_prefix_mode=exact_prefix_mode,
        )
        assert config.suffix_dc_bridge is True
    mixed = ProgressiveTargetInputConfig(
        source_latent_h=4,
        source_latent_w=4,
        transfer_mode="learned_3d",
        learned_upscaler=provider,
        exact_prefix_mode="mixed_grid_low_suffix",
    )
    assert mixed.suffix_dc_bridge is True
    with pytest.raises(TypeError, match="must be boolean"):
        ProgressiveTargetInputConfig(
            source_latent_h=4,
            source_latent_w=4,
            suffix_dc_bridge=1,
        )
''',
)

replace_once(
    "tests/test_target_sparse_node.py",
    'from h3_flow_regenerate.target_sparse_node import H3ProgressiveMixedGridHandoff, H3ProgressiveTargetSparseHandoff\n',
    'from h3_flow_regenerate.nodes import H3ProgressiveTargetInputHandoff\n'
    'from h3_flow_regenerate.target_sparse_node import H3ProgressiveMixedGridHandoff, H3ProgressiveTargetSparseHandoff\n',
)
replace_once(
    "tests/test_target_sparse_node.py",
    '''def test_suffix_dc_bridge_is_exposed_only_on_mixed_grid_and_defaults_on():
    sparse_inputs = H3ProgressiveTargetSparseHandoff.INPUT_TYPES()
    mixed_inputs = H3ProgressiveMixedGridHandoff.INPUT_TYPES()
    assert "suffix_dc_bridge" not in sparse_inputs["required"]
    bridge = mixed_inputs["required"]["suffix_dc_bridge"]
    assert bridge[0] == "BOOLEAN"
    assert bridge[1]["default"] is True
    assert mixed_inputs["required"]["handoff_transfer"][0] == ["learned_3d"]
    assert sparse_inputs["required"]["handoff_transfer"][0] == ["bicubic", "learned_3d"]
''',
    '''def test_suffix_dc_bridge_is_exposed_on_all_continuum_progressive_nodes_and_defaults_on():
    target_inputs = H3ProgressiveTargetInputHandoff.INPUT_TYPES()
    sparse_inputs = H3ProgressiveTargetSparseHandoff.INPUT_TYPES()
    mixed_inputs = H3ProgressiveMixedGridHandoff.INPUT_TYPES()
    for inputs in (target_inputs, sparse_inputs, mixed_inputs):
        bridge = inputs["required"]["suffix_dc_bridge"]
        assert bridge[0] == "BOOLEAN"
        assert bridge[1]["default"] is True
    assert mixed_inputs["required"]["handoff_transfer"][0] == ["learned_3d"]
    assert sparse_inputs["required"]["handoff_transfer"][0] == ["bicubic", "learned_3d"]
''',
)

Path("tests/test_continuum_suffix_bridge.py").write_text('''from __future__ import annotations

import torch

from h3_flow_regenerate.geometry import pack_streams
from h3_flow_regenerate.runtime import _contiguous_exact_video_prefix


class _IdentityBase:
    def process_latent_in(self, value):
        return value


def _packed_mask(video_mask, audio_mask):
    return torch.cat(
        (
            video_mask.reshape(video_mask.shape[0], 1, -1),
            audio_mask.reshape(audio_mask.shape[0], 1, -1),
        ),
        dim=-1,
    )


def test_contiguous_exact_prefix_extraction_uses_model_domain_latent_and_whole_frames():
    video = torch.arange(1 * 24 * 4 * 2 * 2, dtype=torch.float32).reshape(1, 24, 4, 2, 2)
    audio = torch.zeros(1, 32, 2, 3)
    latent, shapes = pack_streams((video, audio))
    video_mask = torch.ones_like(video)
    video_mask[:, :, :2] = 0
    mask = _packed_mask(video_mask, torch.ones_like(audio))

    prefix = _contiguous_exact_video_prefix(_IdentityBase(), latent, mask, shapes)

    assert prefix is not None
    assert torch.equal(prefix, video[:, :, :2])


def test_contiguous_exact_prefix_extraction_skips_noncanonical_partial_mask():
    video = torch.zeros(1, 24, 4, 2, 2)
    audio = torch.zeros(1, 32, 2, 3)
    latent, shapes = pack_streams((video, audio))
    video_mask = torch.ones_like(video)
    video_mask[:, :, :2] = 0
    video_mask[:, :, 2, 0, 0] = 0
    mask = _packed_mask(video_mask, torch.ones_like(audio))

    assert _contiguous_exact_video_prefix(_IdentityBase(), latent, mask, shapes) is None
''')
