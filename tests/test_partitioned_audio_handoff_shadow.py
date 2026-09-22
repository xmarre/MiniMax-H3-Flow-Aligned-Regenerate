from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate.geometry import pack_streams, unpack_streams
from h3_flow_regenerate.partitioned_diagnostics import (
    PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_MODEL_TIMESTEP,
    PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER_EXACT_TIMESTEP,
    PARTITIONED_AUDIO_HANDOFF_SOURCE_MAIN,
    PARTITIONED_AUDIO_HANDOFF_SOURCE_SHADOW,
    PARTITIONED_AUDIO_POSITION_DOMAIN_KEY,
    PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE,
    PARTITIONED_AV_HANDOFF_SOURCE_SHADOW,
    PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT,
    PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_KEY,
    PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE,
    PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
)
from h3_flow_regenerate.partitioned_scheduler import (
    PartitionedPreflightUnsupported,
    _select_source_uniform_shadow_clean_video,
    _source_uniform_audio_shadow_controls,
    _source_uniform_audio_shadow_sampler_contract,
    _source_uniform_av_shadow_stage_contract,
    _splice_source_uniform_shadow_audio_state,
    _validate_audio_handoff_shadow_configuration,
    _validate_av_handoff_shadow_configuration,
)
from h3_flow_regenerate.partitioned_stage import PARTITIONED_STAGE_KEY
from h3_flow_regenerate.runtime import FLOW_STAGE_KEY


class _Metrics:
    def __init__(self):
        self.events = []

    def event(self, kind, **fields):
        self.events.append((kind, fields))


def _valid_shadow_configuration(**overrides):
    values = {
        "audio_handoff_source": PARTITIONED_AUDIO_HANDOFF_SOURCE_SHADOW,
        "prefix_transformer_context": PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT,
        "vdn_linear_diagnostic": PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
        "audio_position_domain": PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE,
        "audio_guided_overlap_mode": PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_MODEL_TIMESTEP,
        "audio_guided_overlap_ticks": 4,
    }
    values.update(overrides)
    return values


def test_audio_handoff_shadow_configuration_is_strictly_one_axis():
    _validate_audio_handoff_shadow_configuration(**_valid_shadow_configuration())
    _validate_audio_handoff_shadow_configuration(
        PARTITIONED_AUDIO_HANDOFF_SOURCE_MAIN,
        prefix_transformer_context="anything",
        vdn_linear_diagnostic="anything",
        audio_position_domain="anything",
        audio_guided_overlap_mode="anything",
        audio_guided_overlap_ticks=0,
    )

    bad_cases = (
        {"prefix_transformer_context": PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE},
        {"vdn_linear_diagnostic": "bypass_partitioned_linear"},
        {"audio_position_domain": "legacy_target"},
        {"audio_guided_overlap_mode": "sampler_mask"},
        {"audio_guided_overlap_ticks": 3},
    )
    for override in bad_cases:
        with pytest.raises(PartitionedPreflightUnsupported, match="audio handoff"):
            _validate_audio_handoff_shadow_configuration(**_valid_shadow_configuration(**override))


def test_shadow_control_scope_restores_main_candidate_options_after_exception():
    transformer = {
        PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_KEY: PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT,
        PARTITIONED_AUDIO_POSITION_DOMAIN_KEY: PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE,
        "keep": "value",
    }
    with (
        pytest.raises(RuntimeError, match="sentinel"),
        _source_uniform_audio_shadow_controls(transformer),
    ):
        assert transformer[PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_KEY] == PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE
        assert PARTITIONED_AUDIO_POSITION_DOMAIN_KEY not in transformer
        assert transformer["keep"] == "value"
        raise RuntimeError("sentinel")

    assert transformer[PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_KEY] == PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT
    assert transformer[PARTITIONED_AUDIO_POSITION_DOMAIN_KEY] == PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE
    assert transformer["keep"] == "value"


def test_shadow_sampler_contract_keeps_flow_stage_unset_for_quiescent_vdn_release():
    guider = SimpleNamespace(model_options={"transformer_options": {}})
    transformer = guider.model_options["transformer_options"]
    metrics = _Metrics()
    plan = object()

    with _source_uniform_audio_shadow_sampler_contract(guider, plan, metrics):
        assert FLOW_STAGE_KEY not in transformer
        assert PARTITIONED_STAGE_KEY in transformer

    assert FLOW_STAGE_KEY not in transformer
    assert PARTITIONED_STAGE_KEY not in transformer


def test_shadow_sampler_contract_rejects_nested_flow_stage():
    guider = SimpleNamespace(model_options={"transformer_options": {FLOW_STAGE_KEY: "low"}})
    with (
        pytest.raises(RuntimeError, match="quiescent Flow stage boundary"),
        _source_uniform_audio_shadow_sampler_contract(guider, object(), _Metrics()),
    ):
        pass


def test_shadow_audio_splice_preserves_main_video_and_exact_prefix():
    main_video = torch.arange(1 * 24 * 2 * 2 * 2, dtype=torch.float32).reshape(1, 24, 2, 2, 2)
    shadow_video = main_video + 1000
    main_audio = torch.arange(1 * 32 * 2 * 6, dtype=torch.float32).reshape(1, 32, 2, 6)
    shadow_audio = main_audio.clone()
    shadow_audio[..., 2:] += 7

    main, shapes = pack_streams((main_video, main_audio))
    shadow, shadow_shapes = pack_streams((shadow_video, shadow_audio))
    assert shadow_shapes == shapes
    video_mask = torch.ones_like(main_video)
    audio_mask = torch.ones_like(main_audio)
    audio_mask[..., :2] = 0
    mask = pack_streams((video_mask, audio_mask))[0]
    metrics = _Metrics()

    hybrid = _splice_source_uniform_shadow_audio_state(main, shadow, list(shapes), mask, metrics)
    hybrid_video, hybrid_audio = unpack_streams(hybrid, list(shapes))

    assert torch.equal(hybrid_video, main_video)
    assert torch.equal(hybrid_audio, shadow_audio)
    kind, fields = metrics.events[-1]
    assert kind == "partitioned_audio_handoff_shadow_splice"
    assert fields["shadow_video_discarded"] is True
    assert fields["main_video_preserved"] is True
    assert fields["protected_audio_prefix_exact"] is True
    assert fields["generated_audio_changed_elements"] > 0


def test_shadow_audio_splice_rejects_protected_prefix_mutation_or_noop():
    video = torch.zeros(1, 24, 2, 2, 2)
    audio = torch.zeros(1, 32, 2, 6)
    main, shapes = pack_streams((video, audio))
    video_mask = torch.ones_like(video)
    audio_mask = torch.ones_like(audio)
    audio_mask[..., :2] = 0
    mask = pack_streams((video_mask, audio_mask))[0]

    bad_audio = audio.clone()
    bad_audio[..., :2] = 1
    bad, _ = pack_streams((video, bad_audio))
    with pytest.raises(RuntimeError, match="protected audio prefix"):
        _splice_source_uniform_shadow_audio_state(main, bad, list(shapes), mask, _Metrics())

    with pytest.raises(RuntimeError, match="no distinct generated audio state"):
        _splice_source_uniform_shadow_audio_state(main, main.clone(), list(shapes), mask, _Metrics())


def test_av_handoff_shadow_configuration_is_strictly_one_axis():
    for audio_mode in (
        PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_MODEL_TIMESTEP,
        PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_SAMPLER_EXACT_TIMESTEP,
    ):
        _validate_av_handoff_shadow_configuration(
            PARTITIONED_AV_HANDOFF_SOURCE_SHADOW,
            audio_handoff_source=PARTITIONED_AUDIO_HANDOFF_SOURCE_MAIN,
            prefix_transformer_context=PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT,
            vdn_linear_diagnostic=PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
            audio_position_domain=PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE,
            audio_guided_overlap_mode=audio_mode,
            audio_guided_overlap_ticks=4,
        )
    bad = (
        {"audio_handoff_source": PARTITIONED_AUDIO_HANDOFF_SOURCE_SHADOW},
        {"prefix_transformer_context": PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_SOURCE},
        {"vdn_linear_diagnostic": "bypass_partitioned_linear"},
        {"audio_position_domain": "legacy_target"},
        {"audio_guided_overlap_mode": "sampler_mask"},
        {"audio_guided_overlap_ticks": 3},
    )
    base = {
        "audio_handoff_source": PARTITIONED_AUDIO_HANDOFF_SOURCE_MAIN,
        "prefix_transformer_context": PARTITIONED_PREFIX_TRANSFORMER_CONTEXT_EXACT,
        "vdn_linear_diagnostic": PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL,
        "audio_position_domain": PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE,
        "audio_guided_overlap_mode": PARTITIONED_AUDIO_GUIDED_OVERLAP_MODE_MODEL_TIMESTEP,
        "audio_guided_overlap_ticks": 4,
    }
    for override in bad:
        values = {**base, **override}
        with pytest.raises(PartitionedPreflightUnsupported, match="AV handoff"):
            _validate_av_handoff_shadow_configuration(
                PARTITIONED_AV_HANDOFF_SOURCE_SHADOW,
                **values,
            )


def test_av_shadow_stage_contract_publishes_low_then_probe():
    guider = SimpleNamespace(model_options={"transformer_options": {}})
    transformer = guider.model_options["transformer_options"]
    metrics = _Metrics()
    with _source_uniform_av_shadow_stage_contract(guider, object(), metrics, "low"):
        assert transformer[FLOW_STAGE_KEY] == "low"
        assert PARTITIONED_STAGE_KEY in transformer
        assert "h3_refinement" not in transformer
    assert FLOW_STAGE_KEY not in transformer
    assert PARTITIONED_STAGE_KEY not in transformer

    with _source_uniform_av_shadow_stage_contract(guider, object(), metrics, "probe"):
        assert transformer[FLOW_STAGE_KEY] == "probe"
        assert PARTITIONED_STAGE_KEY in transformer
        assert transformer["h3_refinement"]["active"] is True
    assert FLOW_STAGE_KEY not in transformer
    assert PARTITIONED_STAGE_KEY not in transformer
    assert "h3_refinement" not in transformer


def test_av_shadow_clean_video_selection_preserves_exact_prefix_and_main_clean_audio():
    main_video = torch.arange(1 * 24 * 4 * 2 * 2, dtype=torch.float32).reshape(1, 24, 4, 2, 2)
    shadow_video = main_video.clone()
    shadow_video[:, :, 2:] += 9
    main_audio = torch.arange(1 * 32 * 2 * 6, dtype=torch.float32).reshape(1, 32, 2, 6)
    shadow_audio = main_audio + 100
    main, shapes = pack_streams((main_video, main_audio))
    shadow, shadow_shapes = pack_streams((shadow_video, shadow_audio))
    assert shadow_shapes == shapes
    exact_prefix = torch.full_like(main_video[:, :, :2], -3)
    metrics = _Metrics()

    selected = _select_source_uniform_shadow_clean_video(
        main,
        shadow,
        list(shapes),
        prefix_t=2,
        exact_prefix_source=exact_prefix,
        metrics=metrics,
    )
    selected_video, selected_audio = unpack_streams(selected, list(shapes))
    assert torch.equal(selected_video[:, :, :2], exact_prefix)
    assert torch.equal(selected_video[:, :, 2:], shadow_video[:, :, 2:])
    assert torch.equal(selected_audio, main_audio)
    kind, fields = metrics.events[-1]
    assert kind == "partitioned_av_handoff_shadow_clean_video"
    assert fields["exact_source_prefix_restored"] is True
    assert fields["main_clean_audio_preserved"] is True
    assert fields["shadow_clean_audio_discarded"] is True
    assert fields["generated_video_changed_elements"] > 0


def test_av_shadow_clean_video_selection_rejects_noop_suffix():
    video = torch.zeros(1, 24, 4, 2, 2)
    audio = torch.zeros(1, 32, 2, 6)
    main, shapes = pack_streams((video, audio))
    with pytest.raises(RuntimeError, match="no distinct generated clean-video state"):
        _select_source_uniform_shadow_clean_video(
            main,
            main.clone(),
            list(shapes),
            prefix_t=2,
            exact_prefix_source=video[:, :, :2],
            metrics=_Metrics(),
        )
