"""Target-band native carry through the real uniform-source scheduler and Core H3 blocks."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import test_partitioned_target_band_e2e as fixture
import torch

from h3_flow_regenerate.geometry import resize_spatial_5d, unpack_streams
from h3_flow_regenerate.source_prefix_carry import (
    SOURCE_PREFIX_CARRY_KEY,
    SOURCE_PREFIX_TARGET_BAND_POLICY,
    SourcePrefixCarry,
    reconcile_target_band,
)

PROTECTED_T = 12
CHUNK_2 = {"h3_continuum": {"active": True, "api": 1, "context_frames": 39, "chunk_index": 2}}


@pytest.fixture(params=["sol", "native"])
def backend(request, monkeypatch):
    monkeypatch.setattr(fixture, "PROTECTED_T", PROTECTED_T)
    monkeypatch.setattr(fixture, "TEMPORAL", 22)
    if request.param == "native":
        owner = fixture._vdn_owner

        def native_owner():
            patched = owner()
            patched._vdn_partitioned_attention_provider_api = 1
            return patched

        monkeypatch.setattr(fixture, "_vdn_owner", native_owner)
    return request.param


def _run(monkeypatch, backend, owner=None, **kwargs):
    transformer = dict(CHUNK_2) if owner is not None else {}
    if backend == "native":
        transformer[fixture.SOL_RUNTIME_KEY] = None
    return fixture._harness(
        monkeypatch,
        spatial_stage_control=fixture.PARTITIONED_SPATIAL_STAGE_UNIFORM_SOURCE,
        protected_audio_ticks=4,
        extra_model_options={SOURCE_PREFIX_CARRY_KEY: owner} if owner is not None else None,
        extra_transformer_options=transformer,
        **kwargs,
    )


def _primed(baseline, *, target_band):
    """Prior pair whose source rendering lacks a background object the target shows."""
    exact, _ = unpack_streams(baseline.latent_image, baseline.shapes)
    prior_target = exact.clone()
    prior_target[:, :, -PROTECTED_T:] = exact[:, :, :PROTECTED_T]
    source = resize_spatial_5d(prior_target, *fixture.SOURCE_HW)
    h, w = fixture.SOURCE_HW
    yy, xx = torch.meshgrid(torch.arange(h).float(), torch.arange(w).float(), indexing="ij")
    missing_object = 1.5 * torch.exp(-((yy - h / 3).square() + (xx - w / 2).square()) / 8.0)
    texture = 0.2 * (((yy + xx) % 2) * 2 - 1)
    source = source - missing_object + texture + 0.125
    owner = SourcePrefixCarry(target_band=target_band)
    with owner.transaction(object()):
        owner.begin_initial()
        owner.stage_source(source)
        owner.prepare_success(prior_target)
    return owner, source


def test_target_band_reaches_generated_suffix_with_exact_prefix_and_audio(monkeypatch, backend):
    baseline = _run(monkeypatch, backend)
    plain_owner, source = _primed(baseline, target_band=False)
    band_owner, _ = _primed(baseline, target_band=True)
    plain = _run(monkeypatch, backend, plain_owner)
    band = _run(monkeypatch, backend, band_owner)

    assert [call["stage"] for call in band.calls] == ["low", "probe", "high"]
    for a, b in zip(plain.calls, band.calls, strict=True):
        assert torch.equal(a["mask"], b["mask"])
        assert torch.equal(a["sigmas"], b["sigmas"])
        if a["stage"] == "low":
            assert torch.equal(a["noise"], b["noise"])

    generic, _ = unpack_streams(baseline.calls[0]["latent"], baseline.calls[0]["shapes"])
    plain_low, plain_audio = unpack_streams(plain.calls[0]["latent"], plain.calls[0]["shapes"])
    band_low, band_audio = unpack_streams(band.calls[0]["latent"], band.calls[0]["shapes"])
    carry = source[:, :, -PROTECTED_T:]
    assert torch.equal(plain_low[:, :, :PROTECTED_T], carry)
    expected, _ = reconcile_target_band(carry, generic[:, :, :PROTECTED_T])
    assert torch.equal(band_low[:, :, :PROTECTED_T], expected)
    assert torch.equal(band_low[:, :, PROTECTED_T:], plain_low[:, :, PROTECTED_T:])
    assert torch.equal(band_audio, plain_audio)

    # The reconciled prefix moves toward the rendered target and keeps the carry's tone.
    target_prefix = generic[:, :, :PROTECTED_T]
    centered = lambda v: v - v.mean((-1, -2), keepdim=True)  # noqa: E731
    lowpass = lambda v: torch.nn.functional.avg_pool2d(  # noqa: E731
        centered(v).flatten(0, 2), 3, stride=1, padding=1, count_include_pad=False
    )
    assert float((lowpass(band_low[:, :, :PROTECTED_T]) - lowpass(target_prefix)).abs().mean()) < 0.5 * float(
        (lowpass(carry) - lowpass(target_prefix)).abs().mean()
    )
    assert torch.allclose(band_low[:, :, :PROTECTED_T].mean((-1, -2)), carry.mean((-1, -2)), atol=1e-5)

    # Generated low/probe tokens observe the changed prefix through Core attention.
    # The random-weight fixture is deterministic, so any suffix change is caused
    # by the prefix; its magnitude says nothing about trained-model strength.
    plain_source = plain.upscaler.inputs[0]
    band_source = band.upscaler.inputs[0]
    assert torch.equal(band_source[:, :, :PROTECTED_T], expected)
    assert not torch.equal(band_source[:, :, PROTECTED_T:], plain_source[:, :, PROTECTED_T:])
    repeat_owner, _ = _primed(baseline, target_band=False)
    assert torch.equal(_run(monkeypatch, backend, repeat_owner).upscaler.inputs[0], plain_source)

    exact, exact_audio = unpack_streams(band.latent_image, band.shapes)
    for run in (plain, band):
        final, final_audio = unpack_streams(run.result, run.shapes)
        assert torch.equal(final[:, :, :PROTECTED_T], exact[:, :, :PROTECTED_T])
        assert torch.equal(final_audio[:, :, :, :4], exact_audio[:, :, :, :4])
        assert bool(torch.isfinite(run.result).all())

    projection = fixture._events(band.metrics, "partitioned_source_prefix_projection")
    assert len(projection) == 1
    assert projection[0]["policy"] == SOURCE_PREFIX_TARGET_BAND_POLICY
    assert projection[0]["carry_policy"] == "native_source_carry_v1"
    assert projection[0]["target_band"]["carry_to_target_band_rms"] > 0.0
    assert projection[0]["extra_vae_decode_calls"] == projection[0]["extra_vae_encode_calls"] == 0
    resample = fixture._events(band.metrics, "partitioned_prefix_source_resample")
    assert resample[0]["policy"] == SOURCE_PREFIX_TARGET_BAND_POLICY

    # Successive chunks keep exact suffix pairing with the reconciled run's own source.
    final, _ = unpack_streams(band.result, band.shapes)
    following = SimpleNamespace(
        model_options={"transformer_options": {"h3_continuum": {**CHUNK_2["h3_continuum"], "chunk_index": 3}}}
    )
    with band_owner.transaction(following):
        carried, receipt = band_owner.project(final[:, :, -PROTECTED_T:], *fixture.SOURCE_HW)
    assert receipt["generation"] == 2 and receipt["previous_sequence"] == (None, 2)
    assert torch.equal(carried, band_source[:, :, -PROTECTED_T:])


def test_target_band_failure_stops_before_sampling_and_keeps_pair_for_retry(monkeypatch, backend):
    from h3_flow_regenerate import partitioned_scheduler as scheduler

    baseline = _run(monkeypatch, backend)
    owner, source = _primed(baseline, target_band=True)

    def fail(*_args, **_kwargs):
        raise ValueError("reconciliation failure")

    with monkeypatch.context() as patch:
        patch.setattr(scheduler, "reconcile_target_band", fail)
        patch.setattr(scheduler, "_begin_capture", lambda *_a, **_k: pytest.fail("entered capture"))
        with pytest.raises(ValueError, match="reconciliation failure"):
            _run(monkeypatch, backend, owner)
    retried = _run(monkeypatch, backend, owner)
    low, _ = unpack_streams(retried.calls[0]["latent"], retried.calls[0]["shapes"])
    generic, _ = unpack_streams(baseline.calls[0]["latent"], baseline.calls[0]["shapes"])
    expected, _ = reconcile_target_band(source[:, :, -PROTECTED_T:], generic[:, :, :PROTECTED_T])
    assert torch.equal(low[:, :, :PROTECTED_T], expected)


def test_target_band_boundary_capture_replays_saved_reconciled_prefix(monkeypatch, tmp_path):
    from h3_flow_regenerate.local_boundary_audit import load_replay_operands

    monkeypatch.setattr(fixture, "PROTECTED_T", PROTECTED_T)
    monkeypatch.setattr(fixture, "TEMPORAL", 22)
    folder_paths = pytest.importorskip("folder_paths")
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    baseline = _run(monkeypatch, "sol")
    owner, _ = _primed(baseline, target_band=True)
    _run(
        monkeypatch,
        "sol",
        owner,
        residual_mode="measure",
        witness_directory=str(tmp_path),
        frame_gauge_repair=False,
    )
    manifests = list(tmp_path.rglob("manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["metadata"]["source_prefix_projection_policy"] == SOURCE_PREFIX_TARGET_BAND_POLICY
    _, stages, identity = load_replay_operands(str(manifests[0]), 175, include_source=True)
    assert stages["source_grid"].shape[-2:] == fixture.SOURCE_HW
    assert identity["source_prefix_projection_policy"] == SOURCE_PREFIX_TARGET_BAND_POLICY


def test_plain_native_carry_bytes_are_unchanged_by_target_band_support(monkeypatch):
    monkeypatch.setattr(fixture, "PROTECTED_T", PROTECTED_T)
    monkeypatch.setattr(fixture, "TEMPORAL", 22)
    baseline = _run(monkeypatch, "sol")
    owner, source = _primed(baseline, target_band=False)
    run = _run(monkeypatch, "sol", owner)
    low, _ = unpack_streams(run.calls[0]["latent"], run.calls[0]["shapes"])
    assert torch.equal(low[:, :, :PROTECTED_T], source[:, :, -PROTECTED_T:])
    projection = fixture._events(run.metrics, "partitioned_source_prefix_projection")[0]
    assert projection["policy"] == "native_source_carry_v1"
    assert "target_band" not in projection
