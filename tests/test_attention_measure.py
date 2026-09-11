import pytest
import torch

from h3_flow_regenerate import attention_measure as m


def request(**overrides):
    args = dict(
        q_rows=30,
        kv_rows=30,
        video_start=6,
        temporal=3,
        prefix_t=1,
        source_grid=(2, 3),
        prefix_grid=(3, 4),
    )
    args.update(overrides)
    return m.build_attention_measure_request(**args)


def test_schema_is_complete_normalized_and_deterministic():
    r = request()
    assert r["segments"] == [
        {"start": 0, "stop": 6, "mass_num": 1, "mass_den": 1},
        {"start": 6, "stop": 18, "mass_num": 1, "mass_den": 2},
        {"start": 18, "stop": 30, "mass_num": 1, "mass_den": 1},
    ]
    assert m.attention_measure_semantic_digest(r) == m.attention_measure_semantic_digest(
        dict(reversed(list(r.items())))
    )

    # Segment spelling is not numerical identity. Equivalent adjacent segments
    # canonicalize before hashing so receipt/history identity stays stable.
    subdivided = {
        **r,
        "segments": [
            {"start": 0, "stop": 2, "mass_num": 7, "mass_den": 7},
            {"start": 2, "stop": 6, "mass_num": 1, "mass_den": 1},
            {"start": 6, "stop": 12, "mass_num": 2, "mass_den": 4},
            {"start": 12, "stop": 18, "mass_num": 1, "mass_den": 2},
            {"start": 18, "stop": 30, "mass_num": 1, "mass_den": 1},
        ],
    }
    assert m.validate_attention_measure_request(subdivided)["segments"] == r["segments"]
    assert m.attention_measure_semantic_digest(subdivided) == m.attention_measure_semantic_digest(r)


def test_measure_equalizes_per_frame_spatial_mass_and_preserves_unit_regions():
    b = m.materialize_key_log_measure(request())
    w = b.exp()
    assert torch.equal(b[:6], torch.zeros(6, dtype=torch.float64))
    assert torch.equal(b[18:], torch.zeros(12, dtype=torch.float64))
    assert float(w[6:18].sum()) == pytest.approx(6.0)
    assert float(w[18:24].sum()) == pytest.approx(6.0)
    assert float(w[24:30].sum()) == pytest.approx(6.0)


def test_equal_grid_is_unit_measure_and_native_identity():
    r = m.build_attention_measure_request(
        q_rows=18,
        kv_rows=18,
        video_start=6,
        temporal=2,
        prefix_t=1,
        source_grid=(2, 3),
        prefix_grid=(2, 3),
    )
    assert r["segments"] == [
        {"start": 0, "stop": 18, "mass_num": 1, "mass_den": 1},
    ]
    assert m.nonunit_exact_key_ranges(r) == ()
    assert torch.equal(m.materialize_key_log_measure(r), torch.zeros(18, dtype=torch.float64))


def test_subdivision_invariance_and_post_scale_bias():
    torch.manual_seed(7)
    q = torch.randn(2, 3, 5, 4, dtype=torch.float64)
    k = torch.randn(2, 3, 30, 4, dtype=torch.float64)
    v = torch.randn(2, 3, 30, 6, dtype=torch.float64)
    bias = m.materialize_key_log_measure(request())
    dense = m.dense_weighted_attention_reference(q, k, v, key_log_measure=bias, scale=0.37)
    for chunk in (1, 2, 7, 16, 29, 64):
        streamed = m.dense_weighted_attention_reference(q, k, v, key_log_measure=bias, scale=0.37, key_chunk_size=chunk)
        torch.testing.assert_close(streamed, dense, rtol=1e-12, atol=1e-12)

    q0 = torch.zeros(1, 1, 1, 2, dtype=torch.float64)
    k0 = torch.zeros(1, 1, 30, 2, dtype=torch.float64)
    v0 = torch.eye(30, dtype=torch.float64).reshape(1, 1, 30, 30)
    out = m.dense_weighted_attention_reference(q0, k0, v0, key_log_measure=bias, scale=9.0)
    assert out[0, 0, 0, 6].item() / out[0, 0, 0, 0].item() == pytest.approx(0.5)


def test_ragged_weighted_interval_forces_every_intersecting_key_block_exact():
    r = m.build_attention_measure_request(
        q_rows=303,
        kv_rows=303,
        video_start=63,
        temporal=3,
        prefix_t=1,
        source_grid=(8, 8),
        prefix_grid=(8, 14),
    )
    assert m.nonunit_exact_key_ranges(r, block_size=64) == ((0, 192),)


def test_boolean_integer_and_nonfinite_inputs_rejected():
    r = request()
    bad = dict(r)
    bad["q_rows"] = True
    with pytest.raises(TypeError):
        m.validate_attention_measure_request(bad)
    bad = {**r, "segments": [dict(s) for s in r["segments"]]}
    bad["segments"][1]["mass_num"] = False
    with pytest.raises(TypeError):
        m.validate_attention_measure_request(bad)
    with pytest.raises(TypeError):
        m.materialize_key_log_measure(r, dtype=torch.int64)
    bias = m.materialize_key_log_measure(r)
    bias[0] = float("nan")
    q = torch.zeros(1, 1, 1, 2)
    k = torch.zeros(1, 1, 30, 2)
    v = torch.zeros(1, 1, 30, 2)
    with pytest.raises(ValueError):
        m.dense_weighted_attention_reference(q, k, v, key_log_measure=bias)


def test_masks_and_all_masked_rows_are_stable_and_chunk_invariant():
    torch.manual_seed(3)
    q = torch.randn(1, 2, 4, 8, dtype=torch.float64)
    k = torch.randn(1, 2, 30, 8, dtype=torch.float64)
    v = torch.randn(1, 2, 30, 5, dtype=torch.float64)
    bias = m.materialize_key_log_measure(request())
    mask = torch.ones(1, 1, 4, 30, dtype=torch.bool)
    mask[..., 1, :] = False
    mask[..., 2, 9:] = False
    dense = m.dense_weighted_attention_reference(q, k, v, key_log_measure=bias, mask=mask)
    streamed = m.dense_weighted_attention_reference(q, k, v, key_log_measure=bias, mask=mask, key_chunk_size=7)
    torch.testing.assert_close(streamed, dense, rtol=1e-12, atol=1e-12)
    assert torch.count_nonzero(dense[..., 1, :]) == 0

    additive = torch.zeros(1, 1, 4, 30, dtype=torch.float64)
    additive[..., 3, 10:] = float("-inf")
    dense = m.dense_weighted_attention_reference(q, k, v, key_log_measure=bias, mask=additive)
    streamed = m.dense_weighted_attention_reference(q, k, v, key_log_measure=bias, mask=additive, key_chunk_size=11)
    torch.testing.assert_close(streamed, dense, rtol=1e-12, atol=1e-12)

    bad_additive = additive.clone()
    bad_additive[..., 0, 0] = float("nan")
    with pytest.raises(ValueError):
        m.dense_weighted_attention_reference(q, k, v, key_log_measure=bias, mask=bad_additive)


def test_equal_unit_measure_matches_unweighted_softmax():
    q = torch.tensor([[[[1.0, 2.0]]]], dtype=torch.float64)
    k = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float64)
    v = torch.tensor([[[[2.0], [7.0]]]], dtype=torch.float64)
    bias = torch.zeros(2, dtype=torch.float64)
    got = m.dense_weighted_attention_reference(q, k, v, key_log_measure=bias, scale=0.5)
    scores = (q @ k.transpose(-1, -2)) * 0.5
    expected = torch.softmax(scores, dim=-1) @ v
    torch.testing.assert_close(got, expected, rtol=1e-14, atol=1e-14)
