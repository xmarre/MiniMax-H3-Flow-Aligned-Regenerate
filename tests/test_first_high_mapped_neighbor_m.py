from __future__ import annotations

from copy import deepcopy

from h3_flow_regenerate import first_high_mapped_neighbor_diagnostic as m
from h3_flow_regenerate import first_high_sol_local_diagnostic as e


def _valid_evidence():
    items = []
    for block in range(2, 50):
        for group in range(11):
            q_rows = e._EXPECTED_LOCAL_Q_ROWS[group]
            kv_rows = e._EXPECTED_WINDOW_KV_ROWS[group]
            q_blocks = (q_rows + 63) // 64
            k_blocks = (kv_rows + 63) // 64
            original = q_blocks * 56 * min(k_blocks, 49)
            added = q_blocks * 56
            items.append(
                {
                    "kind": "mapped_neighbor_route",
                    "block_index": block,
                    "group_index": group,
                    "reason": "mapped_neighbor",
                    "q_rows": q_rows,
                    "kv_rows": kv_rows,
                    "q_blocks": q_blocks,
                    "k_blocks": k_blocks,
                    "original_sink_rows": e._EXPECTED_VIDEO_SPAN[0],
                    "mapped_neighbor_intervals": [[0, 1] for _ in range(q_blocks)],
                    "descriptor_sha256": "a" * 64,
                    "query_positions_sha256": "b" * 64,
                    "original_selected_pairs": original,
                    "added_selected_pairs": added,
                    "effective_selected_pairs": original + added,
                    "total_block_pairs": q_blocks * 56 * k_blocks,
                    "exact_work_increase_fraction": added / original,
                    "additive_only": True,
                    "restricted_domain_unchanged": True,
                }
            )
    items.append(
        {
            "kind": "sol_sparse_provenance",
            "provenance": {
                "contract": "sana-sol-engine-sol-attn-64-rect-sm120-v3",
                "source": "sana-sol-engine",
                "revision": "2936c47637380842aaa4a4488fac5006cc542b70",
                "compute_capability": [12, 0],
            },
        }
    )
    return items


def test_mapped_route_validator_accepts_only_complete_additive_528_call_contract():
    evidence = _valid_evidence()
    report = m._validate_mapped_routes(evidence)

    assert report["execution_valid"] is True
    assert report["identity_complete"] is True
    assert report["route_records"] == 528
    assert report["packaged_sparse_provenance_valid"] is True
    assert report["total_added_selected_pairs"] > 0
    assert report["total_effective_selected_pairs"] == (
        report["total_original_selected_pairs"] + report["total_added_selected_pairs"]
    )

    missing = evidence[1:]
    assert m._validate_mapped_routes(missing)["execution_valid"] is False

    non_additive = deepcopy(evidence)
    non_additive[0]["additive_only"] = False
    assert m._validate_mapped_routes(non_additive)["execution_valid"] is False


def test_mapped_route_validator_rejects_domain_expansion_and_wrong_provenance():
    expanded = _valid_evidence()
    expanded[0]["restricted_domain_unchanged"] = False
    assert m._validate_mapped_routes(expanded)["execution_valid"] is False

    wrong_provenance = _valid_evidence()
    wrong_provenance[-1]["provenance"]["revision"] = "0" * 40
    assert m._validate_mapped_routes(wrong_provenance)["execution_valid"] is False
