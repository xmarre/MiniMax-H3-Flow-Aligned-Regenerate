#!/usr/bin/env python3
"""Validate the pinned MiniMax-H3-Keyless public contract consumed by Flow."""

from __future__ import annotations

import argparse
from pathlib import Path


def require(path: Path, *needles: str) -> None:
    text = path.read_text(encoding="utf-8")
    missing = [needle for needle in needles if needle not in text]
    if missing:
        raise SystemExit(f"{path}: missing required Keyless contract fragments: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyless", type=Path, required=True)
    args = parser.parse_args()

    require(
        args.keyless / "minimax_h3_keyless/contracts.py",
        'ARCHITECTURE = "h3_keyless_core50_v1"',
        'CONTRACT_KEY = "minimax_h3_keyless_contract_v1"',
        'QV_ORDER = "q_effective;v"',
        'ROPE_POLICY = "h3_split_half_96_v1"',
        "HIDDEN_SIZE = 5376",
        "HEADS = 56",
        "HEAD_DIM = 128",
        "CORE_BLOCKS = 50",
        "TOKEN_REFINER_BLOCKS = 2",
        'routing_source: str = "value"',
        'retrieval_source: str = "raw_projected_value"',
        'projection_attr: str = "qv_proj"',
        "class RoutingSpecV1:",
        "def select_value_rows(",
        "selected_v = v.index_select(0, idx)",
        "selected_rope = selected_rope.index_select(1, idx.to(selected_rope.device))",
        "selected_measure = selected_measure.index_select(0, idx.to(selected_measure.device))",
        "routing_position_domain = _compose_row_domain(",
    )
    require(
        args.keyless / "minimax_h3_keyless/model.py",
        "class KeylessMiniMaxH3Model(native_cls):",
        "replace_main_attention_modules(",
        "block.attn = cls(",
        "setattr(model, CONTRACT_KEY, KeylessContractV1())",
    )
    require(
        args.keyless / "minimax_h3_keyless/loader.py",
        "class KeylessBaseModel(comfy.model_base.MiniMaxH3):",
        "unet_model=KeylessDiffusion",
        "existing_contract = getattr(model.diffusion_model, CONTRACT_KEY)",
        "setattr(model.diffusion_model, CONTRACT_KEY, loaded_contract)",
    )
    print("pinned MiniMax-H3-Keyless contract: OK")


if __name__ == "__main__":
    main()
