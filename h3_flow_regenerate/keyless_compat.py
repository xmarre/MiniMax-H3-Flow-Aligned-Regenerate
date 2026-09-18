from __future__ import annotations

from typing import Any

KEYLESS_CONTRACT_KEY = "minimax_h3_keyless_contract_v1"
KEYLESS_ARCHITECTURE = "h3_keyless_core50_v1"
KEYLESS_CORE_BLOCKS = 50
KEYLESS_TOKEN_REFINER_BLOCKS = 2
KEYLESS_HEADS = 56
KEYLESS_HEAD_DIM = 128
KEYLESS_INNER_DIM = 7_168
KEYLESS_HIDDEN_SIZE = 5_376
KEYLESS_QV_ROWS = 14_336
KEYLESS_NORM_EPSILON = 1e-5
KEYLESS_ROPE_POLICY = "h3_split_half_96_v1"
KEYLESS_QV_ORDER = "q_effective;v"


class KeylessCompatibilityError(TypeError):
    """A model advertises Keyless H3 but does not satisfy Flow's v1 boundary."""


def _field(contract: Any, name: str) -> Any:
    if not hasattr(contract, name):
        raise KeylessCompatibilityError(f"{KEYLESS_CONTRACT_KEY} is missing required field {name!r}")
    return getattr(contract, name)


def validate_keyless_contract(inner: Any) -> Any | None:
    """Return the exact public Keyless v1 contract, or None for ordinary H3.

    Flow remains model-opaque for its production trajectory/progressive paths. The
    validator exists to prevent a model that explicitly advertises Keyless semantics
    from silently falling back to native-QKV assumptions in optional compatibility
    paths. It is deliberately duck-typed and never imports the Keyless package.
    """

    if inner is None or not hasattr(inner, KEYLESS_CONTRACT_KEY):
        return None
    contract = getattr(inner, KEYLESS_CONTRACT_KEY)
    expected = {
        "api": 1,
        "architecture": KEYLESS_ARCHITECTURE,
        "core_blocks": KEYLESS_CORE_BLOCKS,
        "token_refiner": "native_qkv",
        "token_refiner_blocks": KEYLESS_TOKEN_REFINER_BLOCKS,
        "heads": KEYLESS_HEADS,
        "head_dim": KEYLESS_HEAD_DIM,
        "inner_dim": KEYLESS_INNER_DIM,
        "hidden_size": KEYLESS_HIDDEN_SIZE,
        "routing_source": "value",
        "retrieval_source": "raw_projected_value",
        "routing_norm": "rmsnorm",
        "routing_norm_epsilon": KEYLESS_NORM_EPSILON,
        "rope_policy": KEYLESS_ROPE_POLICY,
        "qv_order": KEYLESS_QV_ORDER,
        "projection_attr": "qv_proj",
        "checkpoint_format_version": 1,
    }
    mismatches = []
    for name, wanted in expected.items():
        actual = _field(contract, name)
        if actual != wanted:
            mismatches.append(f"{name}={actual!r} (expected {wanted!r})")
    if mismatches:
        raise KeylessCompatibilityError(f"unsupported {KEYLESS_CONTRACT_KEY}: " + ", ".join(mismatches))

    blocks = getattr(inner, "blocks", None)
    refiners = getattr(getattr(inner, "token_refiner", None), "blocks", None)
    try:
        block_count = len(blocks)
        refiner_count = len(refiners)
    except TypeError as exc:
        raise KeylessCompatibilityError("Keyless H3 block topology is not sized") from exc
    if block_count != KEYLESS_CORE_BLOCKS or refiner_count != KEYLESS_TOKEN_REFINER_BLOCKS:
        raise KeylessCompatibilityError("Keyless H3 must expose 50 core blocks and two native-QKV token-refiner blocks")

    for index, block in enumerate(blocks):
        attention = getattr(block, "attn", None)
        if attention is None:
            raise KeylessCompatibilityError(f"Keyless block {index} has no attention module")
        if hasattr(attention, "qkv_proj"):
            raise KeylessCompatibilityError(
                f"Keyless block {index} exposes qkv_proj; Flow will not accept a fake/dead K path"
            )
        projection = getattr(attention, "qv_proj", None)
        weight = getattr(projection, "weight", None)
        shape = tuple(int(value) for value in getattr(weight, "shape", ()))
        if shape != (KEYLESS_QV_ROWS, KEYLESS_HIDDEN_SIZE):
            raise KeylessCompatibilityError(f"Keyless block {index} does not expose canonical qv_proj.weight geometry")
        if not hasattr(attention, "q_norm") or not hasattr(attention, "route_norm"):
            raise KeylessCompatibilityError(f"Keyless block {index} is missing q_norm/route_norm routing semantics")
        if int(getattr(attention, "head_dim", KEYLESS_HEAD_DIM)) != KEYLESS_HEAD_DIM:
            raise KeylessCompatibilityError(f"Keyless block {index} attention head_dim disagrees with the v1 contract")

    identity_fn = getattr(contract, "identity", None)
    if not callable(identity_fn):
        raise KeylessCompatibilityError(f"{KEYLESS_CONTRACT_KEY} must expose callable identity()")
    try:
        identity = tuple(identity_fn())
        hash(identity)
    except (TypeError, ValueError) as exc:
        raise KeylessCompatibilityError(
            f"{KEYLESS_CONTRACT_KEY}.identity() must return a hashable tuple-like value"
        ) from exc
    if not identity:
        raise KeylessCompatibilityError(f"{KEYLESS_CONTRACT_KEY}.identity() may not be empty")
    return contract


def keyless_semantic_identity(inner: Any) -> tuple[Any, ...] | None:
    contract = validate_keyless_contract(inner)
    if contract is None:
        return None
    return (KEYLESS_CONTRACT_KEY, *tuple(contract.identity()))


__all__ = [
    "KEYLESS_CONTRACT_KEY",
    "KeylessCompatibilityError",
    "keyless_semantic_identity",
    "validate_keyless_contract",
]
