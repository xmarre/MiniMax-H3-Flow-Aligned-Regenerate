import importlib.util
import os
from pathlib import Path

import pytest
import torch

from h3_flow_regenerate.geometry import resize_spatial_5d_h3_patch_lattice
from h3_flow_regenerate.transfer_lattice import H3_ROPE_BOX_TRANSFER_LATTICE, H3_TRANSFER_LATTICE


@pytest.mark.parametrize(
    "source,target",
    [
        ((46, 40), (66, 58)),
        ((66, 58), (46, 40)),
        ((36, 54), (50, 76)),
        ((50, 76), (36, 54)),
        ((54, 36), (76, 50)),
        ((76, 50), (54, 36)),
    ],
)
def test_upscaler_and_authoritative_prefix_use_identical_coordinate_transport(source, target):
    root = os.environ.get("H3_UPSCALER_PATH")
    if not root:
        pytest.skip("pinned upscaler source runs in source-contract CI with H3_UPSCALER_PATH")
    spec = importlib.util.spec_from_file_location(
        "h3_transfer_lattice_contract", Path(root) / "nodes/h3_patch_lattice.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = torch.randn(2, 3, 5, *source, generator=torch.Generator().manual_seed(815))
    assert module.H3_PATCH_LATTICE == H3_TRANSFER_LATTICE
    expected = resize_spatial_5d_h3_patch_lattice(value, *target)
    actual = module.resize_h3_patch_lattice(value, *target)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
    # This paired source contract pins the provider that advertises both maps.
    # Runtime compatibility with a legacy provider's default map is tested separately.
    assert module.H3_ROPE_BOX_LATTICE == H3_ROPE_BOX_TRANSFER_LATTICE
    expected = resize_spatial_5d_h3_patch_lattice(value, *target, lattice=H3_ROPE_BOX_TRANSFER_LATTICE)
    actual = module.resize_h3_patch_lattice(value, *target, lattice=H3_ROPE_BOX_TRANSFER_LATTICE)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
