from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate.residual_evidence import export_residual_geometry_evidence


def test_residual_evidence_rejects_sanitized_filename_collision(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "folder_paths",
        SimpleNamespace(get_output_directory=lambda: str(tmp_path)),
    )

    with pytest.raises(ValueError, match="collides with file"):
        export_residual_geometry_evidence(
            {
                "a b": torch.zeros(1, 1),
                "a-b": torch.ones(1, 1),
            },
            session_id="session",
            chunk_id="chunk",
            seed=1,
            sigma=0.5,
            metadata={},
        )

    root = tmp_path / "h3_flow_regenerate" / "residual_geometry"
    assert root.is_dir()
    assert list(root.iterdir()) == []
