from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one match, found {count}: {old[:100]!r}")
    p.write_text(text.replace(old, new, 1))


replace_once(
    "docs/TARGET_SPARSE_CONTINUUM.md",
    "## Shared suffix DC bridge\n\nTarget-Sparse exposes the same `suffix_dc_bridge` control as the other Continuum progressive nodes and defaults it to **on**. It does not alter the sparse early transformer stage. The correction is installed only on the fresh full-grid high-stage sampler lifetime, after the hidden-space lifter boundary.\n",
    "## Continuum suffix DC bridge\n\nTarget-Sparse exposes `suffix_dc_bridge` as one of the two dedicated exact-prefix Continuum nodes and defaults it to **on**. The generic **Progressive Handoff (Target Input)** node does not expose or apply this Continuum-specific seam correction. Target-Sparse does not alter the sparse early transformer stage; the correction is installed only on the fresh full-grid high-stage sampler lifetime, after the hidden-space lifter boundary.\n",
)

replace_once(
    "docs/MIXED_GRID_CONTINUUM.md",
    "## Shared suffix DC bridge\n\n`suffix_dc_bridge` defaults to **on** on all Continuum progressive nodes. The common contract is deliberately narrow: a canonical whole-frame exact prefix, one generated suffix boundary, a per-channel spatial-mean correction, and exactly one corrected suffix latent token. The authoritative prefix and every later suffix token remain unchanged. No video-space crossfade or extra H3 NFE is introduced.\n",
    "## Continuum suffix DC bridge\n\n`suffix_dc_bridge` defaults to **on** only on the two dedicated exact-prefix Continuum nodes: **Progressive Target-Sparse Continuum** and **Progressive Mixed-Grid Continuum**. The generic **Progressive Handoff (Target Input)** node does not expose or apply this Continuum-specific seam correction. The bridge contract is deliberately narrow: a canonical whole-frame exact prefix, one generated suffix boundary, a per-channel spatial-mean correction, and exactly one corrected suffix latent token. The authoritative prefix and every later suffix token remain unchanged. No video-space crossfade or extra H3 NFE is introduced.\n",
)
replace_once(
    "docs/MIXED_GRID_CONTINUUM.md",
    "The other two Continuum progressive paths expose the same control without inventing a learned-prefix surrogate. Their exact-prefix sampler already evaluates H3 on the target grid, so they calibrate from the first **actual** H3 predicted-clean boundary before native mask restoration. Forecasts are deliberately not used for calibration. Target-Sparse guarantees an actual first high-stage call; the conservative Target Input fallback records a skip if no actual model output occurs.\n",
    "Target-Sparse exposes the same Continuum-only control without inventing a learned-prefix surrogate. Its fresh full-grid high stage already evaluates H3 on the target grid, so it calibrates from the first **actual** H3 predicted-clean boundary before native mask restoration. Forecasts are deliberately not used for calibration, and Target-Sparse guarantees an actual first high-stage call.\n",
)
replace_once(
    "docs/MIXED_GRID_CONTINUUM.md",
    "`mixed_grid_transfer` retains A/B/C diagnostics for native learned-upscaler, uncorrected exact restoration, and corrected exact restoration. `mixed_grid_complete` retains D after full target-grid refinement. The matched mixed-grid media gate removed the boundary flash with the one-token weight-1.0 correction; that result justifies the default for the shared control, but the generalized Target Input and Target-Sparse placements still require their own matched decoded-media checks.\n",
    "`mixed_grid_transfer` retains A/B/C diagnostics for native learned-upscaler, uncorrected exact restoration, and corrected exact restoration. `mixed_grid_complete` retains D after full target-grid refinement. The matched Mixed-Grid media gate removed the boundary flash with the one-token weight-1.0 correction, which justifies the default for Mixed-Grid. Target-Sparse remains a separate placement and still requires its own matched decoded-media check; no inference is made for generic Target Input.\n",
)

print("Corrected Continuum seam-bridge documentation scope")
