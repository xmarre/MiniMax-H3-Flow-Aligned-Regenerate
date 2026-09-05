from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one match, found {count}\n--- old ---\n{old}")
    p.write_text(text.replace(old, new, 1))


replace_once(
    "tests/test_attention_reference_runtime.py",
    '            transfer_mode="learned_3d",\n'
    '            learned_upscaler=UnusedProvider(),\n'
    '        ),\n',
    '            transfer_mode="learned_3d",\n'
    '            learned_upscaler=UnusedProvider(),\n'
    '            suffix_dc_bridge=False,\n'
    '        ),\n',
)
replace_once(
    "tests/test_mixed_grid_seam_diagnostics.py",
    '    assert transfer.fields["suffix_dc_bridge_enabled"] is False\n'
    '    assert transfer.fields["suffix_dc_bridge_corrected_tokens"] == 0\n',
    '    assert transfer.fields["suffix_dc_bridge_enabled"] is True\n'
    '    assert transfer.fields["suffix_dc_bridge_corrected_tokens"] == 1\n',
)
