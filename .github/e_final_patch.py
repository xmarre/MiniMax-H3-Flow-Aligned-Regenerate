import hashlib
import json
from pathlib import Path

source = Path("h3_flow_regenerate/first_high_sol_local_diagnostic.py")
text = source.read_text(encoding="utf-8")
old = '''                "route_mismatch_examples": item.get("route_mismatch_examples"),
                "frozen_route_reference": frozen,
'''
new = '''                "route_mismatch_examples": item.get("route_mismatch_examples"),
                "route_margin_summary": item.get("route_margin_summary"),
                "frozen_route_reference": frozen,
'''
if old not in text:
    raise SystemExit("route-margin report anchor not found")
text = text.replace(old, new, 1)
source.write_text(text, encoding="utf-8")

manifest_path = Path("h3_flow_regenerate/first_high_sol_local_e_source_delta.json")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
for entry in manifest["entries"]:
    if (
        entry.get("owner") == "sol"
        and entry.get("module") == "sol_h3.first_high_sol_local_diagnostic"
        and entry.get("relative_path", ".") == "."
    ):
        entry["candidate_git_blob_sha"] = "79b3013aa936a836fc4e5c5cf3e80f8ac06a49ac"
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
