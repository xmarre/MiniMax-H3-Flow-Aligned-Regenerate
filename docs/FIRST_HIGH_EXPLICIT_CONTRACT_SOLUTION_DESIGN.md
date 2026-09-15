# First-high explicit-contract solution design

Status: design-only investigation checkpoint, 2026-09-15. Not implementation authorization.

## Established evidence

The dedicated `replay_report_00001.json` reports attribution_valid=true, exact first-high sampler/H3 video and audio inputs, exact raw H3 output and pre-guidance video, with 3 logical / 2 actual / 1 forecast and zero upscaler calls. Retained low/probe/reload lifecycle state is weakened for the first-high defect; final-high divergence is separate.

The matching 00442 log `Pasted text(20260914-030325).txt` has now been recovered. First high reports phase=dense, seq_len=56349, routes vdn_anchor_native/vdn_dense_warmup/vdn_global_native, count=700. Untwist was active. This weakens missing first-high dense warmup as a sufficient explanation, without claiming a matched R attention counterfactual.

R first-high routes: 50 vdn_global_native, 22 vdn_dense_warmup, 100 vdn_anchor_native, 528 vdn_local_sol. VDN window/gating and adapters remain explicit model behavior in both dense-local and SOL-local routes.

## Preservation and pending work

Preserve #33 independently, #35/#36 contracts, #37 separately, #41 diagnostic-only; #39/#40 stay closed. Branch is based on main 970396db839ae7ab431b9718859f6d48a2e5019b. No existing PR branch is changed.

Pending: exact source-hash reconstruction, first-high representation/ownership audit, bounded causal experiment selection, validation and implementation handoff. No production correction has been established.

Preserve `/home/toor/ComfyUI/output/h3_flow_replay/h3_same_state_replay_234ed062128e43ed8d5ec63e27517b22.json` and its referenced tensor payload. Workstation bytes have not been accessed.
