from pathlib import Path
import subprocess

OLD_PIN = "5256edceabf651bdd9094c224e1907b2f0edd941"
NEW_PIN = "bdc670e5926bcefbe4022e17fe8b171fbfcf15de"

tracked = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
pin_files = []
for name in tracked:
    p = Path(name)
    try:
        text = p.read_text()
    except (UnicodeDecodeError, IsADirectoryError):
        continue
    if OLD_PIN in text:
        p.write_text(text.replace(OLD_PIN, NEW_PIN))
        pin_files.append(name)
assert pin_files, "old provider pin not found"

p = Path("README.md")
text = p.read_text()
start = text.index("`handoff_transfer=bicubic` is the default and preserves the released behavior.")
end = text.index("\nUse a complete 1-to-0 H3 sigma schedule.", start)
learned = """`handoff_transfer=bicubic` remains the compatibility default. For the validated learned path,
install the companion latent upscaler, create **MiniMax H3 Latent Upscaler Provider (3D) [Experimental]**,
connect its `H3_LATENT_UPSCALER` output to the Target Input node, and select
`handoff_transfer=learned_3d`. The learned CNN replaces only the exact-probe clean-video spatial
transfer. Conditional re-noising, deterministic target noise, sampler/Spectrum history boundaries,
the mandatory first high-grid actual call, caller audio, masks, and all H3 model-call semantics remain
unchanged. One learned inference is added per physical chunk and no extra H3 NFE is added.

Decoded-media validation now covers substantially more aggressive transitions than the original
46×46→56×56 D14 plan. Around a 0.995 MP target, bicubic at roughly a 0.55 MP private source produced
visible body/spatial handoff artifacts in the tested difficult prompt; replacing only that boundary with
`learned_3d` fixed the majority of those artifacts. At the same target, `source_scale=0.70` resolved to
800×608→1152×864 and was judged excellent, while `0.65` resolved to 736×576→1152×864 and began losing
reference likeness / tonal stability. The final higher-resolution gate kept `source_scale=0.70` and
resolved 832×640→1184×896 (~1.061 MP target); the generated action was different but the result was
again judged very good. Its two BF16 CUDA learned calls took about 0.60 s and 0.77 s and added zero H3
NFEs. These latest learned-transfer media runs used `direction+acceleration`; they are not evidence for
promoting acceleration over direction-only.

For roughly 1 MP targets, `0.70` is therefore the current tested quality/compute sweet spot for this
prompt, not a universal optimum. `0.65` is below the current useful quality floor in this case, while
higher source scales remain the safer choice when preserving source-grid fidelity matters more than
compute. Keep bicubic available for compatibility and matched controls rather than silently changing
existing workflows. Provider mode fails instead of silently falling back when its configured inference
device is unavailable.
"""
text = text[:start] + learned + text[end:]

marker = (
    "The actual source dimensions are snapped to valid H3 latent geometry, so these values are starting points rather than exact pixel guarantees. "
    "Check the `handoff_plan` metrics event for the resolved source/target latent dimensions. With fixed handoff, increasing MP alone does not require changing "
    "`handoff_coordinate`; keep the tested 0.35 initially and adjust `source_scale` separately to control the low-stage compute budget."
)
assert marker in text
addition = marker + (
    "\n\nThe table is a geometry heuristic, not a quality guarantee. At approximately 1 MP, real decoded media showed that `source_scale=0.70` can work very well with `learned_3d`, while `0.65` crossed the tested prompt's fidelity/tonal floor. Because H3 snaps both axes, compare the resolved `handoff_plan` geometry rather than assuming two nearby decimal scales produce a smooth change."
)
text = text.replace(marker, addition, 1)
p.write_text(text)

p = Path("docs/BENCHMARKS.md")
text = p.read_text()
old_row = "| D14-transfer-A/B | Same D14 direction topology; bicubic vs learned 3D clean-state transfer only | Implementation/synthetic validation complete; decoded media pending |"
new_row = "| Learned-transfer media | Progressive Target Input around 1.0–1.06 MP; aggressive source scales; learned 3D clean-state transfer | Decoded-media positive; 0.70 strongest tested quality/compute point, 0.65 begins fidelity loss |"
assert old_row in text
text = text.replace(old_row, new_row, 1)

start = text.index("### Pending strict D14 transfer A/B")
end = text.index("\nThe D12 predecessor used fixed 0.35", start)
section = """### Learned 3D transfer decoded-media validation

The original strict 46×46→56×56 D14 A/B plan was superseded by a more informative aggressive-handoff
sweep around 1 MP. The learned-transfer conclusion comes from decoded media; synthetic contract tests
remain supporting structural evidence only.

The useful progression was:

- around `1152×864` (~0.995 MP) target, a bicubic handoff from roughly a ~0.55 MP private source showed
  substantial body/spatial artifacts in the difficult-motion prompt;
- replacing only the exact-probe clean-state resize with `learned_3d` fixed the majority of those
  artifacts while preserving the same sampler boundary semantics;
- `source_scale=0.70` resolved to `800×608 → 1152×864` and was judged excellent;
- `source_scale=0.65` resolved to `736×576 → 1152×864` and began losing reference-picture likeness and
  tonal stability, so it is not promoted;
- the final higher-resolution gate used `source_scale=0.70` at `832×640 → 1184×896` (~1.061 MP target).
  The video was different in action/content but was again judged very good.

The final ~1.061 MP run used 10 SA-Solver-PECE outer steps, fixed handoff 0.35 snapping to schedule index
6 / unshifted coordinate ~0.400000016 / sigma ~0.888888896, and `direction+acceleration` guidance. Across
two physical chunks it recorded 38 logical H3 calls / 28 actual transformer NFEs / 10 Spectrum
forecasts: low 22/16/6, exact probes 2/2/0, high 14/10/4. It also recorded 6 progressive sampler
invocations, 4 history boundaries, copied audio, rebuilt high-grid conditioning, and an actual first
high-grid H3 call for both chunks. The BF16 CUDA learned provider took about 602 ms and 771 ms; total
learned-transfer wall time was about 628 ms and 802 ms. No sampler failure occurred and the CNN added no
H3 NFE.

This does not establish universal learned-transfer superiority or a universal source-scale optimum.
For this prompt around 1 MP, `0.70` is the current tested quality/compute sweet spot; `0.65` is below the
observed quality floor. The latest learned runs had acceleration enabled, so they must not be cited as
new direction-only or acceleration-promotion evidence.
"""
text = text[:start] + section + text[end:]
p.write_text(text)

p = Path("docs/RESEARCH.md")
text = p.read_text()
start = text.index("The intended D14 test is strict:")
end = text.index("\nThe original difficult-motion D10 reference established", start)
research = """Decoded media now validates the learned boundary beyond the originally planned D14 46×46→56×56
control. Around a 0.995 MP target, an aggressive bicubic transition showed substantial body/spatial
artifacts in the tested difficult prompt; changing only the exact-probe clean-video transfer to the
versioned learned 3D provider fixed the majority of them. At that target, `source_scale=0.70` resolved
to 800×608→1152×864 and was judged excellent, whereas `0.65` resolved to 736×576→1152×864 and began
losing reference likeness / tonal stability.

A final `source_scale=0.70` higher-resolution gate resolved 832×640→1184×896 (~1.061 MP). The generated
action was different but the decoded result was again judged very good. The run preserved the expected
38 logical / 28 actual / 10 forecast topology, two exact probes, six sampler invocations, four history
boundaries, copied audio, and actual high-grid anchors. Learned BF16 CUDA inference cost about 0.60 s
and 0.77 s for the two physical chunks and added no H3 NFE. These learned-transfer media runs used
`direction+acceleration`, so they do not change the earlier conclusion that acceleration has not shown
a clear matched advantage over direction-only. Around 1 MP, 0.70 is the current tested learned-transfer
quality/compute sweet spot for this prompt; no cross-prompt optimum is claimed.
"""
text = text[:start] + research + text[end:]
p.write_text(text)

joined = "\n".join(Path(name).read_text(errors="ignore") for name in tracked if Path(name).is_file())
assert OLD_PIN not in joined
assert "Pending strict D14 transfer A/B" not in joined
assert "No quality or speed advantage is claimed before matched decoded-media validation." not in joined
print("provider pin updated in:", ", ".join(pin_files))
