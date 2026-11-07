# Agent B Review — Pass 1 (2026-11-06)

**No feedback — chapter approved.**

# Agent B Review — Pass 2 (2026-11-06)

1. **[01-pcie-peer-bar-dma.md] line 78, "Why device-originated beats host-relayed", second bullet** — The sentence "The one restructure that *did* pay was measured: spreading round traffic over the receiving chip's leads was tried and dropped" is self-contradictory and states the opposite of both its own continuation and its cited source (`notes/glm-ccl.md` lines 175-207 is the note's *"Tried: spreading round traffic over the receiving chip's leads (negative)"* section, which ends "Dropped"). A reader skimming the bullet lead-ins takes away that the spread restructure pays, and a porter could re-try it expecting a win; the very next sentence says the opposite ("The root-relay shape is the measured winner"). Concrete fix: reword to "The one restructure that was tried did **not** pay: spreading round traffic over the receiving chip's leads was measured and dropped — for small slices the forward hop costs more than the ~0.44 us of root inbound serialization it saves, and for larger slices the strict lead PCIe BAR windows serialize the data."

# Agent B Review — Pass 3 (2026-11-06)

**No feedback — chapter approved.**

Pass-2 item resolved: the "did pay / tried and dropped" self-contradiction in `01-pcie-peer-bar-dma.md` ("Why device-originated beats host-relayed", second bullet) now reads "The one restructure that was tried did **not** pay: spreading round traffic over the receiving chip's leads was measured and dropped" — matching `notes/glm-ccl.md` lines 175-207 ("Tried: ... (negative)" ... "Dropped") and the bullet's own continuation. Fix verified in-tree.

Cold sweep of the chapter found no new material errors: spot-verified [MEAS] numbers against `glm-ccl.md` (latency table, in-situ round kinds, gather fits/crossover, ordering-mode and spread-restructure sections, stability history), `glm-eth.md` (message-service bench, traced 0.39 us link, relay/round series, one-link penalty, ladder penalty, cabled-link delta), `glm-glx-ethdec.md` (tree bench series, in-situ buckets, t/s rows, +943 us decomposition, far-PCIe, round tail), `glm-glx-layouts.md` (handoff pings, stage-hop, EP paths, c01u20 power-limit host key), `glm-decode-roofline.md` §2 (round/barrier census), and [FACT] slice sizes against `src/constants.h` (HC record 4224 B, KDA 2048 B, Ra 144 B, Rb/Round-max 4608 B, index record 3088 B); arithmetic re-checked (window offsets, ring area 0x4000-0x58A80, 34 KB crossover, 0.44/0.91 us serialization split, 135000000 cycles = 100 ms); all compendium row cross-references from `01`/`02`/`03` point at the correct rows; navigation footers and chapter links intact.
