# Agent B Review: Chapter 1 — Architecture and Mental Model

## Pass 1

**No feedback — chapter approved.**

## Pass 2

**No feedback — chapter approved.**

## Pass 3

1. **Overlap formula contradicts the prose (Section 1.1.8, `01_tensix_core_model.md`).** The text builds a detailed argument that TRISC advances into the Matmul phase while DM0/DM1 are still finishing Mcast, implying true temporal overlap between data movement and compute. But the formula immediately following says `T_fused = T_mcast + T_compute` -- which is fully sequential, not overlapped. If the overlap described in the prose is real, the formula should reflect it (e.g., `max(T_mcast, T_compute)` or partial overlap). As written, the only savings shown is `T_init` (inter-kernel dispatch overhead), which is a much weaker claim than the "free overlap" the surrounding text argues for. A reader trying to estimate fused kernel performance from this formula would get the wrong answer.

2. **Python code shown with C++ comment syntax (Section 1.2.2, `02_blaze_compilation_pipeline.md`).** The "Step 1: Python emit()" block in the CT arg flow walkthrough uses `//` comments inside what is described as Python code (`f.trisc_ct_args([...])`). Python uses `#` for comments. The entire block is fenced as ` ```cpp `, which makes the language ambiguity worse -- a reader copying this as a Python implementation template will get syntax errors from the `//` comment tokens. Either re-fence as Python with `#` comments, or split the walkthrough so the Python and C++ steps are in separate, correctly-fenced code blocks.

## Pass 3 -- Fixes Applied (Agent A)

1. **Overlap formula fixed (Section 1.1.8, `01_tensix_core_model.md`).** Replaced the contradictory single-formula model with a two-benefit explanation: (a) eliminated dispatch overhead between phases, and (b) true pipeline overlap across phases in steady state, expressed as `T_fused_per_iteration = max(T_data_movement, T_compute)` vs. the separate-kernel cost `T_separate_per_iteration = T_dispatch + T_data_movement + T_init + T_compute`. Also clarified that for a single pass (non-steady-state), the benefit is the elimination of dispatch and init latency, since processors are pre-positioned at phase boundaries.

2. **Python/C++ code block split (Section 1.2.2, `02_blaze_compilation_pipeline.md`).** Split the single mixed-language `cpp` code block into four separate, correctly-fenced blocks: the Python `emit()` step uses a `python` fence with `#` comments, while the three C++ steps (generated header, kernel header, kernel instantiation) each use `cpp` fences with `//` comments. Each block is preceded by a bold step heading for clarity.

## Pass 4

**Verification of Pass 3 fixes:**

1. **Overlap formula (Section 1.1.8):** Fix is correct. The text now presents two distinct benefits: (a) eliminated dispatch overhead, and (b) true pipeline overlap expressed as `max(T_data_movement, T_compute)` for steady state, contrasted with the sequential `T_dispatch + T_data_movement + T_init + T_compute` baseline. The single-pass caveat is also included. The formula now matches the prose.

2. **Python/C++ code blocks (Section 1.2.2):** Fix is correct. The four steps are in separate, correctly-fenced blocks (Python with `#` comments, C++ with `//` comments). No remaining language/fence mismatches.

**Remaining issues:**

**No feedback — chapter approved.**
