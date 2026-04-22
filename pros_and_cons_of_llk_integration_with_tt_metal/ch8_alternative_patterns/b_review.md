# Chapter 8 Review -- Agent B (Critic, Pass 1)

## Scope

Flagging only issues where a reader would get a wrong numerical answer, implement something incorrectly, or be materially misled.

## Issues Found

### Issue 1 -- compiled_library_model.md, line 20-21: Template parameter list is incomplete

**Severity: Low-Medium (could cause incorrect implementation)**

The text says explicit instantiation would need to cover "every valid combination of `<EltwiseBinaryType, BroadcastType, MathFidelity, EltwiseBinaryReuseDestType>`" for `_llk_math_eltwise_binary_`. This lists only 4 of the 6 actual template parameters. The real signature (at `tt_llk_blackhole/llk_lib/llk_math_eltwise_binary.h:571-577`) is:

```
template <EltwiseBinaryType, BroadcastType, DstSync, bool is_fp32_dest_acc_en, MathFidelity, EltwiseBinaryReuseDestType>
```

The omitted parameters `DstSync Dst` and `bool is_fp32_dest_acc_en` significantly increase the combinatorial explosion. Someone attempting to build the explicit instantiation lists described in this section would produce an incomplete set if they relied on this text. The document correctly states "6 template parameters" later on line 84, making the two statements internally inconsistent.

**Suggested fix:** Update line 20-21 to list all 6 template parameters, or change the wording to say "a subset of" or "including" rather than implying the list is exhaustive.

---

### Issue 2 -- compiled_library_model.md, line 70-71: Overly narrow claim about `-flto=auto` scope

**Severity: Low (slightly misleading)**

The text says `-flto=auto` is "currently used for compute kernels." In `build.cpp:147`, `-flto=auto` is part of `common_flags` applied to all RISC-V kernel compilations (compute, datamovement, firmware), not just compute kernels. This is a minor inaccuracy. The point being made (that LTO exists but cross-library LTO is fragile) remains valid regardless.

**Suggested fix:** Change "currently used for compute kernels" to "currently used for kernel compilations" or "currently passed as a common flag to the RISC-V cross-compiler."

---

No other issues at the "materially misleading / would cause incorrect implementation" threshold were found. The line counts (wormhole 7,646; blackhole 6,615; quasar 4,210), file references, path references, architectural claims, and code structure descriptions all verified correctly against the source repositories.

# Agent B Review: Chapter 8 — Alternative Patterns — Pass 4

**No feedback — chapter approved.**
