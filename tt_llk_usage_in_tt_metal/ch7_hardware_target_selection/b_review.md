# Agent B Review: Chapter 7 — Hardware Target Selection — Pass 1

1. **File count for `ARCH_QUASAR` guards is wrong (quasar_divergence.md, line 131).** The chapter states "18 files under `tt_metal/hw/inc/` that contain `#ifndef ARCH_QUASAR` or `#ifdef ARCH_QUASAR` guards." The actual count is 21 files. The enumerated list that follows also only covers 16 of them, omitting `hostdev/dev_msgs.h`, `compute/compute_kernel_api.h`, `internal/debug/stack_usage.h`, `internal/hw_thread.h`, and `internal/tensix_functions.h`. A reader using this list to audit Quasar guard coverage would miss five files.

2. **"Structure is identical to Wormhole" claim for Blackhole is misleading (hal_architecture_dispatch.md, line 84).** The chapter states Blackhole's `includes()` structure "is identical to Wormhole but paths point to `blackhole/`." In the actual source, Wormhole pushes `firmware/src/tt-1xx` only inside the `TENSIX` case (wh_hal.cpp line 123), while Blackhole pushes it after the entire switch statement (bh_hal.cpp line 135), meaning Blackhole adds that include for all core types (TENSIX, ACTIVE_ETH, IDLE_ETH). Someone relying on this "identical structure" claim while implementing or debugging per-core-type include resolution for Blackhole would get the wrong picture.

# Agent B Review: Chapter 7 — Hardware Target Selection — Pass 2

No feedback — chapter approved.
