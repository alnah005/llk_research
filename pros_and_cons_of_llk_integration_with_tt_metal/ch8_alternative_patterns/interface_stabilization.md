# Alternative: Interface Stabilization

## Overview

In this alternative, the Layer 1/Layer 2 boundary between LLK and Metal would be formalized
with a versioned API contract. Today, Metal's Layer 2 wrappers (e.g.,
`[tt-metal] hw/ckernels/blackhole/metal/llk_api/llk_math_binary_api.h`) directly call LLK's
`_llk_*` functions by including the full implementation headers. There is no formal
declaration of which functions constitute the stable API surface and which are internal
implementation details.

This alternative preserves the header-only compilation model -- and therefore all the
performance benefits documented in Ch2 -- while adding a thin contract layer that enables
independent development cadences and clearer ownership.

## How It Would Work

### Step 1: Define `llk_api.h` per architecture

A new header file would be introduced in each LLK architecture directory:

```
[tt-llk] tt_llk_blackhole/llk_api.h
[tt-llk] tt_llk_wormhole_b0/llk_api.h
[tt-llk] tt_llk_quasar/llk_api.h
```

Each `llk_api.h` would:
- Declare the public `_llk_*` function signatures that Metal is permitted to call.
- Include an API version macro: `#define LLK_API_VERSION 1`.
- Document parameter contracts (valid ranges, preconditions) for each function.
- Mark all other functions and implementation details as internal (e.g., by convention using
  an `_llk_internal_*` prefix or by placing them in a `detail/` subdirectory).

### Step 2: Add build-time version checking

Metal's Layer 2 wrappers would include a static assertion:

```cpp
#include "llk_api.h"
static_assert(LLK_API_VERSION >= 1 && LLK_API_VERSION < 2,
    "Metal Layer 2 wrappers require LLK API version 1.x");
```

This would cause a clear, immediate compile error if the LLK submodule is updated to an
incompatible API version, rather than the current failure mode of cryptic template
instantiation errors deep in the include chain (as described in Ch3,
[developer ergonomics](../ch3_disadvantages/developer_ergonomics.md)).

### Step 3: Restructure LLK headers

Within each architecture's `llk_lib/` directory, functions would be organized into:
- **Public API** (included via `llk_api.h`): functions called by Metal's Layer 2 wrappers.
  These are the `_llk_*` functions that appear in Metal's `llk_api/*.h` files today, such as
  `_llk_math_eltwise_binary_init_<>`, `_llk_math_eltwise_binary_<>`,
  `_llk_unpack_AB_init_<>`, `_llk_pack_<>`, etc.
- **Internal implementation**: helper functions like `eltwise_binary_configure_addrmod<>` in
  `[tt-llk] tt_llk_blackhole/llk_lib/llk_math_eltwise_binary.h`, address mode
  configuration, register manipulation utilities. These would remain accessible (they are
  still `inline` in headers) but would not be part of the versioned contract.

### Step 4: Introduce contract tests

As identified in Ch7 ([testing gaps](../ch7_testing_strategy/testing_gaps.md), Gap 1), there
are no tests that verify the Layer 1/Layer 2 interface. With a formal `llk_api.h`, a
lightweight CI check could:
- Parse the `llk_api.h` header to extract declared function signatures.
- Verify that Metal's Layer 2 wrappers call only functions declared in the API header.
- Flag any Metal code that reaches into LLK's internal implementation directly.

## Pros

### Enables independent development cadences

Today, any change to LLK's internal implementation risks breaking Metal if it alters a
function signature that Metal happens to call. With a stable API contract, LLK developers
could refactor internals freely as long as the public API signatures are preserved. This
directly addresses the coupling pain documented in Ch3
([coupling and synchronization](../ch3_disadvantages/coupling_and_synchronization.md)).

For example, LLK could reorganize the helper functions in
`[tt-llk] tt_llk_blackhole/llk_lib/llk_math_eltwise_binary.h` (address mode configuration,
FPU instruction sequencing) without coordinating with Metal, as long as the public
`_llk_math_eltwise_binary_init_<>` and `_llk_math_eltwise_binary_<>` signatures remain stable.

### Clearer ownership boundaries

The current ambiguity about which repo owns what -- highlighted in Ch4
([leaky abstractions](../ch4_api_boundary/leaky_abstractions.md)) with the example of
`DST_SYNC_MODE` and `DST_ACCUM_MODE` as Metal concepts leaking into LLK's API surface --
would be partially addressed. The `llk_api.h` header would serve as a single document
defining "LLK provides these functions; Metal provides these parameters."

### Preserves all runtime performance characteristics

Unlike the compiled library model, interface stabilization does not change the compilation
model. LLK headers are still included at compile time, inlining and LTO still apply, and
template specialization still produces optimal code for each kernel. The `llk_api.h` header
is purely a documentation and validation artifact -- it adds no runtime cost.

### Better error diagnostics on submodule updates

When the LLK submodule is bumped today and a function signature has changed, the error
manifests as a template instantiation failure deep in the include chain, often with error
messages spanning dozens of lines (Ch3,
[developer ergonomics](../ch3_disadvantages/developer_ergonomics.md)). With a version-checked
`llk_api.h`, the error would instead be a clear `static_assert` message indicating an API
version mismatch, immediately directing the developer to the changelog.

## Cons

### Added ceremony

Every API change in LLK would require updating `llk_api.h`, bumping the version number,
documenting the change, and coordinating with Metal to update the version check range. For
a rapidly evolving codebase, this ceremony could slow development velocity.

The current model's lack of ceremony (Ch2,
[tight coupling benefits](../ch2_advantages/tight_coupling_benefits.md)) is a genuine
advantage during periods of rapid iteration. Interface stabilization assumes the API is
mature enough to benefit from stability guarantees.

### Risk of abstraction ossification

A versioned API contract creates pressure to maintain backward compatibility. This can
prevent beneficial API changes that would improve performance or simplify the implementation.

For example, if a new Tensix architecture requires a fundamentally different parameter
interface for binary operations (as seen with quasar's structural divergence from
wormhole/blackhole documented in Ch5,
[duplication analysis](../ch5_architecture_duplication/duplication_analysis.md)), a versioned
API might resist the change rather than accommodate it cleanly.

### May limit hardware-specific optimizations

Some optimizations require passing architecture-specific parameters that do not generalize
across all targets. A stable API contract must choose between:
- A lowest-common-denominator interface (losing arch-specific optimization opportunities), or
- Per-architecture API extensions (increasing the API surface area and complexity).

The current model avoids this tension because Metal's per-architecture wrappers in
`[tt-metal] hw/ckernels/<arch>/metal/llk_api/` are free to call whatever LLK functions
exist in that architecture's `llk_lib/`, including architecture-specific functions that do
not exist in other architectures.

### Requires coordinated effort across three architectures

The `llk_api.h` header would need to be created and maintained for each of the three
architecture directories in LLK. Given the structural divergence documented in Ch5 (quasar
uses different file names and function decompositions than wormhole/blackhole), the API
contracts might differ significantly across architectures, reducing the value of
standardization.

## Assessment

**Verdict: recommended incremental improvement.**

Interface stabilization offers the best risk-to-reward ratio of the alternatives considered.
It addresses the implicit API coupling (Ch4), improves error diagnostics (Ch3), and enables
contract testing (Ch7, Gap 1) without sacrificing runtime performance (Ch2) or requiring a
large-scale refactoring effort.

The key implementation question is timing: this alternative works best when the LLK API
surface has stabilized enough that the ceremony of versioning is justified. Given that the
current `_llk_*` function signatures have remained relatively stable across the existing
three architectures, this threshold appears to have been reached.

**Recommended starting point:** Create `llk_api.h` for the blackhole architecture first
(since it has the most overlap with wormhole_b0), add `LLK_API_VERSION` checking to
Metal's blackhole Layer 2 wrappers in
`[tt-metal] hw/ckernels/blackhole/metal/llk_api/`, and run the approach for one development
cycle before extending to wormhole_b0 and quasar.

---

**Next:** [`monorepo_and_code_generation.md`](./monorepo_and_code_generation.md)
