# Alternative: Monorepo Absorption and Code Generation

This page covers two related but distinct alternatives. The first (monorepo absorption)
addresses the submodule synchronization friction documented in Ch3. The second (code
generation) addresses the per-architecture duplication documented in Ch5. They can be
adopted independently, but they complement each other well.

---

## Alternative A: Monorepo Absorption

### Overview

In this alternative, the contents of the `tt-llk` repository would be absorbed directly
into `tt-metal`, eliminating the git submodule at
`[tt-metal] tt_metal/third_party/tt_llk/`. The LLK source files would become first-class
citizens of the Metal repository, subject to Metal's CI, code review, and branching
workflows.

### How It Would Work

1. **Migration:** The contents of `[tt-llk] tt_llk_wormhole_b0/`, `tt_llk_blackhole/`,
   `tt_llk_quasar/`, and `common/` would be moved into a directory within Metal, such as
   `[tt-metal] tt_metal/hw/llk/` (alongside the existing `ckernels/` directory).

2. **Build system update:** The `.gitmodules` entry for `tt_llk` would be removed. The
   include paths currently assembled in `JitBuildEnv::init()` and the HAL files
   (`[tt-metal] tt_metal/llrt/hal/tt-1xx/`, `tt-2xx/`) would be updated to point to the
   new in-tree location. The `GLOB_RECURSE` over `third_party/tt_llk/` in
   `[tt-metal] tt_metal/CMakeLists.txt` would be updated accordingly.

3. **CI consolidation:** LLK's independent test suite
   (currently in `[tt-llk] tests/`) would be migrated into Metal's test tree, perhaps at
   `[tt-metal] tests/tt_metal/hw/llk/`. LLK's Python test harness and tt-exalens-based
   hardware tests would run as part of Metal's CI pipeline.

### Pros

#### Atomic commits across the boundary

Today, a change that modifies both LLK and Metal requires two coordinated PRs: one to LLK
and one to Metal that bumps the submodule pointer. This is the synchronization friction
documented in Ch3
([coupling and synchronization](../ch3_disadvantages/coupling_and_synchronization.md)).

In a monorepo, a single commit can modify both `_llk_math_eltwise_binary_init_<>` in
`llk_lib/llk_math_eltwise_binary.h` and its Layer 2 wrapper in
`llk_math_binary_api.h`, with both changes reviewed and tested atomically. This eliminates
the window during which the submodule pointer can drift or conflict across branches.

#### Simplified CI

The current `.gitmodules` configuration pins LLK to the `main` branch. Metal's CI must
check out the correct LLK commit for each build. Any mismatch between the pinned commit
and the expected LLK state causes build failures that are not immediately obvious (the error
appears as a missing or changed function in a deeply nested header).

With monorepo absorption, the LLK code is always at the same commit as the Metal code that
consumes it. There is no submodule to synchronize, no pinned commit to bump, and no branch
divergence to reconcile.

#### No version synchronization overhead

The `TODO(pgk) this list is insane` comment at `build.cpp:268` reflects, in part, the
complexity of assembling include paths that span the submodule boundary. While the include
paths themselves would not change significantly, the build system no longer needs to account
for the submodule potentially being at a different version than expected.

### Cons

#### Loss of LLK's independent testing capability

LLK's test infrastructure (documented in Ch7,
[standalone testing](../ch7_testing_strategy/llk_standalone_testing.md)) is designed to
validate LLK functions in isolation using tt-exalens and a dedicated Python harness. This
isolation is valuable: it allows LLK developers to test hardware register interactions
without running Metal's full JIT compilation pipeline.

Absorbing LLK into Metal risks entangling the test suites. LLK-focused tests might
gradually acquire Metal dependencies, making it harder to validate low-level LLK changes
in isolation. Maintaining a clean separation within the monorepo would require discipline
and explicit test directory boundaries.

#### Larger repository

Metal is already a large repository. Adding LLK's source tree (approximately 74,000+ lines
of LLK code across three architecture directories plus common, or approximately 141,000 lines
for the full repository including tests, documentation, and CI configuration)
increases clone time, CI resource usage, and repository complexity.

The `[tt-llk]` repository also contains infrastructure that is specific to standalone LLK
development: `setup_clangd.sh`, `pyproject.toml`, `package.json`, `CODEOWNERS`,
`CONTRIBUTING.md`. Migrating these into Metal requires decisions about what to keep, merge,
or discard.

#### Harder for LLK-focused contributors

Engineers who work primarily on LLK (hardware register manipulation, address mode
configuration, FPU instruction sequencing) currently have a focused repository with a
smaller code surface. In a monorepo, they would need to navigate Metal's full codebase
to find LLK files, use Metal's CI pipeline for validation, and follow Metal's contribution
workflow even for purely LLK-internal changes.

### Assessment

**Verdict: viable with caveats.**

Monorepo absorption directly solves the highest-friction problem documented in this guide:
cross-repository synchronization. The trade-off is organizational rather than technical --
it requires clear internal boundaries (directory structure, CODEOWNERS rules, test isolation)
to preserve LLK's identity within the larger codebase.

This alternative is most attractive if:
- Cross-boundary changes are frequent (suggesting the two codebases are tightly coupled in
  practice, not just in architecture).
- LLK's independent contributor base is small (reducing the organizational cost).
- Metal's CI infrastructure can accommodate LLK's hardware-specific tests without significant
  rework.

---

## Alternative B: Template-Based Code Generation

### Overview

In this alternative, the per-architecture LLK implementations
(`tt_llk_wormhole_b0/llk_lib/`, `tt_llk_blackhole/llk_lib/`, `tt_llk_quasar/llk_lib/`)
would be generated from a shared set of architecture-agnostic templates combined with
per-architecture configuration files. This addresses the duplication documented in Ch5,
where wormhole_b0 has 7,646 lines in `llk_lib/` (excluding `experimental/`), blackhole has 6,615 (excluding `experimental/`), and quasar has 4,210
-- with significant algorithmic overlap and divergence only in register addresses,
instruction encodings, and a few structural differences.

### How It Would Work

1. **Shared templates:** A new directory (e.g., `[tt-llk] llk_templates/`) would contain
   architecture-agnostic implementations of each LLK operation. These templates would use
   placeholder tokens or Jinja2/Mako-style template variables for architecture-specific
   values:

   ```
   // llk_templates/math_eltwise_binary.h.j2
   template <EltwiseBinaryType eltwise_binary_type, BroadcastType bcast_type, MathFidelity math_fidelity>
   inline void eltwise_binary_configure_addrmod() {
       constexpr std::uint32_t fidelity_increment = is_high_fidelity(math_fidelity) ? 1 : 0;
       constexpr std::uint8_t srcb_incr = (bcast_type == BroadcastType::NONE || bcast_type == BroadcastType::COL)
           ? {{ arch.MAX_FPU_ROWS }} : 0;
       // ...
   }
   ```

2. **Architecture configuration files:** Per-architecture YAML or JSON files would define
   the values that differ across targets:

   ```yaml
   # arch_configs/blackhole.yaml
   MAX_FPU_ROWS: 16
   DEST_REGISTER_HALF_SIZE: 8
   supports_binary_sfpu: true
   unpack_AB_variant: "standard"    # vs. quasar's "binary_operands"
   ```

3. **Generation step:** A build-time script would render the templates with each
   architecture's configuration, producing the current per-architecture header files.
   The generated files would be committed to the repository (not generated on the fly)
   to preserve debuggability and avoid build-time dependencies on the template engine.

### Pros

#### Single source of truth for algorithmic logic

The core algorithm for operations like eltwise binary (address mode configuration, FPU
instruction sequencing, iteration logic) is largely the same across wormhole_b0 and
blackhole, with only register addresses and a few constants differing. As documented in Ch5
([duplication analysis](../ch5_architecture_duplication/duplication_analysis.md)), these two
architectures have nearly identical file lists and function structures.

A shared template would ensure that bug fixes and algorithmic improvements are applied once
and propagated to all architectures automatically. This directly addresses the maintenance
burden documented in Ch5
([maintenance burden](../ch5_architecture_duplication/maintenance_burden.md)), where bug
fixes must currently be applied N times.

#### Per-architecture specialization via configuration

Quasar's structural divergence (different file names, different operation decomposition)
can be accommodated through conditional blocks in templates rather than entirely separate
implementations. Where quasar requires a fundamentally different algorithm (not just
different constants), the template can include architecture-specific blocks:

```
{% if arch.name == "quasar" %}
    // Quasar-specific binary broadcast implementation
{% else %}
    // Standard wormhole/blackhole implementation
{% endif %}
```

This makes the divergence explicit and auditable, unlike the current model where differences
between `llk_math_eltwise_binary_sfpu.h` (wormhole/blackhole) and
`llk_math_eltwise_binary_broadcast.h` (quasar) must be discovered by manual file comparison.

#### Reduced review burden for cross-architecture changes

When a new LLK operation is added, a reviewer currently needs to inspect three separate
implementations (one per architecture) to verify consistency. With code generation, the
reviewer inspects one template and one configuration delta per architecture.

### Cons

#### Significant upfront investment

Building the code generation infrastructure requires:
- Analyzing every file in `llk_lib/` across all three architectures to identify which
  lines differ and which are shared.
- Designing a template language or choosing an existing one (Jinja2, Mako, or a custom
  C preprocessor-based approach).
- Creating the architecture configuration files.
- Writing the generation script and integrating it into the build process.
- Validating that generated output matches the current hand-written implementations
  exactly (to avoid introducing regressions).

Given the volume of code (approximately 18,000+ lines across three architectures in
`llk_lib/` alone, plus per-architecture common includes), this is a multi-week effort.

#### Code generation tooling maintenance

The template engine, configuration schema, and generation scripts become part of the
project's critical infrastructure. They must be maintained, documented, and understood by
contributors. If the template engine has bugs or limitations, they become LLK's bugs.

Additionally, the choice of template language introduces a dependency: Python-based engines
(Jinja2, Mako) require a Python environment in the build path; C preprocessor-based
approaches are limited in expressiveness.

#### Harder to debug generated code

When a kernel compilation fails with an error in a generated LLK header, the developer
must trace the error back to the template source. Line numbers in error messages refer to
the generated file, not the template. This adds a layer of indirection to an already
challenging debugging experience (Ch3,
[developer ergonomics](../ch3_disadvantages/developer_ergonomics.md)).

Committing the generated files (rather than generating on the fly) mitigates this partially:
developers can read and debug the generated code directly, and the template is consulted
only when making changes. However, this creates the risk of generated files drifting from
their templates if someone edits the generated file directly.

#### Quasar's structural divergence limits template reuse

As documented in Ch5, quasar's LLK implementation differs from wormhole/blackhole not just
in constants but in structure: different file names (`llk_unpack_binary_operands.h` vs.
`llk_unpack_AB.h`), different operation decompositions, and different function signatures.
These structural differences may require so many conditional blocks in templates that the
templates become harder to read than the current separate implementations.

### Assessment

**Verdict: high potential, high cost -- suitable as a long-term initiative.**

Code generation is the only alternative that directly addresses the per-architecture
duplication documented in Ch5. The maintenance burden of applying every change three times
is real and will only grow as new architectures are added.

However, the upfront investment is substantial and the risk of template complexity
(especially for quasar) is non-trivial. A recommended approach would be:

1. **Start with wormhole/blackhole only.** These two architectures have the highest overlap
   and would benefit most from shared templates. Quasar can remain hand-written initially.

2. **Begin with a single operation family.** Generate `llk_math_eltwise_binary*.h` from a
   shared template as a proof of concept. Validate that the generated output matches the
   current hand-written code exactly.

3. **Expand incrementally.** If the proof of concept demonstrates maintainability
   improvements without excessive template complexity, extend to other operation families.

4. **Keep generated files committed.** This preserves debuggability and avoids adding the
   template engine to the critical build path.

---

**End of guide.** Return to [Guide Index](../index.md)
