# 03 -- Build System and JIT Compilation

<- [Kernel Headers and LLK Extensions](02_kernel_headers.md) | [Index](index.md)

---

**Source files:** `build_blaze.sh`, `env.sh`, `blaze/kernel_codegen.py`, `blaze/program.py`

## Two-Layer Build Architecture

TT-Blaze uses a two-layer build system. The outer layer (`build_blaze.sh` + CMake) builds host-side C++ libraries and Python bindings -- things that run on the CPU. The inner layer (JIT, driven by tt-metal at runtime) compiles individual kernel `.cpp` files into RISC-V firmware when a program is first dispatched to hardware. This section traces both layers through the mcast -> matmul -> gather example from Sections 01 and 02, showing the complete path from Python `emit()` to compiled firmware on silicon.

```
Host Build (build_blaze.sh)           JIT Kernel Compilation (dispatch time)
  tt-metal -> libtt_metal.so            kernel .cpp + CT args
  migration -> libmigration.so   ->    tt-metal JIT compiler
  pybinds -> _migration.so              -> NCRISC .elf
                                         -> BRISC .elf
                                         -> TRISC .elf
```

---

## Outer Build: build_blaze.sh

**Source file:** `build_blaze.sh` (225 lines)

The top-level build script orchestrates two CMake projects. The design rationale for defaulting to migration-only is iteration speed: tt-metal takes 20-30 minutes to build, while the migration library builds in under a minute. During kernel development, the host infrastructure rarely changes, so the common case is a quick migration rebuild.

### Build Targets

```bash
./build_blaze.sh                    # Default: migration library only
./build_blaze.sh --with-metal       # tt-metal + migration
./build_blaze.sh --metal-only       # tt-metal only
./build_blaze.sh --all              # everything
```

### Build Types

| Flag             | CMake Build Type | Use Case                        |
|------------------|------------------|---------------------------------|
| `--release`      | Release          | Production (default)            |
| `--development`  | RelWithDebInfo   | Debug symbols + optimization    |
| `--debug`        | Debug            | Full debug, no optimization     |

Build artifacts go to `build_$build_type/` (e.g., `build_Release/`), with a `build/` symlink pointing to the active configuration. This means multiple build types can coexist without stepping on each other.

### What build_blaze.sh Does

1. **Builds tt-metal** (if `--with-metal` or `--all`): delegates to `tt-metal/build_metal.sh` with the matching build type flag. The `--build-tests` and `--enable-ccache` flags pass through directly.

2. **Builds the migration library** (default target): a CMake project under `disaggregation/migration/` that links against the tt-metal build. The key CMake variable `TT_METAL_BUILD_DIR` tells the migration project where to find tt-metal's libraries and headers. The script includes fallback logic: if `tt-metal/build_$build_type` does not exist, it falls back to `tt-metal/build`.

   ```bash
   cmake -S "$MIGRATION_DIR" -B "$migration_build_dir" \
       -DCMAKE_BUILD_TYPE="$build_type" \
       -DTT_METAL_BUILD_DIR="$metal_build_dir"
   cmake --build "$migration_build_dir" -j"$jobs"
   ```

3. **Builds Python bindings** (unless `--no-pybinds`): the `_migration` pybind11 extension module. The script locates pybind11's CMake config from the installed Python package. The rationale for making pybinds optional (`--no-pybinds`) is that pybind11 compilation adds time, and pure C++ test iterations do not need the Python bridge.

4. **Creates symlinks**: `build/` -> `build_Release/` (or whichever type), and `disaggregation/migration/build` -> the blaze build dir, so relative paths resolve consistently.

### Other Options

| Flag | Effect |
|------|--------|
| `-j N` / `--jobs N` | Parallel build jobs (default: `nproc`) |
| `--configure-only` | Run cmake configure only, skip build |
| `--clean` | Remove all `build_*` directories and run `build_metal.sh --clean` |

### Build Directory Layout

```
tt-blaze/
    build_Release/
        test/               # C++ test binaries
        python/
            _migration.cpython-311-x86_64-linux-gnu.so
    build -> build_Release   # symlink
    disaggregation/migration/build -> ../../../build_Release  # symlink
```

### What build_blaze.sh Does NOT Do

The outer build does **not** compile kernel `.cpp` files. Kernels are compiled at runtime by the JIT system inside tt-metal. The outer build produces only the tt-metal runtime (compiler infrastructure, device drivers), the migration C++ library, and Python bindings.

---

## env.sh: Environment Setup

**Source file:** `env.sh` (8 lines)

Before any TT-Blaze Python code can run, the environment must be configured:

```bash
source env.sh
```

This does three things:

1. **Activates the tt-metal Python virtual environment**:
   ```bash
   source "$SCRIPT_DIR/tt-metal/python_env/bin/activate"
   ```
   This provides the `ttnn` Python package with its C++ extension modules (the `ttnn.KernelDescriptor`, `ttnn.ProgramDescriptor`, etc. that `UnifiedKernelDescriptor` and `BlazeProgram` depend on).

2. **Sets TT_METAL_HOME** to the tt-metal submodule root:
   ```bash
   export TT_METAL_HOME="$SCRIPT_DIR/tt-metal"
   ```
   This is required by tt-metal's JIT compiler to locate firmware sources, LLK headers, and the RISC-V toolchain. Without it, `#include "api/dataflow/dataflow_api.h"` cannot be found and kernel compilation fails.

3. **Configures PYTHONPATH** for all TT-Blaze modules:
   ```bash
   export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/tt-metal:$SCRIPT_DIR/disaggregation/migration/build/python:$SCRIPT_DIR/build/python:$PYTHONPATH"
   ```

   | Path Entry | Purpose |
   |------------|---------|
   | `$SCRIPT_DIR` | TT-Blaze repo root -- enables `import blaze` |
   | `$SCRIPT_DIR/tt-metal` | tt-metal root -- enables `import ttnn` |
   | `.../migration/build/python` | Migration pybind module (`_migration`) |
   | `.../build/python` | Alternative build output path |

The design rationale for using a sourced script (rather than a wrapper or package installer) is that TT-Blaze is a development framework, not a deployed package. Developers need the environment variables set in their interactive shell, not just in a subprocess.

---

## JIT Compilation: From Python emit() to Firmware

The most important compilation path in TT-Blaze is not `build_blaze.sh` -- it is the JIT compilation that happens when a program is dispatched to hardware. This section traces the complete pipeline through the mcast -> matmul -> gather example.

### Step 1: Kernel Source Generation

**Source file:** `blaze/kernel_codegen.py`

When `BlazeProgram.build()` is called (or when a `FusedProgram` invokes codegen), the `kernel_codegen.py` module generates a `.cpp` file from the `BlazeGraph`:

```python
def generate_kernel_file(graph, output_dir=None, name=None):
    source = generate_kernel(graph)
    content_hash = hashlib.sha256(source.encode()).hexdigest()[:12]
    label = name or "generated"
    out_path = out_dir / f"{label}_{content_hash}.cpp"
    if os.getenv("BLAZE_DEBUG_KERNELS", "0") != "0":
        out_path = out_dir / f"{label}_{content_hash}_debug.cpp"
    # Atomic write: write to .tmp, then rename
    ...
    return str(out_path)
```

The content-hash filename ensures that identical graph compositions produce the same file. If the kernel already exists on disk, generation is skipped entirely. The `_debug` suffix ensures debug and release kernels have separate cache entries.

`BlazeProgram` integrates with codegen through three paths:

```python
# Path 1: Eager -- generate immediately from graph
program = BlazeProgram(graph=my_graph, core_ranges=all_cores)
# Internally calls: self._kernel = generate_kernel_file(graph)

# Path 2: Deferred -- generate later
program = BlazeProgram(core_ranges=all_cores)
program.set_kernel_from_graph(graph, name="shared_expert")

# Path 3: Handwritten -- no codegen
program = BlazeProgram(kernel="models/demos/.../my_kernel.cpp", core_ranges=all_cores)
```

Handwritten kernels use the same `COMPILE_FOR_*` guards and `ct_args::` namespaces as generated kernels -- the difference is purely in who writes the `.cpp`.

### Step 2: Generated Kernel Source

The codegen produces a complete, self-contained `.cpp` file containing includes, type aliases, and a `kernel_main()` with lifecycle calls for each op. See [Ch3 S05 -- Kernel Codegen](../ch03_compilation_pipeline/05_kernel_codegen.md) for the full generated kernel structure and examples. Key properties of the generated code:

- **TRISC initialization**: `deepseek_compute_kernel_init()` inside a `COMPILE_FOR_TRISC` guard sets up the compute engine.
- **No COMPILE_FOR_* guards around op calls**: RISC-specific gating is inside each op's methods (see Section 02). When NCRISC compiles, compute-only code optimizes to empty.
- **DeviceZoneScopedN**: Tracy profiler zones wrap each op (zero-cost when disabled).
- **Mcast lifecycle**: `init()`/`teardown()` bracket the persistent sender state across `operator()()` and `run_as<>()` calls.

### The Codegen Algorithm

The codegen algorithm (4-phase pipeline: group nodes, build PhaseDecls, group mcasts for persistent state, emit C++) and the prefix-to-namespace mapping functions (`_to_ct_args_type()`, `_to_alias()`) are covered in detail in [Ch3 S05 -- Kernel Codegen](../ch03_compilation_pipeline/05_kernel_codegen.md). The key point for the build system: the codegen produces a deterministic `.cpp` file whose content depends only on the graph structure and op registrations.

### Type Alias Generation

The `_to_alias()` function produces readable C++ type alias names:

```python
("act_mcast", "mcast", "Mcast")  -> "ActMcast"
("gu", "kn_sliced_matmul", "Matmul")  -> "GuMatmul"
("matmul", "matmul", "Matmul")  -> "Matmul"
("mcast2", "mcast", "Mcast")  -> "Mcast2"  # suffix already in prefix
```

It splits the prefix on `[._]`, capitalizes each part, and appends the suffix unless the suffix is already contained in the alias.

### Step 3: KernelDescriptor Construction

`BlazeProgram.build()` creates the `UnifiedKernelDescriptor` and calls `get_kernel_descriptors()`. For our example with sender/receiver splits, this produces 6 `KernelDescriptor` objects (see Section 01 for the splitting algorithm):

```
Descriptor 0: NCRISC, sender core(s), ct_args with is_sender=1
Descriptor 1: BRISC,  sender core(s), ct_args with is_sender=1
Descriptor 2: TRISC,  sender core(s), ct_args with is_sender=1
Descriptor 3: NCRISC, receiver cores, ct_args with is_sender=0
Descriptor 4: BRISC,  receiver cores, ct_args with is_sender=0
Descriptor 5: TRISC,  receiver cores, ct_args with is_sender=0
```

Each descriptor carries:
- `kernel_source`: Path to the generated `.cpp` file (same file for all 6)
- `source_type`: `FILE_PATH` (JIT reads from disk, not inline source)
- `core_ranges`: Which cores this descriptor covers
- `compile_time_args`: Positional CT args (used by TensorAccessorArgs)
- `named_compile_time_args`: Named CT args as `(name, value)` tuples
- `config`: Either `DataMovementConfigDescriptor` (NCRISC/BRISC) or `ComputeConfigDescriptor` (TRISC)
- `defines`: Additional preprocessor definitions

### Step 4: ProgramDescriptor Assembly

The descriptors are packaged into a `ProgramDescriptor` (see Ch3 S06 for the `BlazeCompiler` orchestration):

```python
# From program.py build()
return ttnn.ProgramDescriptor(
    kernels=ukd.get_kernel_descriptors().kernels,  # 6 kernel descriptors
    cbs=self._cb_descriptors,                       # CB allocation descriptors
    semaphores=self._semaphore_descriptors,         # program-level semaphores
)
```

### Step 5: JIT Compilation by tt-metal

When `ttnn.generic_op(tensors, program_descriptor)` is called, tt-metal's runtime takes over:

1. **Generates the `ct_args::` header** from each descriptor's `named_compile_time_args`. For the sender-group NCRISC descriptor with args `[("act_mcast.src", 0), ("act_mcast.src_num_pages", 4), ("matmul.k_num_tiles", 4), ("gather.dest_noc_x", 1), ...]`, tt-metal generates:

   ```cpp
   // Auto-generated ct_args header (auto-included via -include)
   namespace ct_args {
   struct act_mcast {
       static constexpr uint32_t src = 0;
       static constexpr uint32_t src_num_pages = 4;
       static constexpr bool src_is_tensor_backed = 1;
       static constexpr uint32_t receiver_semaphore = 4096;
       static constexpr uint32_t dst = 3;
       static constexpr uint32_t dst_num_pages = 4;
       static constexpr bool is_sender = 1;
       static constexpr bool is_receiver = 0;
       static constexpr bool pop_src = 1;
       static constexpr bool init_src = 1;
   };
   struct matmul {
       static constexpr bool is_active = 1;
       static constexpr bool pop_in0 = 1;
       static constexpr uint32_t in0 = 3;
       static constexpr uint32_t k_num_tiles = 4;
       static constexpr uint32_t in0_is_tensor_backed = 0;
   };
   struct gather {
       static constexpr bool is_sender = 1;
       static constexpr bool is_receiver = 0;
       static constexpr bool pop_src = 1;
       static constexpr bool use_per_core = 1;
       static constexpr uint32_t sender_idx = 0;
       static constexpr uint32_t dest_noc_x = 1;
       static constexpr uint32_t dest_noc_y = 1;
       // ...
   };
   }
   ```

2. **Generates the runtime args header** from named runtime args. For matmul's TRISC descriptor:

   ```cpp
   // runtime_args_generated.h (auto-included via -include)
   namespace rt_args {
   struct matmul {
       static constexpr uint32_t weights_l1_address = 0;  // index into RT arg array
   };
   }
   ```

   The kernel reads this via `get_common_arg_val<uint32_t>(rt::matmul::weights_l1_address)`. Note the `rt::` namespace -- this is distinct from `ct_args::`.

3. **Compiles each descriptor** with the appropriate RISC define:

   ```bash
   riscv32-unknown-elf-g++ \
       -DCOMPILE_FOR_NCRISC \
       -include generated_ct_args.h \
       -include runtime_args_generated.h \
       -I $TT_METAL_HOME/... \
       -I $BLAZE_ROOT/blaze/kernels/kernel_includes/ \
       fused_abc123def456.cpp \
       -o ncrisc_sender.elf
   ```

   TRISC compilation additionally defines `UCK_CHLKC_UNPACK`, `UCK_CHLKC_PACK`, `TRISC_MATH`, `TRISC_UNPACK`, and `TRISC_PACK` for the sub-RISC dispatch macros (`MATH((...))`, `UNPACK((...))`, `PACK((...))` described in Section 02).

4. **Caches compiled binaries** by a key derived from source content + CT args + defines + compute config.

5. **Downloads firmware** to each core's instruction memory and launches execution.

---

## The COMPILE_FOR_* Guards in Practice: Per-RISC Views

To make the per-RISC compilation model concrete, here is what the JIT compiles for each RISC from our generated kernel. These are simplified post-preprocessing views showing what each RISC "sees" after the preprocessor has eliminated all dead code.

### NCRISC Compilation (with -DCOMPILE_FOR_NCRISC)

```cpp
inline constexpr bool is_ncrisc = true;
inline constexpr bool is_brisc = false;
inline constexpr bool is_trisc = false;

void kernel_main() {
    // TRISC init: eliminated

    // Mcast::Op::init() -> NCRISC path:
    //   setup_sharded_buffer(src, src_num_pages)  [if is_sender && src_is_tensor_backed]

    // Mcast::Op::operator()() -> NCRISC path:
    //   noc_semaphore_wait + cb_push_back  [if is_receiver]

    // Mcast::Op::teardown() -> empty for NCRISC

    // Matmul::Op::init() -> NCRISC path:
    //   setup_sharded_buffer(in0, k_num_tiles)  [if is_active && in0_is_tensor_backed]

    // Matmul::Op::operator()() -> empty for NCRISC

    // Gather::Op::operator()() -> NCRISC path:
    //   noc_async_write + noc_semaphore_inc  [if is_sender]
}
```

### BRISC Compilation (with -DCOMPILE_FOR_BRISC)

```cpp
inline constexpr bool is_ncrisc = false;
inline constexpr bool is_brisc = true;
inline constexpr bool is_trisc = false;

void kernel_main() {
    // Mcast::Op::init() -> BRISC path:
    //   init_persistent_sender()  [if is_sender]

    // Mcast::Op::operator()() -> BRISC path:
    //   send_data_mcast + send semaphore  [if is_sender]

    // Mcast::Op::teardown() -> BRISC path:
    //   teardown_persistent_sender()  [if is_sender]

    // Matmul::Op -> empty for BRISC (no init, no operator(), no teardown)

    // Gather::Op::operator()() -> BRISC path:
    //   noc_semaphore_wait + cb_push_back  [if is_receiver]
}
```

### TRISC Compilation (with -DCOMPILE_FOR_TRISC)

```cpp
inline constexpr bool is_ncrisc = false;
inline constexpr bool is_brisc = false;
inline constexpr bool is_trisc = true;

void kernel_main() {
    deepseek_compute_kernel_init();

    // Mcast::Op -> empty for TRISC (no init, no operator(), no teardown)

    // Matmul::Op::operator()() -> TRISC path:
    //   custom_mm_block  [if is_active]

    // Gather::Op -> empty for TRISC (no operator())
}
```

Each RISC sees only the code paths relevant to its role. The `if constexpr` guards on per-core flags (like `is_sender`, `is_active`) further eliminate unused paths within each RISC's compilation. The combination of preprocessor guards and constexpr elimination means the final firmware binary for each (RISC, core-group) pair contains exactly the instructions needed for that processor on that subset of cores.

---

## Compilation Matrix: What Gets Compiled Per Group

To be precise about how many compilations occur, here is the full matrix for our example on a grid where one core is the sender and the rest are receivers:

| Group    | RISC   | Key Differentiating Args                     | Binary |
|----------|--------|----------------------------------------------|--------|
| Sender   | NCRISC | `is_sender=1, is_receiver=0, sender_idx=0`   | A      |
| Sender   | BRISC  | `is_sender=1, is_receiver=0`                 | B      |
| Sender   | TRISC  | `is_sender=1, is_active=1`                   | C      |
| Receiver | NCRISC | `is_sender=0, is_receiver=1, sender_idx=*`   | D      |
| Receiver | BRISC  | `is_sender=0, is_receiver=1`                 | E      |
| Receiver | TRISC  | `is_sender=0, is_active=1`                   | F      |

If the `gather.sender_idx` PerCore values cause further splits among receivers (e.g., each receiver has a unique `sender_idx`), the receiver group fragments into multiple subgroups. Each unique combination of CT arg values produces a separate compilation.

In practice, `sender_idx` only varies among sender cores (via `PerCoreCompileTimeDescriptor` scoped to the sender grid). Receiver cores all share `sender_idx=0` (the `other_value`), so they form a single group. This keeps the compilation count at 6 for a typical mcast -> matmul -> gather fusion.

---

## Content-Hashed Caching

The content-hash naming mechanism (`generate_kernel_file()` using `sha256(source)[:12]` as the filename) and the basic caching flow are covered in [Ch3 S05 -- Kernel Codegen](../ch03_compilation_pipeline/05_kernel_codegen.md). This section focuses on the **build-system rationale** for this design.

### Why Content Hashing

**Correctness under concurrency.** Multiple Python processes (e.g., data-parallel training with multiple devices) may generate the same kernel simultaneously. Content-hash naming means they all produce the same filename. The `exists()` check short-circuits redundant writes. The atomic write pattern ensures no process ever sees a partial `.cpp` file:

```python
fd, tmp = tempfile.mkstemp(dir=out_dir, suffix=".cpp.tmp")
closed = False
try:
    os.write(fd, source.encode())
    os.close(fd)
    closed = True
    os.rename(tmp, out_path)  # atomic on same filesystem
except BaseException:
    if not closed:
        os.close(fd)
    with contextlib.suppress(OSError):
        os.unlink(tmp)
    raise
```

The `os.rename` is atomic on POSIX when source and destination are on the same filesystem (guaranteed by `tempfile.mkstemp(dir=out_dir)`).

**Compilation cache reuse.** tt-metal's kernel compiler caches compiled binaries by source file path. Same content = same filename = cached binary reuse. Re-running the same model produces zero recompilations after the first run.

**Deterministic builds.** No timestamps, PIDs, or random seeds in the generated code. Two runs with the same graph produce byte-identical source files, making debugging reproducible.

### Two Levels of Caching

1. **Python-level**: If the same `BlazeGraph` produces the same source, the file already exists on disk and generation is skipped entirely.

2. **JIT-level**: tt-metal maintains its own compilation cache keyed on source content + CT args + defines + compute config. Even if the Python regenerates the file, the JIT recognizes the unchanged configuration and returns the cached binary.

A change in any CT arg value produces a different JIT cache key, triggering recompilation. This is why per-core CT arg splitting multiplies compilation -- each unique CT arg combination is a separate kernel binary.

### Where Generated Files Live

```
tt-blaze/
    generated/
        kernels/
            shared_expert_a1b2c3d4e5f6.cpp
            lm_head_7890abcdef12.cpp
            dense_mlp_deadbeef0123.cpp
```

The `generated/` directory is at the repository root (sibling of `blaze/`). It is gitignored and populated at runtime. Each file is self-contained -- it includes `ops.hpp` and references `ct_args::*` structs that are generated separately by the tt-metal compiler.

---

## Debug Kernel Support

### BLAZE_DEBUG_KERNELS

When `BLAZE_DEBUG_KERNELS` is set, the codegen inserts per-RISC `DPRINT` statements for tracing. The per-RISC filtering mechanism, `_RISC_GUARDS` mapping, value table, and compound-guard generation are covered in [Ch3 S05 -- Kernel Codegen](../ch03_compilation_pipeline/05_kernel_codegen.md).

From the build system perspective, the key behaviors are:

- Debug-enabled kernels get a `_debug` filename suffix (`{label}_{hash}_debug.cpp`), ensuring they never collide with release kernels in the compilation cache.
- The `_debug` suffix propagates to the JIT cache key, so debug and release binaries are compiled and cached independently.
- Enabling DPRINTs on all RISCs can deadlock (DPRINT serializes through a shared buffer), which is why per-RISC filtering exists.

### BLAZE_L1_PROFILE

When set, `BlazeProgram.run()` prints CB allocation statistics before dispatch:

```python
if os.environ.get("BLAZE_L1_PROFILE"):
    from blaze.l1_profile import print_cb_stats
    print_cb_stats(self)
```

---

## Runtime Args: Named Common, Per-Core, and Arrays

Compile-time args are baked into the firmware binary. Runtime args are written to L1 memory before each program invocation and read by the kernel at execution time. TT-Blaze supports three categories.

### Named Common Runtime Args

```python
f.trisc_rt_args([
    ("matmul.weights_l1_address", weights_l1_address),
])
```

The JIT generates `runtime_args_generated.h`:

```cpp
namespace rt_args {
struct matmul {
    static constexpr uint32_t weights_l1_address = 0;  // slot index
};
}
```

The kernel reads the value with zero string overhead:

```cpp
const uint32_t addr = get_common_arg_val<uint32_t>(rt::matmul::weights_l1_address);
```

A typo in the name produces an immediate compile error, unlike the older string-based API (`rt_str.hpp`).

### Per-Core Runtime Args

Some values must vary per core at runtime (e.g., fabric routing tables):

```python
f.unified_per_core_rt_args([
    ("fabric.route_id", {core_a: 5, core_b: 7}),
], other_value=0)
```

This writes different values to each core's runtime arg memory while using the same named-accessor API on the device side. Cores not in the mapping get `other_value`.

### Runtime Arg Arrays

For contiguous multi-slot data (e.g., compressed tensor format metadata):

```python
f.unified_rt_arg_arrays([
    ("tensor_meta.strides", [128, 256, 512]),
])
```

This occupies 3 contiguous runtime arg slots. Per-core variants use `(name, {CoreCoord: [val0, ...]})`:

```python
f.ncrisc_per_core_rt_arg_arrays([
    ("routing.table", {core_a: [1, 2, 3], core_b: [4, 5, 6]}),
])
```

---

## Include Path Architecture

The JIT compiler sets up include paths that enable the three-tier include pattern:

1. **tt-metal system includes**: `api/dataflow/dataflow_api.h`, `api/compute/compute_kernel_api.h`, etc. These come from `$TT_METAL_HOME`.

2. **Blaze kernel includes**: `blaze/kernels/ops.hpp`, `blaze/kernels/kernel_utils.hpp`, etc. Resolved relative to the kernel source path.

3. **LLK extension includes**: Files in `blaze/kernels/kernel_includes/` shadow and extend tt-metal headers. Op headers include them via explicit relative paths:
   ```cpp
   #include "../../../kernels/kernel_includes/tt_metal/include/compute_kernel_api/rmsnorm.h"
   ```

The LLK extensions (see Section 02 for the three-layer hierarchy) are resolved through these include paths. The `TRISC_MATH` and `TRISC_UNPACK` defines control which sub-headers are pulled in:

```cpp
// In kernel_includes/.../rmsnorm.h
#ifdef TRISC_MATH
#include "../../hw/ckernels/blackhole/metal/llk_api/llk_math_rmsnorm_bcast_scalar_dest_reuse_api.h"
#endif
#ifdef TRISC_UNPACK
#include "../../hw/ckernels/blackhole/metal/llk_api/llk_unpack_A_rmsnorm_api.h"
#endif
```

---

## End-to-End Flow: Python to Silicon

```
Host (Python)                        Device (Silicon)
--------------                       ----------------

1. source env.sh
   (activate venv, set TT_METAL_HOME)

2. ./build_blaze.sh
   (compile tt-metal runtime,
    migration library, pybinds)

3. Python program runs:
   a. emit() calls accumulate CT args
      on BlazeProgram
   b. kernel_codegen generates .cpp
      (content-hashed, atomic write)
   c. BlazeProgram.build() creates
      UnifiedKernelDescriptor
   d. get_kernel_descriptors() splits
      into per-group KernelDescriptors
   e. ProgramDescriptor assembled

4. ttnn.generic_op(tensors, prog_desc)
   a. JIT generates ct_args:: header
   b. JIT generates rt:: header
   c. Compiles .cpp 3x per group         ->  NCRISC .elf
      (NCRISC, BRISC, TRISC)             ->  BRISC .elf
                                          ->  TRISC .elf
   d. Caches binaries by content hash
   e. Downloads firmware to cores         ->  Core IMEM loaded
   f. Writes runtime args to L1           ->  Core L1 written
   g. Launches program                    ->  All 3 RISCs execute
                                              in parallel on each core
```

The key design principle throughout: **compile-time constants flow from Python through JIT compilation into RISC-V constants.** A Python `("is_sender", 1)` becomes a C++ `static constexpr bool is_sender = true;` which the RISC-V compiler folds into branch elimination. No runtime dispatch, no virtual calls, no string lookups -- just constants baked into silicon-specific binaries.

---

<- [Kernel Headers and LLK Extensions](02_kernel_headers.md) | [Index](index.md)
