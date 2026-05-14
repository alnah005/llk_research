# 8.3 Common Hang Causes

Device hangs are the most frustrating class of bug in Blaze development. The device stops making progress, the host blocks indefinitely waiting for completion, and the only signal is silence. This section catalogs the most common root causes and provides a systematic diagnosis workflow.

## CB Deadlocks

### Producer/Consumer Page Count Mismatches

A circular buffer deadlock occurs when the producer and consumer disagree on how many pages the CB holds. The CB acts as a bounded FIFO -- the producer blocks when the buffer is full (waiting for the consumer to pop), and the consumer blocks when the buffer is empty (waiting for the producer to push).

**Root cause**: If the producer's CB descriptor says `num_pages=2` but the consumer's descriptor says `num_pages=4`, the producer may fill what it thinks is the full buffer (2 pages), then wait for the consumer to pop. But the consumer thinks there are 4 pages and may wait for more data before popping. Neither side makes progress.

**How this happens in Blaze**: In `FusedProgram`, when you allocate a scratch CB with `cb_scratch()`, the `num_pages` parameter must match on both the producer and consumer sides. If a micro-op's `emit()` allocates a scratch CB with `num_pages=2` but a downstream op expects a different page count, a deadlock results.

In the graph compilation path, the CB engine computes `num_pages` from graph kwargs:

```python
# blaze/cb_engine.py
num_pages = node.kwargs.get(f"num_tiles_{port_spec.name}", DEFAULT_NUM_PAGES)
```

`DEFAULT_NUM_PAGES = 1`. Many ops need `num_pages >= 2` for double-buffering. If you forget to pass `num_tiles_in0=2` in the graph kwargs, the CB gets 1 page but the kernel tries to double-buffer, causing an immediate deadlock.

**Source reference** -- `fused_program.py` `cb_scratch()`:

```python
def cb_scratch(
    self, name: str, *,
    num_pages, core_ranges, data_format, tile, page_size,
    balanced: bool = True,
) -> CBHandle:
    cb_id = self.program.cb_scratch(
        name=name, num_pages=num_pages,
        core_ranges=core_ranges, data_format=data_format,
        tile=tile, page_size=page_size,
    )
    ...
    handle = CBHandle(
        cb_id=cb_id, num_pages=num_pages,
        page_size=page_size, core_ranges=core_ranges,
        data_format=data_format,
        tile_desc=BlazeProgram._normalize_tile(tile),
        scratch_name=name,
    )
    return handle
```

The `CBHandle` carries the agreed-upon `num_pages` and `page_size`. Downstream ops should read these values from the handle rather than using their own constants.

### Missing `init()` Push

Some ops have an `init()` method that performs an initial push into a CB to prime the pipeline. If `init()` is skipped (e.g., because `init_is_empty` is incorrectly set to `True` in the `PhaseInfo`), the consumer blocks indefinitely waiting for the first page.

**Source reference** -- `kernel_codegen.py`:

```python
emit_init = not pdecl.phase_info.init_is_empty
emit_teardown = not pdecl.phase_info.teardown_is_empty

if pdecl.op_type == "mcast":
    ...
    if emit_init:
        lines.append(f"    {var_name}.init();")
    ...
else:
    ...
    if emit_init:
        lines.append(f"        {var_name}.init();")
    ...
```

If `PhaseInfo(init_is_empty=True)` is set but the C++ `Op::init()` actually does work (like an initial NOC read), the codegen skips the `init()` call and the pipeline stalls.

**Specific example**: In the mcast op, the `init_src` flag controls whether the sender core initializes the source CB. When `needs_init` is `True` (source is a tensor that needs an NCRISC read) but the flag is incorrectly set to `False`, the mcast sender never fills the source CB, and all receivers hang waiting for the multicast.

### Unbalanced Push/Pop Across Cores

When a CB is pushed on some cores but popped on a different (or smaller) set of cores, the cores that pushed but never popped leave stale data in the FIFO. The `balanced` parameter on `cb_scratch()` controls whether the CB participates in temporal compaction:

```python
cb_scratch(..., balanced=False)
```

Setting `balanced=False` marks the CB in `_unbalanced_cbs`, excluding it from temporal reuse:

```python
# From blaze/fused_program.py
# Scratch CBs whose push/pop is NOT balanced per core (e.g. mcast
# dst pushed on every receiver but popped only by the consumer
# subset). Excluded from temporal reuse -- fifo state would leak
# onto cores that pushed but never popped.
self._unbalanced_cbs: set[int] = set()
```

Forgetting this flag when the push/pop pattern is genuinely unbalanced (e.g., mcast destination CB -- pushed on every receiver, popped only by the consumer subset) can cause the compaction pass to reuse the CB for a later phase, leading to data corruption or deadlock.

### Fan-Out CB Conflicts

When one output feeds multiple consumers (fan-out), the CB engine assigns a single CB ID with merged core ranges. A deadlock can occur if the consumers pop at different rates -- if consumer A pops before consumer B, the producer may overwrite data B still needs.

**Prevention**: Fan-out CBs should use enough pages to accommodate the worst-case consumer lag, or the pipeline should ensure consumers run synchronously.

## Semaphore Mismatches

### Sender/Receiver Asymmetry

Semaphores in Blaze coordinate producer-consumer handshakes between cores. The sender increments the semaphore; the receiver waits for it to reach a target value and then resets it. If the number of senders does not match the receiver's expected count, a hang occurs.

**Common scenarios**:
- A gather op expects `num_senders = N` but only `N-1` cores actually send (e.g., a grid mismatch where one core is excluded)
- The sender increments semaphore ID `X` but the receiver waits on semaphore ID `Y`
- Passing `sender_semaphore` where `receiver_semaphore` is expected in CT args (or vice versa)

The `SemProtocol` enum defines valid protocols:

```python
# blaze/blaze_op.py
class SemProtocol(str, Enum):
    SENDER = "sender_semaphore"
    RECEIVER = "receiver_semaphore"
    NOC0_RECEIVER = "noc0_receiver_semaphore"
    NOC1_RECEIVER = "noc1_receiver_semaphore"
```

### Wrong Initial Values

Semaphore initial values matter. In `FusedProgram.semaphore()`:

```python
def semaphore(self, name: str | None = None, *,
              program_semaphore: bool = False,
              initial_value: int = 0, ...):
    ...
```

If a barrier semaphore starts at a nonzero value, the receiver may think it has already received data and proceed without waiting, causing data corruption or a later deadlock when the pipeline state becomes inconsistent.

For named mesh-global semaphores, the `initial_value` only applies on first allocation. If two ops request the same semaphore name with different initial values, only the first call's value is used:

```python
# From fused_program.py
def _alloc_mesh_semaphore(mesh_device, name, sem_dict,
                           initial_value=0):
    """initial_value applies on first allocation only; later
    dedup-returns ignore it. Passing different values for the
    same name is a caller bug."""
    if name not in sem_dict:
        ...
        sem_dict[name] = ttnn.create_global_semaphore(
            mesh_device, avail, initial_value)
    return sem_dict[name]
```

### Semaphore ID Collision

If two independent sync patterns share the same semaphore (e.g., because `f.semaphore(name)` is called with the same `name` for different purposes), increments from one pattern interfere with the other.

The `FusedProgram.semaphore()` method detects duplicate calls within one op:

```python
if any(s is sem for s in self.program._global_semaphores):
    raise RuntimeError(
        f"f.semaphore({name!r}) called twice within one op -- "
        "give distinct sync points distinct names."
    )
```

But this check is per-op, not cross-op. Two different ops using the same name will silently share a semaphore. Use the prefix convention (`f"{prefix}.barrier"`) to namespace semaphores within composed ops.

For program-local semaphores (`program_semaphore=True`), slot-based allocation means two ops on overlapping core ranges using the same slot will interfere. Use `core_ranges=` to restrict semaphores to specific cores when they do not need to be global:

```python
# Per-device slot-based sync, restricted to matmul cores
sem = f.semaphore(program_semaphore=True, core_ranges=matmul_core_grid)
```

## Phase Ordering Errors

### Wrong Execution Sequence

The generated kernel executes phases in the order they appear in the shadow graph's node list, which matches the `emit()` call order. If ops are emitted in the wrong order, a consumer may run before its producer has started.

From `kernel_codegen.py`:

```python
# Use node insertion order -- this matches the user's emit() call order,
# which defines the correct execution sequence.
node_order = graph.nodes
```

For example, if you emit Matmul before Mcast, the Matmul's compute phase will try to read from the activation CB before Mcast has written to it, causing an indefinite wait.

**Wrong:**
```python
matmul_out = Matmul.emit(f, some_cb, weights, prefix="matmul")  # CB empty!
mcast_out = Mcast.emit(f, input, prefix="mcast")                # Too late
```

**Correct:**
```python
mcast_out = Mcast.emit(f, input, prefix="mcast")               # Fill first
matmul_out = Matmul.emit(f, mcast_out, weights, prefix="matmul")  # Then consume
```

### Mcast Leader/Follower Ordering

Mcast ops are special-cased in codegen. Multiple mcasts sharing the same grid are grouped, with the first becoming the "leader" (owns `init()`/`teardown()`) and subsequent ones becoming "followers" (call `run_as<>()` on the leader object):

```cpp
// Generated for mcast leader + 1 follower:
Mcast mcast;
mcast.init();
{
    DeviceZoneScopedN("MCAST");
    mcast();
}
mcast.template setup_src<ct_args::mcast2>();
{
    DeviceZoneScopedN("MCAST2");
    mcast.template run_as<ct_args::mcast2>();
}
mcast.teardown();
```

The leader's `teardown()` is only called after the last follower completes. If you add a new mcast to an existing pipeline but emit it out of the leader/follower sequence, the kernel structure breaks.

### Missing Role Flags

Many ops use per-core boolean flags (`is_sender`, `is_active`, `is_receiver`) to control which cores participate in each phase. If a flag is not declared for all required cores, some cores may skip their role in the pipeline, leaving downstream consumers waiting.

From `BlazeOp`, all flags must be explicitly declared, even when disabled. The `flag()` method on `FusedProgram` creates per-core CT arg tuples:

```python
def flag(self, name, cores, enabled=True):
    """Build a per-core CT arg tuple for a boolean flag.
    The flag is 1 on the given cores when enabled, 0 when disabled.
    Cores not in the range always get 0."""
    return (name, {cores: int(enabled)})
```

A common mistake is forgetting to emit a flag for a core range that was added later in development. The C++ side reads the flag as 0 (uninitialized) and the core does nothing.

**Example**: If `gather.is_sender` is not set on the matmul cores, those cores never send their data to the gather receiver. The receiver blocks waiting for `noc0_num_senders` signals that never arrive.

## NOC Congestion and DRAM Worker Conflicts

### NOC Congestion

On large grids, heavy NOC traffic from one phase can block NOC transactions in another phase if they share NOC channels. This does not cause a true deadlock (it is a livelock that eventually resolves), but it can appear as a hang if the timeout is too short.

Symptoms:
- Test passes on small grids but hangs on full grids
- Adding `DeviceZoneScopedN` Tracy markers shows one phase taking orders of magnitude longer than expected

### DRAM Worker Conflicts

DRAM-streaming matmul ops read weight tiles from DRAM banks. If two ops read from the same DRAM bank simultaneously, bank contention can cause very long stalls. This is a performance issue rather than a correctness issue, but can trigger test timeouts.

The grid configuration excludes DRAM worker cores from compute grids. Always use `GridConfig.from_device(device)` to get safe core sets. If you manually construct a core grid that includes DRAM worker positions, NOC transactions to those cores may conflict with DRAM controller operations.

### Grid Mismatches

If a producer writes to cores `[(0,0), (0,1), (0,2)]` but the consumer reads from `[(0,0), (0,1), (0,2), (0,3)]`, core `(0,3)` waits for data that never arrives. This is a silent hang, not a crash.

Grid mismatches often arise from:
- Using `f.all_cores` when the op should only run on `f.matmul_cores`
- Computing `core_ranges` from a tensor's shard spec that does not match the intended grid
- `sender_grid` vs `mcast_receiver_grid` confusion
- Forgetting that the sender core (e.g., `(12, 9)` on Blackhole) is excluded from matmul cores

## Diagnosis Workflow

### Step 1: Identify the Hanging Phase

```bash
BLAZE_DEBUG_KERNELS=1 TT_METAL_DPRINT_CORES="0,0" pytest test_my_op.py -s -v --timeout=30
```

Look for the last `[PHASE] <NAME> start` without a corresponding `[PHASE] <NAME> done`. That phase is where the hang occurs.

Example output:
```
[PHASE] ACT_MCAST start
[PHASE] ACT_MCAST done
[PHASE] MATMUL start
# ... no "MATMUL done" -> hang is inside the MATMUL phase
```

### Step 2: Narrow to a Specific RISC

```bash
BLAZE_DEBUG_KERNELS=ncrisc TT_METAL_DPRINT_CORES="0,0" pytest ...
BLAZE_DEBUG_KERNELS=trisc TT_METAL_DPRINT_CORES="0,0" pytest ...
BLAZE_DEBUG_KERNELS=brisc TT_METAL_DPRINT_CORES="0,0" pytest ...
```

If only one RISC shows the hang (e.g., NCRISC prints "start" but not "done" while TRISC completes), the hang is in that RISC's code path for that phase. If NCRISC prints but TRISC does not, the compute kernel is waiting on a CB that was never filled (NCRISC reader never ran).

### Step 3: Inspect CB State

```bash
BLAZE_L1_PROFILE=1 pytest test_my_op.py -s -v
```

Check:
- Do all CBs have the expected `num_pages` and `page_size`?
- Are `core_ranges` correct for each CB?
- Are tensor-backed vs scratch classifications correct?
- Do any CBs have unexpected `l1_addr` values (suggesting overlapping allocations)?
- Do page sizes match tile geometry (e.g., a `bfloat16` `32x32` tile = 2048 bytes)?

### Step 4: Verify Push/Pop Balance

For the hanging phase, trace through the C++ Op implementation:
- Count the number of `cb_push_back()` calls on the producer side
- Count the number of `cb_pop_front()` calls on the consumer side
- Verify they match over one iteration of the loop
- Check that the number of loop iterations matches on both sides (derived from the same CT arg, e.g., `k_num_tiles`)

### Step 5: Check Semaphore State

If the hang involves inter-core synchronization:
- Verify that `num_senders` CT arg matches the actual number of cores that call `noc_semaphore_inc`
- Verify the receiver's wait threshold matches the number of expected increments
- Check that the semaphore is reset after each use (receiver does `noc_semaphore_set(addr, 0)` after the wait)

### Step 6: Bisect with Escape Hatches

If the hang is intermittent or grid-size-dependent:

```bash
# Bisect CB sharing
BLAZE_DISABLE_TENSOR_CB_SHARE=1 pytest test_my_op.py -s -v

# Bisect temporal reuse
BLAZE_DISABLE_TEMPORAL_REUSE=1 pytest test_my_op.py -s -v
```

If the first resolves the hang, the CB sharing optimization assigned overlapping addresses to CBs that are actually used concurrently. If the second resolves it, temporal CB reuse is conflicting.

### Step 7: Inspect Generated Kernel

Read the generated kernel to verify the phase execution order and init/teardown calls are correct:

```python
f = FusedProgram(kernel=None, device=device, name="debug")
# ... emit ops ...
compiled = f.build()
print(f.program.generated_kernel)
```

Verify:
- Phases appear in the expected order
- `init()` is called for phases that need it (not suppressed by `init_is_empty=True`)
- `teardown()` is called for phases that need cleanup
- Mcast followers correctly use `run_as<>()` with the right ct_args type

### Step 8: Narrow to a Specific Core

```bash
TT_METAL_DPRINT_CORES="(5,3)" BLAZE_DEBUG_KERNELS=1 pytest test_my_op.py
```

Compare DPRINT output between different cores to find asymmetric behavior.

### Step 9: Visualize the Graph

```python
blaze.visualize(ctx.graph, name="debug_pipeline")
```

Check for unexpected edges, missing CBs, or incorrect semaphore assignments.

## Quick Reference: Hang Symptom to Cause

| Symptom | Likely Cause | First Check |
|---------|-------------|-------------|
| All cores hang at same phase | CB page count mismatch | `BLAZE_L1_PROFILE=1` |
| Sender hangs, receivers proceed | Receiver semaphore never sent | Semaphore names in emit() |
| Receivers hang, sender proceeds | Sender never multicasts | `is_sender`/`is_receiver` flags |
| Only some cores hang | Grid mismatch | CB core_ranges vs kernel core_ranges |
| Phase N hangs, Phase N-1 fine | Phase N's input CB not filled | Emit order / data flow |
| Hang after adding new op | Missing flags or CT args | Compare against working op's emit() |
| Intermittent hang | Semaphore initial value or race | Check `initial_value` and timing |
| Hang on full grid but not small | NOC congestion or grid mismatch | Test on reduced grid, check ranges |
