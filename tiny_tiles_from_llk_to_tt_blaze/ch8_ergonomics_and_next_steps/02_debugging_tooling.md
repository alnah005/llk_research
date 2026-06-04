# 8.2 Debugging Tooling for Tiny-Tile Programs

This chapter surveys the debugging tools that TT-Blaze already exposes and assesses how much each one tells a developer about *tile geometry* — the dimension that, in tiny-tile programs, is the single most common source of silent corruption and CB-wait deadlocks. Today every existing tool surfaces geometry only *implicitly* (as a page-size, a CT-arg, or a compiled artifact on disk). This section catalogs what is already there, then proposes four targeted enhancements that would surface geometry mismatches within seconds of program compilation rather than hours of post-hoc PCC chasing.

**Prerequisites:** Chapter 1 (tiny-tile geometry and the `(num_faces, face_r_dim, face_c_dim)` triple), Chapter 4 (CB descriptors and the `TileDescriptor` flow on op-emit), Chapter 7 Section 1 (the three-place declaration burden).

---

## 8.2.1 `BLAZE_L1_PROFILE` — CB stats, with geometry implicit in page size

`BLAZE_L1_PROFILE` is the most commonly-reached-for tool when something is wrong with a tiny-tile op. It is plumbed in two places: `program.py:927` invokes `print_cb_stats(self)` from `l1_profile.py` after CB allocation, and `compiler.py:120, 1035` invokes the same routine at the end of the compile path so the dump is available even when the program is never run.

The output is a per-CB table giving, for each CB on each core: the L1 base address, the CB type (input / output / intermediate), the data format (e.g. `Bfp8_b`, `Float16_b`), the page size in bytes, and the number of pages. From these five columns the geometry of the tile is recoverable, but only by manual arithmetic. Consider the Q tile in Flash MLA on DeepSeek V3 B1: tile dtype is `Bfp8_b`, tile shape is 8x32. The page size reported in the profile is the byte size of one such tile: `8 * 32 * 1 byte/elem + (8/16) * 16 bytes shared_exp ≈ 256` bytes per page. A full 32x32 `Bfp8_b` tile would page at roughly 1024 bytes. The 4x ratio is the only signal that the CB carries tiny tiles.

This works for a developer who already knows the answer. It does not work for someone diagnosing a bug. The page-size column is not labeled with units of "(rows × cols)" — only bytes — and any datatype change (BFP8 → FP16) silently doubles every number, hiding tile-geometry signal under format-size noise.

**Proposed enhancement:** add an explicit `tile_shape` column to `print_cb_stats()`. The information is already on hand at the call site — every CB descriptor was built from a `TileDescriptor` (see Chapter 4) which wraps a `ttnn.Tile((H, W))`. Emitting that pair next to the page-size column costs nothing and makes the dump self-describing:

```text
# proposed tile_shape column in BLAZE_L1_PROFILE output
core   cb   addr      type    dtype     tile_shape   page_size   pages
0,0    24   0x10000   input   Bfp8_b    (8, 32)      288          4
0,0    25   0x11200   input   Bfp8_b    (32, 32)     1056         4
0,0    26   0x14600   output  Bfp8_b    (8, 32)      288          2
```

A developer scanning that table would notice immediately when a downstream CB feeding a "full-tile" consumer carries `(8, 32)` pages — the most common tiny-tile pipeline bug.

---

## 8.2.2 `BLAZE_DEBUG_KERNELS` and `named_args_generated.h` — phase markers and resolved CT args

Two other tools that *can* be used for tile-geometry diagnosis are `BLAZE_DEBUG_KERNELS` and direct inspection of `named_args_generated.h`.

### Phase-marker injection

`BLAZE_DEBUG_KERNELS` is parsed in `kernel_codegen.py:130–149` into a `(enabled, preprocessor_guard)` tuple, then threaded into the JIT pipeline at line 264. When enabled, the codegen emits per-RISC debug phase markers around each kernel section: unpack-init, math main loop, pack-out, etc. The markers feed the device-print pipeline so a hang can be localized to a specific phase.

The connection to tile geometry is indirect but important. The classic tiny-tile failure mode is a `cb_wait_front` deadlock caused by a producer pushing N-byte pages while the consumer waits for 4N-byte pages. With `BLAZE_DEBUG_KERNELS` enabled, the device-print log freezes on a marker like `[unpack][cb_wait_front cb=24]` and stays there, identifying not only that the deadlock is in the unpacker but on which specific CB. Once the CB id is known, `BLAZE_L1_PROFILE` (above) can be cross-referenced to see whether the producer and consumer page sizes actually agree.

`BLAZE_DEBUG_KERNELS` does *not* itself print tile geometry. It only narrows the search.

### Resolved compile-time arguments

The deeper question — "what value of `TILE_HEIGHT` (or `FACE_R_DIM`, `NUM_FACES`) did the kernel actually compile with?" — has a definitive answer on disk: `named_args_generated.h`. This header is auto-generated per-kernel by the JIT pipeline and stored in the kernel's build cache. `viz_export.py:756, 771` is the only place in the codebase that *collects* it; everywhere else the header sits silently in the JIT cache directory and is inspectable only if the developer knows where to look.

The header contains a block of `constexpr` definitions for every CT-arg resolved by the op-emit code. For a Flash MLA Q-side kernel this includes the resolved `TILE_HEIGHT = 8`, the SFPU iteration count (if it was passed as a CT-arg rather than hardcoded), and the face dimensions. A geometry mismatch — say, a kernel compiled with `TILE_HEIGHT=32` despite being fed an `(8, 32)` CB — is fully visible here.

**Proposed enhancement:** make the path to `named_args_generated.h` *discoverable* at compile time rather than archeological. Two concrete steps:

1. When `BLAZE_DEBUG_KERNELS` is enabled, log the absolute path of the generated header for each compiled kernel, e.g. `[debug-kernels] resolved CT args for flash_mla_q_unpack -> /tmp/.../named_args_generated.h`. This eliminates the "find it in the JIT cache" step.
2. Add a `--dump-resolved-args` flag (or a complementary `BLAZE_DUMP_NAMED_ARGS` env var) that walks every compiled kernel and prints a one-line summary: kernel name, the subset of CT args that name tile geometry (`TILE_HEIGHT`, `FACE_R_DIM`, `NUM_FACES`), and the page size of every CB the kernel reads or writes.

Both changes are additive and require no kernel-side code change.

---

## 8.2.3 Visualizer — adding tile geometry to CB edges

`viz_export.py` serializes a compiled `BlazeGraph` to JSON for consumption by the HTML visualizer (`visualizer.html`). The exported representation captures each op node and the CB-mediated edges between them. Edge metadata today carries the source op, the sink op, the CB id, and the data format — sufficient to draw the dataflow graph but *silent on tile geometry*. A tiny-tile op feeding a full-tile op renders identically to a full-tile op feeding a full-tile op.

This is the most user-facing of the four tools and arguably the one with the highest leverage. A developer scrolling through the visualizer to understand a new pipeline is already paying attention to edges; surfacing geometry there is the fastest way to make a mismatch visible.

**Proposed enhancement:** extend the JSON edge schema to carry tile geometry, and render it as an edge label in `visualizer.html`. Concretely:

```python
# proposed CB-edge JSON schema (additions marked +)
{
  "src_op":       "flash_mla_unpack_q",
  "sink_op":      "flash_mla_math",
  "cb_id":        24,
  "data_format":  "Bfp8_b",
+ "tile_shape":   [8, 32],
+ "num_faces":    2,
+ "face_r_dim":   4
}
```

The producer-side `TileDescriptor` is already attached to the `CBHandle` (`cb_handle.py:48`, `tile_desc: object`); the exporter simply has to walk to it and emit the three fields. The visualizer can then render the edge as `cb24 Bfp8_b (8x32, 2f)` and color-code edges where producer and consumer geometries disagree. This last hook depends on the lint pass below resolving the consumer-side geometry, but the visualizer side of the work is independent and incremental — the label alone is useful immediately.

A secondary nice-to-have: when a graph contains *any* tiny-tile CB, the visualizer's legend should call it out at the top of the view. Most pipelines are full-tile; the eye is bad at noticing the one tiny-tile edge buried in a 200-node graph.

---

## 8.2.4 Lint pass — flag producer/consumer geometry mismatches

The most consequential addition is a lint pass that runs over the compiled op graph and flags every CB handoff where the producer's `TileDescriptor` and the consumer's resolved `TILE_HEIGHT` (and `NUM_FACES`, `FACE_R_DIM`) disagree.

Today the contract is implicit. The producer op constructs a `TileDescriptor` at op-emit time (e.g. `q_tile_descriptor = ttnn.TileDescriptor(q_tiny_tile)` in Flash MLA `op.py:548`) and stashes it on the `CBHandle` (`cb_handle.py:48`). The consumer op, completely separately, declares its CT args — `TILE_HEIGHT`, `NUM_FACES`, an SFPU iteration count of `<approx_mode, false, 2>` (see `unified_kernels/matmul.hpp:173–176`). Nothing checks that the two agree. The validator `validate_tensor_shape_tile_dependent_ops_` in `tensor_shape.h:87–94` enforces *legal* `(num_faces, face_r_dim)` combinations (`num_faces ∈ {1, 2, 4}`, `face_r_dim ∈ {1, 2, 4, 8, 16}`, `face_c_dim = 16`), but it operates on each tensor independently — it cannot see that a producer's legal `(8, 32)` and a consumer's legal `(32, 32)` are illegal *together* when they share a CB.

The lint pass would close that gap. Sketch of the implementation:

```python
# proposed: blaze/lint/cb_geometry.py
def lint_cb_geometries(program):
    """Walk every CB in the program; check producer/consumer geometry agreement."""
    for cb in program.circular_buffers:
        # Producer-side geometry is on the CBHandle stashed at op-emit time.
        producer_tile = cb.handle.tile_desc.tile_shape  # e.g. (8, 32)

        # Each consumer kernel exposes its resolved CT args via the JIT cache.
        for consumer in cb.consumers:
            ct_args = consumer.resolved_ct_args  # parsed from named_args_generated.h
            consumer_tile = (
                ct_args.get("TILE_HEIGHT", 32),
                ct_args.get("TILE_WIDTH",  32),
            )
            if producer_tile != consumer_tile:
                yield LintError(
                    cb=cb.id,
                    producer=cb.producer.name,
                    consumer=consumer.name,
                    producer_tile=producer_tile,
                    consumer_tile=consumer_tile,
                )
```

The pass should run automatically at the end of compile (just before `print_cb_stats`), with severity controlled by an env var: `BLAZE_LINT_CB_GEOMETRY=error|warn|off`, defaulting to `warn`. A warning during compile is far cheaper than a `cb_wait_front` hang on device, and orders of magnitude cheaper than the silent-corruption case — SFPU iteration count off by 2, garbage left in unprocessed rows, PCC quietly drops by 0.001 and the bug ships.

Three classes of mismatch are worth surfacing distinctly:

1. **Page-size mismatch.** Producer page size ≠ consumer page size in bytes. This always hangs `cb_wait_front`. Severity: error.
2. **Geometry mismatch with matching page size.** Same byte count but different `(H, W)` (e.g. an 8x32 tile vs. a 4x64 tile). Symptom is silent PCC divergence because SFPU iteration counts and face traversal disagree. Severity: error.
3. **SFPU iteration-count mismatch.** Consumer kernel hardcodes `<approx_mode, false, N>` with `N ≠ NUM_FACES` of the producer tile. This is the Flash-MLA failure mode (Chapter 7 Section 1). Severity: error. Requires the kernel body, not just CT args, to be inspected — the template argument is a literal in the kernel source. Static parsing of the kernel C++ at compile time is sufficient; the JIT pipeline already has the source in hand.

The first two checks are graph-walks over `CBHandle`. The third requires the kernel-codegen pass to record the literal template argument used for each SFPU call into `named_args_generated.h` alongside the CT args, then the lint pass cross-references.

---

## 8.2.5 Putting them together

The four enhancements above are independent and each cheap on its own:

| Enhancement                                 | Implementation site              | Touch points                                |
|---------------------------------------------|----------------------------------|---------------------------------------------|
| `BLAZE_L1_PROFILE` tile_shape column        | `l1_profile.py: print_cb_stats`  | Add one column from existing `TileDescriptor` |
| `named_args_generated.h` path logging       | `kernel_codegen.py:130–149, 264` | Print absolute path under `BLAZE_DEBUG_KERNELS` |
| Visualizer tile_shape on CB edges           | `viz_export.py`, `visualizer.html` | Emit three extra JSON fields; render as label |
| `lint_cb_geometries` pass                   | New `blaze/lint/cb_geometry.py`  | Walk CBs; parse `named_args_generated.h`     |

Together they convert tile geometry from an implicit, manually-tracked invariant into a first-class, compile-time-checked property of the program. Every existing tiny-tile bug discovered during the Flash MLA, Deepseek tilize-untilize, and Quasar-matmul work (see Chapters 6 and 7) would have been surfaced within seconds of compilation by some combination of these four checks. None of them require touching the LLK, the kernel ISA, or the hardware. They are pure ergonomics — and the ergonomics is where the friction lives.
