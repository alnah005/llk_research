# Multi-Device CCL and Fabric: Tiny Tiles Across Chip Boundaries

When a tiny-tile activation leaves a chip via `ccl_all_reduce`, `ccl_broadcast`, `d2d_exchange`, or `reduce_to_one_b1`, it stops being a private LLK concern and becomes a fabric-transport concern: a frame whose payload size is determined by tile geometry, but whose routing, semaphore plumbing, and alignment are not. This file separates what tiny tiles change (page size, header-to-payload ratio, alignment padding risk) from what they do not (routing topology, semaphore counts, fabric link selection), and grounds each claim in the relevant Blaze and CCL guide code.

**Prerequisites:** Chapter 1 (tiny-tile definition), Chapter 4 (TT-Metal tile metadata / `TileDescriptor`), and the sibling file `01_cross_architecture_support.md` in this chapter. For deeper fabric mechanics, see [[comprehensive_guide_to_creating_new_micro_ops_in_tt_blaze]] Ch. 6 — this file is the tiny-tile-specific overlay on top of that material.

---

## 1. Fabric Packet Structure and Alignment

A fabric packet on Tenstorrent mesh hardware is structurally `header || payload`, where the header carries routing metadata (destination FabricNodeId, link index, flow control bits) and the payload carries one or more tiles worth of activation data. The packet size that the fabric layer actually transmits is `align(header_bytes + payload_bytes, fabric_alignment)`. For tiny-tile activations, two of those three terms collapse:

- The header is a fixed-cost constant determined by the fabric protocol, not the payload.
- The payload is `tile.get_tile_size(dtype) * tiles_per_frame`. For an 8x32 BFP8 tile this is ~512 bytes; for canonical 32x32 BFP8 it is 2048 bytes (`tt-blaze/blaze/fused_program.py:125-136`, where `tile.get_tile_size(dtype)` is computed from height, width, and dtype).
- The alignment boundary is determined by the underlying transport. The PCIE-facing path uses `PCIE_ALIGNMENT = 64` (`tt-blaze/blaze/socket_config.py:14`), and the fabric layer typically aligns to a multiple of that — 64 or 128 bytes, depending on the link.

The consequence is that a tile payload that is itself a clean multiple of 64 bytes will not be padded. But a tile payload that lands awkwardly (for example, 1x32 BFP8 ≈ 128 bytes — fine; but 1x32 in float16 plus a non-aligned header offset, or a custom datum-per-face arrangement) can be rounded up. Padding does not corrupt data — the kernel reads the canonical payload region and ignores the trailing bytes — but it does eat fabric bandwidth.

---

## 2. CCL Op Routing Is Independent of Tile Size

The mesh-routing logic in CCL ops does not inspect tile geometry. From `tt-blaze/blaze/ops/ccl_broadcast/op.py:40-113`, `compute_routing()` derives sender/receiver flags and fabric link indices per device coordinate by walking the mesh topology:

```python
# tt-blaze/blaze/ops/ccl_broadcast/op.py:40-81 (excerpt)
def compute_routing(mesh_shape, ...):
    # Returns per-coordinate (is_sender, link_indices) tuples.
    # No reference to tile shape, num_faces, or face_r_dim.
    ...
```

Likewise, `compute_dst_nodes()` (lines 84-113) resolves `FabricNodeId` destinations from mesh coordinates. Tile geometry never enters the topology graph. The same routing table services a tiny-tile broadcast and a canonical-tile broadcast — what differs is only the per-frame size that the fabric stamps into each packet header.

The practical implication: porting a CCL op to support tiny tiles requires no changes to `compute_routing()` or `compute_dst_nodes()`. The change surface is the page-size / payload-size declarations that flow into `setup_fabric()` and the socket config (next two sections).

---

## 3. Fabric Connection Setup via `setup_fabric()`

`setup_fabric(fp, prefix, worker_core, risc, dst_nodes, link_indices)` (covered in [[comprehensive_guide_to_creating_new_micro_ops_in_tt_blaze]] Ch. 6.3) allocates the semaphores and runtime args that a CCL kernel needs to drive one fabric connection. It returns a `fabric_arg_idx` pointing into the worker's runtime-arg block.

Crucially, what `setup_fabric()` allocates is **per-connection**, not **per-frame**:

- One teardown semaphore per destination device.
- One buffer-index semaphore per destination device.
- Runtime args encoding `FabricNodeId`s and link indices.

Tile geometry does not enter any of these. A worker that broadcasts 100 tiny 8x32 tiles uses the same semaphore count as a worker that broadcasts 100 full 32x32 tiles to the same destinations. The fabric kernel at runtime computes destination addresses from `(base_addr, frame_index, frame_size)` triples; only `frame_size` reflects the tile geometry, and it is passed as a CT or RT arg derived from `tile.get_tile_size(dtype)`.

This is why tiny-tile CCL support is a low-touch change at the op level: the structural plumbing — semaphores, routing args, link selection — is geometry-blind.

---

## 4. D2D Socket Page Size Configuration

D2D exchange (`d2d_exchange`) uses the socket layer rather than raw fabric writes. The socket layer requires an explicit page size, and this is the one place where the producer's tile geometry must be propagated correctly. From `tt-blaze/blaze/socket_config.py:18-24`:

```python
# tt-blaze/blaze/socket_config.py:18-24 (excerpt)
@dataclass
class SocketConfig:
    page_size_bytes: int          # MUST match producing stream's tile geometry
    ...
```

The page size is `tile.get_tile_size(dtype) * pages_per_device_shard`. For 8x32 BFP8 with 18 pages per shard, that is `512 * 18 = 9216` bytes. For the canonical case (32x32 BFP8, same page count) it would be `2048 * 18 = 36864`. If a downstream consumer is configured with the canonical 36864-byte page while the producer emits 9216-byte pages, the socket will silently misframe — the consumer will read one frame that spans four producer frames, with shape-incoherent data.

The propagation path is: producer micro-op declares its output `tile_desc` (an `ndarray.TileDescriptor`-compatible object on `CBHandle.tile_desc` — see `tt-blaze/blaze/cb_handle.py:48`); the pipeline builder reads `tile_desc.tile_shape` from the producer's `CBHandle` and feeds it to the downstream `SocketConfig`. If this propagation is skipped (for example, by hardcoding `page_size_bytes = canonical_tile_bytes * pages`), the D2D exchange will deserialize incorrectly.

The same is true for `reduce_to_one_b1` and any other op that materializes a socket between devices.

---

## 5. Bandwidth and Header Overhead Implications

The header-to-payload ratio matters in proportion to how much fabric link bandwidth a given workload uses. The table below summarizes a representative BFP8 case, assuming a 32-byte fabric header and 64-byte alignment:

| tile geometry | tile_bytes (BFP8) | alignment | padded_packet_size | header_ratio |
|---------------|-------------------|-----------|--------------------|--------------|
| 32x32         | 2048              | 64        | 2080               | 1.6%         |
| 16x32         | 1024              | 64        | 1056               | 3.1%         |
| 8x32          | 512               | 64        | 576 (header padded)| 6.3% effective, ~11% with alignment slack |
| 4x32          | 256               | 64        | 320                | ~12.5% effective |
| 1x32          | 128               | 64        | 192                | ~25% effective |

Two things are worth pulling out:

- The **effective header ratio** rises non-linearly as the tile shrinks, because (a) the header is fixed and (b) alignment slack consumes a larger fraction of small payloads.
- Below roughly 4x32, alignment padding starts to dominate the calculus. A 1x32 BFP8 packet at 128 bytes payload + 32-byte header + alignment to 64 lands at 192 bytes — meaning that 33% of the wire bytes carry no useful data. This is when alignment becomes "problematic" in the sense of being structurally lossy, not merely a small constant overhead.

The mitigation is **frame batching**: most fabric transports support multiple tiles per packet. If a CCL op can pack four 8x32 tiles into one frame, the per-frame header is amortized and the effective ratio recovers to roughly the 32x32 case. Whether a given micro-op exploits this is a property of its kernel — see [[comprehensive_guide_to_creating_new_micro_ops_in_tt_blaze]] Ch. 6.4 for the per-frame batching pattern.

The other axis — absolute throughput — is independent of header overhead. A tiny-tile stream moves less data per frame, so if a worker is producer-bound by core compute (i.e., the LLK is the bottleneck), the fabric link is not saturated and header overhead is irrelevant. Header overhead matters only when fabric bandwidth is the gating resource, which is most common in dense all-reduce or all-gather phases.

---

## 6. Practical Checklist When Adding Tiny-Tile CCL Support

1. **Do not modify routing.** `compute_routing()` / `compute_dst_nodes()` are tile-blind. Reuse them as-is.
2. **Do not modify semaphore allocation.** `setup_fabric()` is per-connection. Tile changes do not introduce or remove semaphores.
3. **Propagate `tile_desc` through `CBHandle.tile_desc`** (`tt-blaze/blaze/cb_handle.py:48`) so downstream sockets pick up the smaller page size.
4. **Set `SocketConfig.page_size_bytes`** from `tile.get_tile_size(dtype) * pages_per_shard`, not from a hardcoded canonical constant.
5. **Estimate header overhead** for the smallest tile you plan to ship. If the padded packet is more than ~15% header+padding, consider whether the op should batch multiple tiles per frame, or whether the tiny-tile choice is worth the bandwidth tax for this particular boundary.
6. **Verify on a two-device mesh** before assuming a multi-device run will work. The single-device path uses a different transport (intra-chip NoC) and will not surface socket page-size mismatches.

The summary: tiny tiles change the size of each frame the fabric carries; they do not change the structure of the fabric itself. The fragile coupling point is `SocketConfig.page_size_bytes` — get that right by deriving it from the producer's `tile_desc`, and everything else above the wire is unchanged.
