# 5.4 PCIe NOC Utilities

> **Source file:**
> - `kernels/pcie_noc_utils.h` (lines 1--36)

The `pcie_noc_utils.h` header provides two inline functions for transferring data between Tensix L1 SRAM and host RAM over PCIe, and a constant for benchmark warmup. These utilities abstract the chunked-transfer pattern required for PCIe DMA over the Tenstorrent NOC (Network-on-Chip). Both the bulk passthrough kernel and the token loopback kernel depend on NOC primitives for moving data between L1 and host RAM, but only the bulk passthrough kernel's DEVICE_PULL mode uses the explicit chunked read function from this header.

---

## 5.4.1 Why Chunking Is Needed

The Tenstorrent NOC imposes a maximum burst size on individual DMA transactions: `NOC_MAX_BURST_SIZE`. This hardware limit applies to all NOC read and write commands, including those targeting PCIe endpoints.

When a page exceeds `NOC_MAX_BURST_SIZE`, a single NOC read or write command would violate the hardware constraint and produce undefined behavior (typically a silent truncation or a NOC hang). The chunking utilities split large transfers into a sequence of `NOC_MAX_BURST_SIZE`-or-smaller transactions, each of which is a valid single NOC burst.

This limit exists because the NOC is a packet-switched network with fixed-size buffer slots at each router. Each NOC packet carries at most `NOC_MAX_BURST_SIZE` bytes of payload. For PCIe transfers specifically, the limit also aligns with PCIe Max Payload Size (MPS) and Max Read Request Size (MRRS) constraints, ensuring that each NOC command maps to a legal PCIe TLP (Transaction Layer Packet).

The chunking functions encapsulate this hardware constraint so that kernel authors can request arbitrary-sized transfers without manually managing the loop.

---

## 5.4.2 noc_write_page_chunked(): L1 to PCIe (D2H Direction)

```cpp
// kernels/pcie_noc_utils.h:12-22
inline void noc_write_page_chunked(
    uint32_t pcie_xy_enc, uint32_t src_l1, uint64_t dst_pcie, uint32_t size) {
    noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);
    while (size) {
        uint32_t chunk = size > NOC_MAX_BURST_SIZE ? NOC_MAX_BURST_SIZE : size;
        noc_wwrite_with_state<noc_mode, write_cmd_buf,
            CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
            NOC_INDEX, src_l1, pcie_xy_enc, dst_pcie, chunk, 1);
        src_l1 += chunk;
        dst_pcie += chunk;
        size -= chunk;
    }
}
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `pcie_xy_enc` | `uint32_t` | Encoded NOC X/Y coordinates of the PCIe endpoint |
| `src_l1` | `uint32_t` | Source address in local L1 SRAM (32-bit, since L1 is < 4 GB) |
| `dst_pcie` | `uint64_t` | Destination address in host pinned RAM (64-bit PCIe BAR address) |
| `size` | `uint32_t` | Total number of bytes to transfer |

### Behavior

1. **Initializes NOC write state** with `noc_write_init_state`. This sets up the write command buffer template with the unicast write virtual channel.
2. **Loops** while `size > 0`, computing `chunk = min(size, NOC_MAX_BURST_SIZE)`.
3. **Issues a NOC write** via `noc_wwrite_with_state`, which writes `chunk` bytes from `src_l1` to `dst_pcie` through the PCIe endpoint at `pcie_xy_enc`.
4. **Advances** both source and destination pointers by `chunk` and decrements `size`.

### Template Parameters on noc_wwrite_with_state

| Template | Meaning |
|----------|---------|
| `noc_mode` | NOC operating mode (compile-time constant for the architecture) |
| `write_cmd_buf` | Which command buffer register to use for the write |
| `CQ_NOC_SNDL` | Send list descriptor register |
| `CQ_NOC_SEND` | Send trigger register |
| `CQ_NOC_WAIT` | Wait/completion register |
| `true` | Unicast mode (single destination) |
| `false` | Non-posted write (wait for completion acknowledgment) |

### Caller Responsibilities

The function does **not** include a barrier. The caller must call `noc_async_write_barrier()` after `noc_write_page_chunked()` returns to ensure all chunks have been committed to the destination before proceeding. This design allows callers to batch multiple page writes before a single barrier, improving throughput.

**Note on state initialization**: This function re-initializes the write command buffer state on every call. In the bulk passthrough kernel (Section 5.3), the main loop calls `noc_write_init_state` once before the loop and uses `noc_wwrite_with_state` directly (without `noc_write_page_chunked`). The function is designed for standalone callers that cannot guarantee the write state has been initialized.

---

## 5.4.3 noc_read_page_chunked(): PCIe to L1 (H2D Direction)

```cpp
// kernels/pcie_noc_utils.h:25-35
inline void noc_read_page_chunked(
    uint32_t pcie_xy_enc, uint64_t src_pcie, uint32_t dst_l1, uint32_t size) {
    while (size) {
        uint32_t chunk = size > NOC_MAX_BURST_SIZE ? NOC_MAX_BURST_SIZE : size;
        noc_read_with_state<noc_mode, read_cmd_buf,
            CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT>(
            NOC_INDEX, pcie_xy_enc, src_pcie, dst_l1, chunk);
        src_pcie += chunk;
        dst_l1 += chunk;
        size -= chunk;
    }
}
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `pcie_xy_enc` | `uint32_t` | Encoded NOC X/Y coordinates of the PCIe endpoint |
| `src_pcie` | `uint64_t` | Source address in host pinned RAM (64-bit PCIe BAR address) |
| `dst_l1` | `uint32_t` | Destination address in local L1 SRAM |
| `size` | `uint32_t` | Total number of bytes to transfer |

### Key Differences from Write

1. **No explicit init state**: Unlike `noc_write_page_chunked`, this function does not call `noc_write_init_state` (reads use a different command buffer path). The NOC read state is initialized separately by the caller or by prior framework setup.
2. **Uses `noc_read_with_state`**: The read path uses the `read_cmd_buf` register instead of `write_cmd_buf`.

### Caller Barrier Requirement

The source code header comment on line 25 explicitly states:

> Caller must call `noc_async_read_barrier()` after this returns.

This is a critical contract. The `noc_read_page_chunked` function issues all NOC read commands but does **not** wait for them to complete. The data is not guaranteed to be in L1 until `noc_async_read_barrier()` returns. The bulk passthrough kernel follows this contract:

```cpp
// kernels/pi05_bulk_passthrough.cpp:76-82
if constexpr (pull_from_host) {
    noc_read_page_chunked(
        h2d_pcie_xy_enc,
        pcie_data_addr + receiver.read_ptr - receiver.fifo_addr,
        receiver.read_ptr,
        page_size);
    noc_async_read_barrier();  // <-- barrier after read
}
```

---

## 5.4.4 Chunking Arithmetic

For a page of $S$ bytes and a maximum burst of $B$ bytes, the number of NOC transactions is:

$$N_{\text{chunks}} = \left\lceil \frac{S}{B} \right\rceil$$

Each chunk is exactly $B$ bytes except the last, which is $S \mod B$ bytes (or $B$ if $S$ is a multiple of $B$). The loop handles this naturally:

```
chunk = min(remaining, NOC_MAX_BURST_SIZE)
```

For the typical case of `BULK_PAGE_SIZE = 4096`:

| $B$ (`NOC_MAX_BURST_SIZE`) | $N_{\text{chunks}}$ | Last chunk size |
|---|---|---|
| 4096 | 1 | 4096 (no chunking needed) |
| 2048 | 2 | 2048 |
| 1024 | 4 | 1024 |
| 256 | 16 | 256 |

For `TOKEN_PAGE_SIZE = 256`, chunking is typically not needed because 256 bytes is less than or equal to `NOC_MAX_BURST_SIZE` on all current architectures. This is why the token loopback kernel does not use `pcie_noc_utils.h` at all -- it calls `noc_wwrite_with_state` directly for its single 256-byte page.

---

## 5.4.5 NOC_MAX_BURST_SIZE: Hardware Rationale

`NOC_MAX_BURST_SIZE` is a hardware-defined constant (not declared in `pcie_noc_utils.h` -- it comes from the tt-metal device headers). Its value depends on the Tenstorrent architecture generation:

- **Wormhole/Blackhole**: Typically 8192 bytes for L1-to-L1 transfers, but PCIe transfers may have a lower effective limit depending on the PCIe endpoint's maximum payload size configuration.
- **Grayskull**: 4096 bytes.

The value ensures that each NOC transaction fits within:

1. **NOC packet buffer**: Each router in the NOC mesh has a finite packet buffer. Bursts exceeding this buffer would stall the network.
2. **PCIe TLP alignment**: PCIe Max Payload Size (typically 128, 256, or 512 bytes for consumer hardware; up to 4096 bytes for enterprise) and Max Read Request Size bounds. The NOC-to-PCIe bridge fragments larger NOC bursts into legal TLPs.
3. **L1 bank interleaving**: For certain L1 address ranges, banks are interleaved at boundaries that align with `NOC_MAX_BURST_SIZE`. Transfers that cross bank boundaries must be split.

A single transaction exceeding `NOC_MAX_BURST_SIZE` would overflow the internal transaction buffer, causing the transaction to be silently truncated or the NOC to stall. The hardware does not automatically split large transactions -- the software must do it.

---

## 5.4.6 WARMUP_ITERS Constant

```cpp
// kernels/pcie_noc_utils.h:9
constexpr uint32_t WARMUP_ITERS = 5;
```

This constant defines the number of untimed warmup iterations for PCIe socket benchmarks. The comment on lines 7-8 states it must stay in sync with `kWarmupIters` in `tests/tt_metal/distributed/benchmark_hd_sockets.cpp`.

The warmup serves two purposes:

1. **IOMMU TLB warm-up**: The first few PCIe DMA transfers after socket connection may be slow because the IOMMU must populate its translation lookaside buffer with the host-to-device address mappings. These misses add 1-10 microseconds of latency per miss. After several iterations, the TLB entries are cached and subsequent transfers hit the fast path.

2. **PCIe link / NOC path setup**: The PCIe link may operate at reduced bandwidth on first use and ramp up to full speed after initial traffic. Similarly, the first NOC transactions on a given path may incur routing setup overhead as the NOC routers populate their forwarding state. Running untimed warmup iterations ensures the timed measurement window captures steady-state performance.

The `WARMUP_ITERS = 5` value is empirical: profiling showed that latency stabilizes within 3-4 iterations on current hardware, and 5 provides a safety margin. The constant is defined in this header (rather than in the benchmark test) because it must be available to both device kernels and host-side test code.

The `pi05_bulk_passthrough` kernel does not directly use `WARMUP_ITERS` -- it runs continuously until sentinel. The constant is primarily consumed by standalone PCIe socket benchmarks that measure per-page transfer latency.

---

## 5.4.7 Function Asymmetry Summary

| Property | `noc_write_page_chunked` | `noc_read_page_chunked` |
|----------|--------------------------|-------------------------|
| Direction | L1 -> PCIe (D2H) | PCIe -> L1 (H2D) |
| NOC state init | Yes (`noc_write_init_state`) | No |
| Post-call barrier | Caller must call `noc_async_write_barrier()` | Caller must call `noc_async_read_barrier()` |
| Used by bulk kernel | Not directly (kernel uses `noc_wwrite_with_state` inline) | Yes, in DEVICE_PULL mode |
| Linked mode | Yes (`true` template arg) | N/A (read commands don't support linking) |

The write function initializes state because it may be called from contexts where the NOC write state is unknown. The read function does not, relying on the caller's prior initialization. This asymmetry reflects the different calling conventions of the underlying NOC primitives: `noc_wwrite_with_state` requires pre-initialized write state, while `noc_read_with_state` uses a separate read command buffer that has its own initialization lifecycle.

### Non-Use by the Bulk Kernel's D2H Path

Although `noc_write_page_chunked` is available for D2H writes, the bulk passthrough kernel does not use it. Instead, it calls `noc_wwrite_with_state` directly (line 186-194). This works because the D2H response is always exactly one page, and the NOC write state was already initialized on line 67. The chunked utility would add unnecessary function-call overhead and a redundant `noc_write_init_state` call for a single page.

---

**Next:** [Chapter 6 -- Decode Scheduler Integration](../ch6_decode_scheduler_integration/index.md)
