# 7.2 -- Bulk Transfer Metrics

Bulk transfer metrics track the performance of real PCIe-based H2D (Host-to-Device) and D2H (Device-to-Host) data transfers. These metrics are computed in socket and hybrid modes only -- simulation mode does not involve real hardware transfers.

---

## 7.2.1 The `BulkTransferMetrics` Struct

All raw transfer measurements are accumulated in a single struct:

```cpp
// pi05_pipeline_runner.cpp:241-246
struct BulkTransferMetrics {
    std::vector<double> h2d_transfer_us;
    std::vector<double> d2h_transfer_us;
    uint64_t total_h2d_bytes = 0;
    uint64_t total_d2h_bytes = 0;
};
```

| Field | Type | Description |
|-------|------|-------------|
| `h2d_transfer_us` | `vector<double>` | Per-transfer H2D duration in microseconds |
| `d2h_transfer_us` | `vector<double>` | Per-transfer D2H duration in microseconds |
| `total_h2d_bytes` | `uint64_t` | Cumulative H2D bytes sent (including headers) |
| `total_d2h_bytes` | `uint64_t` | Cumulative D2H bytes received (including headers) |

A single instance `bulk_metrics` is created at line 938 and populated throughout the benchmark loop. The `_us` vectors collect one sample per bulk transfer operation.

---

## 7.2.2 H2D Transfer Measurement

Each H2D transfer sends a `BulkH2DHeader` (8 bytes) followed by pixel data and optional action history. The timing is captured inside `BulkH2DChannel::send()`:

```cpp
// pi05_pipeline_runner.cpp:536-542
auto start = Clock::now();
socket_->write(staging_.data(), num_pages);
auto end = Clock::now();

return std::chrono::duration<double, std::micro>(end - start).count();
```

The returned microsecond duration is then accumulated:

```cpp
// pi05_pipeline_runner.cpp:993-995 (initial turn -- no action history)
bulk_metrics.h2d_transfer_us.push_back(h2d_us);
bulk_metrics.total_h2d_bytes += sizeof(BulkH2DHeader) + pixel_data.size();
```

For subsequent turns, action history is also included:

```cpp
// pi05_pipeline_runner.cpp:1105-1107 (subsequent turns -- includes action history)
bulk_metrics.h2d_transfer_us.push_back(h2d_us);
bulk_metrics.total_h2d_bytes += sizeof(BulkH2DHeader) + pixel_data.size()
    + pipeline_config.pixel.action_history_bytes;
```

**Payload sizes by turn** (with default `PixelPayloadConfig`):

| Turn | Payload composition | Size with defaults |
|------|--------------------|--------------------|
| First | `BulkH2DHeader(8B)` + pixel data | $8 + 224 \times 224 \times 3 \times 3 = 451{,}592$ bytes |
| Subsequent | `BulkH2DHeader(8B)` + pixel data + action history | $8 + 451{,}584 + 4{,}096 = 455{,}688$ bytes |

---

## 7.2.3 D2H Transfer Measurement

Each D2H transfer reads one page containing a `BulkD2HHeader` (8 bytes) followed by `ACTION_OUTPUT_BYTES` (3,200 bytes) of bfloat16 action output. The timing is captured inside `BulkD2HChannel::recv()`:

```cpp
// pi05_pipeline_runner.cpp:593-605
auto start = Clock::now();
socket_->read(recv_buf_.data(), num_pages);
auto end = Clock::now();
...
return std::chrono::duration<double, std::micro>(end - start).count();
```

The spin-wait time in `has_data()` polling is deliberately **excluded** from the transfer duration -- only the actual `read()` call is timed:

```cpp
// pi05_pipeline_runner.cpp:579-591
while (!socket_->has_data()) {
    if (++spin_iters % 10000000 == 0) {
        auto elapsed = std::chrono::duration<double>(Clock::now() - spin_start).count();
        if (elapsed > 5.0) {
            std::cerr << "WARNING: D2H recv spin-wait exceeded ..."
```

Accumulation:

```cpp
// pi05_pipeline_runner.cpp:1072-1073
bulk_metrics.d2h_transfer_us.push_back(d2h_us);
bulk_metrics.total_d2h_bytes += sizeof(BulkD2HHeader) + ACTION_OUTPUT_BYTES;
```

**D2H payload size** is constant:

| Component | Size |
|-----------|------|
| `BulkD2HHeader` | 8 bytes |
| `ACTION_OUTPUT_BYTES` | 3,200 bytes ($50 \times 32$ bfloat16 values) |
| **Total** | **3,208 bytes** |

Since $3{,}208 < 4{,}096$ (`BULK_PAGE_SIZE`), D2H always reads exactly one page.

---

## 7.2.4 Bandwidth Computation

At the end of the benchmark, `compute_stats()` aggregates the per-transfer microsecond vectors, and bandwidth is derived from the mean transfer time:

```cpp
// pi05_pipeline_runner.cpp:1168-1174
auto h2d_stats = compute_stats(bulk_metrics.h2d_transfer_us);
auto d2h_stats = compute_stats(bulk_metrics.d2h_transfer_us);
uint32_t h2d_payload = sizeof(BulkH2DHeader) + pipeline_config.pixel.total_bytes();
double h2d_bw_gbps = (h2d_stats.mean > 0)
    ? (static_cast<double>(h2d_payload) / (h2d_stats.mean * 1000.0)) : 0.0;
double d2h_bw_gbps = (d2h_stats.mean > 0)
    ? (static_cast<double>(sizeof(BulkD2HHeader) + ACTION_OUTPUT_BYTES) / (d2h_stats.mean * 1000.0)) : 0.0;
```

**H2D bandwidth formula:**

$$BW_{\text{H2D}} = \frac{\text{h2d\_payload bytes}}{\text{mean\_h2d\_us} \times 1000} \quad \text{(GB/s)}$$

The denominator converts microseconds to nanoseconds ($\mu s \times 1000 = ns$), so bytes/nanosecond = GB/s.

Breaking down with default pixel config:

- `h2d_payload` = `sizeof(BulkH2DHeader)` + `pixel.total_bytes()` = $8 + (224 \times 224 \times 3 \times 3) + 4096 = 455{,}688$ bytes
- Note: This uses `total_bytes()` which includes `action_history_bytes`, even though the first turn does not send action history. The bandwidth figure is therefore an approximation using the maximum payload size.

**D2H bandwidth formula:**

$$BW_{\text{D2H}} = \frac{\text{sizeof(BulkD2HHeader)} + \text{ACTION\_OUTPUT\_BYTES}}{\text{mean\_d2h\_us} \times 1000} \quad \text{(GB/s)}$$

With the constant D2H payload of 3,208 bytes:

- At 100 us mean transfer time: $3208 / (100 \times 1000) = 0.032$ GB/s
- At 10 us mean transfer time: $3208 / (10 \times 1000) = 0.321$ GB/s

**Division-by-zero guard:** Both formulas check `mean > 0` and return 0.0 if no samples exist.

---

## 7.2.5 The `pcie_bandwidth_gbps` Caveat

The `SocketConfigParams` struct contains a field that appears related to bandwidth:

```cpp
// pi05_pipeline_runner.cpp:224
double pcie_bandwidth_gbps = 16.0;
```

**This field is informational ONLY.** It is parsed from the config JSON's `"socket"."pcie_bandwidth_gbps"` key (line 434) but is **never used** in any bandwidth calculation or transfer operation. The actual measured bandwidth derives entirely from the timed `write()`/`read()` calls as shown above. The `pcie_bandwidth_gbps` config value exists as a reference annotation for the engineer -- it documents the theoretical PCIe link speed but plays no functional role.

---

## 7.2.6 Socket vs. Hybrid Mode: Identical Bulk Metrics

Both socket modes -- `"full"` (SocketPipeline for tokens + real bulk sockets) and `"bulk_only"` (PipelineSimulator for tokens + real bulk sockets) -- use the exact same `BulkTransferMetrics` collection and reporting code. The only difference is which backend drives the token pipeline:

| Aspect | `mode: "full"` | `mode: "bulk_only"` |
|--------|----------------|---------------------|
| Token pipeline | `SocketPipeline` (real device) | `PipelineSimulator` (CPU simulation) |
| Bulk H2D/D2H | Real `H2DSocket`/`D2HSocket` | Real `H2DSocket`/`D2HSocket` |
| Bulk metrics | Identical collection | Identical collection |
| Bandwidth output | Same format | Same format |
| Output label | `"Socket"` | `"Hybrid"` |

The `mode_label` variable controls the output header:

```cpp
// pi05_pipeline_runner.cpp:1176
const char* mode_label = bulk_only ? "Hybrid" : "Socket";
```

This design allows measuring real PCIe bulk transfer performance even when the decode pipeline is simulated -- which is the purpose of the hybrid ("bulk_only") mode.

---

## 7.2.7 Sample Count Considerations

The number of bulk transfer samples depends on the benchmark configuration:

$$N_{\text{H2D}} = N_{\text{D2H}} = \text{num\_users} \times \text{max\_turns}$$

Each user generates one H2D transfer and one D2H transfer per turn. With the default 4 users and 3 turns, there are 12 samples of each -- subject to the same P99 gotcha described in Section 7.1.7 (P99 collapses to the maximum for $n < 100$).

---

**Next:** [Output Format](03_output_format.md)
