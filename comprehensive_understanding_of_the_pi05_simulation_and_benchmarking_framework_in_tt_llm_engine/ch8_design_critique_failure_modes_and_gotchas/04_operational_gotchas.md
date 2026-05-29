# 8.4 -- Operational Gotchas

This section is a quick-reference checklist of nine practical issues that operators encounter when running the Pi0.5 benchmark framework. Each gotcha has bitten at least one developer. Where a topic overlaps with Sections 8.1 or 8.2, the entry provides a brief symptom and fix with a cross-reference to the full analysis; unique gotchas (8.4.3, 8.4.6, 8.4.8) retain their full treatment here.

---

## 8.4.1 /dev/shm Socket Files Must Be Cleaned

**Symptom:** The runner hangs during "Waiting for socket descriptors" or connects to a dead IOMMU mapping.

**Quick fix:** Always clean before running: `rm -f /dev/shm/tt_d2h_* /dev/shm/tt_h2d_* /dev/shm/tt_socket_manifest_* /dev/shm/pi05_*`, or use `run_hybrid.sh` which does this automatically. There is no `--clean` flag on the runner.

For full root-cause analysis (why `_exit(EXIT_SUCCESS)` prevents self-cleanup, stale descriptor mechanics, and residual risks), see [Section 8.2.2](02_failure_modes.md#822-devshm-cleanup-stale-descriptors-persist). For the sentinel-related consequence of `_exit()`, see [Section 8.2.6](02_failure_modes.md#826-sentinel-duplication-two-sentinels-corrupt-next-run).

---

## 8.4.2 TT_VISIBLE_DEVICES Must Be Set Correctly

**Symptom:** Segfault or `std::out_of_range` crash during tt-metal topology discovery.

**Quick fix:** Set `TT_VISIBLE_DEVICES` to the specific device index before running:

```bash
export TT_VISIBLE_DEVICES=0    # or whichever device is available
```

**Key gotcha:** `run_hybrid.sh` defaults to device `1`, not `0`. If your machine has only one N150, override with `TT_VISIBLE_DEVICES=0 ./examples/run_hybrid.sh`. The parent/child asymmetry (parent defaults to 1, child defaults to 0 in bulk-only mode) can cause device targeting mismatches.

For full analysis of the topology discovery crash, defensive code paths, and residual risks, see [Section 8.2.5](02_failure_modes.md#825-tt_visible_devices-interaction-topology-discovery-crash).

---

## 8.4.3 TT_METAL_RUNTIME_ROOT Must Point to tt-metal Source

**Symptom:** Runtime error mentioning missing kernel files, or "cannot find compiled kernel" errors from tt-metal:

```
Cannot open kernel source file: <path>/kernels/pi05_bulk_passthrough.cpp
```

**Root cause:** tt-metal locates its kernel binaries and runtime files relative to `TT_METAL_RUNTIME_ROOT`. If this variable is unset or points to the wrong directory, kernel compilation and loading fail. The kernel path is constructed from `DS_KERNEL_DIR` (a compile-time define, defaulting to `"kernels"`) relative to the runtime root. The `#ifndef DS_KERNEL_DIR` guard in the launcher (line 44-46) allows overriding at build time, but the default assumes the kernel source is at `$TT_METAL_RUNTIME_ROOT/kernels/`.

**Fix:** Set the variable to your tt-metal checkout:

```bash
export TT_METAL_RUNTIME_ROOT=/path/to/tt-metal
```

The `run_hybrid.sh` script hardcodes this path (line 11):

```bash
# examples/run_hybrid.sh:11
TT_METAL_RUNTIME_ROOT=/localdev/salnahari/testing_dir/tt-metal \
```

This path is machine-specific and must be updated when the script is used on a different machine or with a different tt-metal checkout. For out-of-tree builds using `DS_KERNEL_DIR`, set it to the absolute path of the kernel directory.

---

## 8.4.4 PCIe Alignment (4096 Bytes)

**Symptom:** Silent data corruption in H2D or D2H transfers, or PCIe transaction errors in `dmesg`.

**Quick fix:** Never change `BULK_PAGE_SIZE` in one file without changing it in all three:

1. `examples/pi05_pipeline_runner.cpp` (host-side structs, lines 205-215)
2. `kernels/pi05_bulk_passthrough.cpp` (kernel-side header parsing, lines 88-89)
3. `examples/pi05_device_launcher.cpp` (page size and FIFO size calculation)

Similarly, any change to `BulkH2DHeader` or `BulkD2HHeader` field layout must be synchronized across all three files, as the kernel reads `slot_id` and `payload_length` from fixed word offsets.

For full analysis of alignment violation failure modes (IOMMU faults, partial writes, incorrect BAR offsets), see [Section 8.2.4](02_failure_modes.md#824-pcie-alignment-violations).

---

## 8.4.5 Process Cleanup: SIGKILL + Sleep

**Symptom:** New runs fail with "device busy" errors because the previous run's processes are still holding the device.

**Quick fix:** Run the `run_hybrid.sh` cleanup sequence before each run: `pkill -9 -f pi05_device_launcher; pkill -9 -f pi05_pipeline_runner_device; sleep 2`. `SIGKILL` is required because `SIGTERM` may be blocked by tt-metal calls. Do not reduce the sleep below 2 seconds; on heavily loaded systems, increase to 3-4 seconds.

For the full analysis of zombie processes, `PR_SET_PDEATHSIG` race conditions, and why `SIGKILL` prevents clean MeshDevice teardown, see [Section 8.2.1](02_failure_modes.md#821-zombie-processes-after-runner-crash).

---

## 8.4.6 output_tokens=1 Confusion

**Symptom:** The benchmark reports extremely low throughput, strange timing numbers, or operators misunderstand what `output_tokens=1` means.

**Root cause:** The default Pi0.5 config sets `output_tokens: 1`:

```json
// examples/pi05_config_hybrid.json:5
"output_tokens": 1,
```

This is *correct for Pi0.5* but surprising if you come from LLM inference. Pi0.5 is a VLA (Vision-Language-Action) model -- each inference produces a single action tensor, not a sequence of text tokens. The `output_tokens=1` value means:

- The `DecodeScheduler` generates exactly 1 token per turn.
- The throughput metric "tok/s" reports output tokens per second, which with `output_tokens=1` equals the frame rate (turns per second). For the benchmark config with 400 turns and 1 user, the total token count is 400, not thousands.
- The `max_new_tokens` field in the `ISRequest` is set to `max_decode`, which comes from `output_tokens` (line 1004).
- The `TPOT` (time per output token) metric is **undefined** when `output_tokens=1` because the formula requires at least 2 tokens per turn:

```cpp
// pi05_pipeline_runner.cpp:1081-1084
if (u.turn_token_count > 1) {
    double tpot_ms = std::chrono::duration<double, std::milli>(
        u.last_token_time - u.first_token_time).count() / (u.turn_token_count - 1);
    tpot_samples.push_back(tpot_ms);
}
```

With `output_tokens=1`, `turn_token_count` never exceeds 1, so `tpot_samples` remains empty and the TPOT stats show `Mean=0.00 Median=0.00 P99=0.00`.

If you set `output_tokens > 1` for experimentation, be aware that D2H slot_id mismatches become more likely because completion order can diverge from submission order. See [Section 8.1.6](01_design_critique.md#816-d2h-slot_id-mismatch-logged-as-warning) for the full analysis of the mismatch and its cascading corruption consequences.

**Fix:** For Pi0.5 benchmarking, always use `output_tokens=1`.

---

## 8.4.7 D2H FIFO Sizing for Deadlock Prevention

**Symptom:** The runner hangs in the D2H `recv()` spin-wait while the device kernel is also blocked on `socket_reserve_pages`.

**Quick fix:** The D2H FIFO is sized to `BULK_PAGE_SIZE * num_users` (one page per user). If you increase `num_users` significantly (e.g., 64), monitor L1 usage -- `64 * 4096 = 256 KB` consumes 17% of a Tensix core's 1.5 MB L1. The minimum safe D2H FIFO size is `2 * BULK_PAGE_SIZE` (double-buffering), but this requires the host to drain D2H results without backpressure. Note that the framework validates H2D FIFO sizing at startup but does **not** validate D2H FIFO sizing.

For the full deadlock analysis (producer-consumer mechanics, `output_tokens > 1` ordering risks, and the H2D payload-exceeds-FIFO failure), see [Section 8.2.7](02_failure_modes.md#827-fifo-size-underrun-payload-exceeds-fifo_size).

---

## 8.4.8 stagger_us Auto-Calculation

**Symptom:** Unexpected TTFT values or uneven user submission timing.

**Root cause:** When `--stagger-us` is not specified, the framework computes it automatically:

```cpp
// pi05_pipeline_runner.cpp:831-832
const uint32_t stagger_us = stagger_explicit ? stagger_us_cli
    : static_cast<uint32_t>(total_latency_us / num_users);
```

For the default config with `total_latency_us = 42130` and `num_users = 8`:

$$\text{stagger\_us} = \lfloor \frac{42130}{8} \rfloor = 5266 \text{ us}$$

This spaces users 5.266 ms apart, so the last user's submission starts at $7 \times 5.266 = 36.9$ ms after the first user. The intent is to fill the pipeline evenly: each user enters the pipeline one "slot width" after the previous user.

**Gotchas:**

1. **Optimal only when `num_users <= total_stages`.** The formula divides *total latency* by *num_users*, which spaces users across the entire pipeline. When `num_users > total_stages` (e.g., 64 users with 54 stages), the stagger becomes $42130 / 64 = 658$ us, which is less than the bottleneck stage duration (2200 us for vision). Users pile up at the vision bottleneck, creating contention that the stagger was meant to avoid.

2. **Stagger is added to measured TTFT.** The measured TTFT for user $k$ includes $k \times \text{stagger\_us}$ of intentional waiting. The corrected TTFT formula attempts to account for this, but only as an average.

3. **Stagger interacts with H2D transfer time.** In socket mode, the stagger sleep happens **between** H2D transfers (line 985-986). The effective stagger is `stagger_us + h2d_transfer_time`. For large payloads, the H2D time may dominate the stagger, making the `--stagger-us` parameter misleading.

4. **Integer truncation.** The division `total_latency_us / num_users` uses integer division, losing the remainder. For `total_latency_us = 42130` and `num_users = 3`, the stagger is $\lfloor 42130 / 3 \rfloor = 14043$ us, losing 1 us. Negligible, but pedantically imprecise.

**Fix:** For benchmarking with many users, consider setting `--stagger-us` explicitly to the bottleneck stage duration:

```bash
./pi05_pipeline_runner_device --config ... --stagger-us 2200
```

This ensures each user enters the pipeline only when the bottleneck stage is free. When `num_users = 1`, the auto-calculated stagger is the full pipeline latency (42130 us), which is a no-op.

---

## 8.4.9 action_history Chain Corruption Propagation

**Symptom:** A single corrupted D2H result causes all subsequent turns for that user to produce incorrect action predictions.

**Quick fix:** For production, each user must have its own `action_output` and `action_history` buffers (vector of vectors indexed by user or slot). The shared-buffer design is acceptable for benchmarking where action data is synthetic, but must not be carried forward to production. The D2H slot_id should be validated before accepting output; on mismatch, retry the turn or evict the user.

For the full analysis of the D2H slot_id mismatch that triggers this corruption chain (user A receives user B's output, propagating through turn k+1 and beyond), the shared-buffer concurrency risk, and the lack of integrity checks or rollback, see [Section 8.1.6](01_design_critique.md#816-d2h-slot_id-mismatch-logged-as-warning).

---

**End of guide.** Return to [Guide Index](../index.md)
