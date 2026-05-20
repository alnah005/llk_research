# Agent B Review: Chapter 5 — Pass 1

1. **File:** `socket_pipeline.md`, ~line 16-23 (PIMPL diagram) and line 28
   **Error:** The PIMPL diagram shows `bool use_deepseek_md_format` as a field inside `struct SocketPipeline::Impl`, and the prose on line 28 states "The `use_deepseek_md_format` flag is stored on the `Impl` struct." In the actual source (`socket_pipeline.hpp` line 35, `socket_pipeline.cpp` lines 18-31), `use_deepseek_md_format_` is a direct member of `SocketPipeline` itself, NOT part of `Impl`. The `Impl` struct contains only `h2d_socket`, `d2h_socket`, `write_buf`, `read_buf`, and `stop_requested`. All call sites in `socket_pipeline.cpp` reference `use_deepseek_md_format_` (the outer class member), not `impl_->use_deepseek_md_format`. A reader implementing their own PIMPL following this diagram would put the flag in the wrong struct.
   **Fix:** Move `use_deepseek_md_format` out of the `Impl` box in the diagram and into the `SocketPipeline` box (alongside `unique_ptr<Impl>`). Change the prose to: "The `use_deepseek_md_format_` flag is stored directly on `SocketPipeline`, not on the `Impl` struct, because it is needed by both `inject()` and `shutdown()` without going through the PIMPL indirection."

2. **File:** `socket_pipeline.md`, ~line 32-33
   **Error:** The constructor is shown as `SocketPipeline(const SocketConfig& config)`, implying it takes a `SocketConfig` struct. The actual source (`socket_pipeline.hpp` lines 21-22) shows the constructor takes individual parameters: `SocketPipeline(const std::string& h2d_socket_id, const std::string& d2h_socket_id, uint32_t connect_timeout_ms = 30000, bool use_deepseek_md_format = false)`. A reader looking at this signature to understand the API would have the wrong constructor interface.
   **Fix:** Replace the constructor signature with the actual one from source. The `SocketConfig`-to-individual-parameter unpacking happens at the `std::visit` call site (shown in `index.md`), not in the constructor itself.

3. **File:** `socket_pipeline.md`, ~line 61-69 (`read_result()` code)
   **Error:** The shown `read_result()` implementation omits the `impl_->read_buf.fill(0)` call that exists in the actual source (`socket_pipeline.cpp` line 47) before `d2h_socket->read()`. This zero-fill ensures stale data from a previous read does not leak into unused fields of the deserialized `ResultDescriptor`. A reader reimplementing this without the fill could get corrupted results when switching between DeepSeek and default formats or when reserved fields carry garbage.
   **Fix:** Add `impl_->read_buf.fill(0);` before the `d2h_socket->read()` call in the code listing.

4. **File:** `pipeline_interface.md`, ~line 109
   **Error:** The table states `predicted_token_type` is "Always `SPEC` for the predicted token." However, the default value in the source (`pipeline_types.hpp` line 55) is `TokenType::BASE`, and the default-mode `deserialize_result()` in `wire_format.hpp` does not populate this field at all — it remains `BASE`. Only the mock/simulator backends and DeepSeek deserialization set it to `SPEC`. Stating "always SPEC" is factually wrong for the default wire format path; a reader relying on this would incorrectly assume `predicted_token_type == SPEC` after deserializing a default-mode result page.
   **Fix:** Change to: "`TokenType` — Set to `SPEC` by mock/simulator backends and DeepSeek deserialization. Remains at default (`BASE`) in default wire format results, which do not transmit this field."

5. **File:** `pipeline_simulator.md`, ~line 77-78 (pseudocode for `read_result()`)
   **Error:** The pseudocode shows `read_result()` as only a busy-wait on `clock::now()`. The actual implementation (`pipeline_simulator.hpp` lines 100-106) first performs a blocking `emitCv.wait()` on a condition variable until the inflight deque is non-empty, and only THEN busy-waits on the exit time. This is a material omission: the condition variable wait is the mechanism that avoids 100% CPU spin when the pipeline is idle (no tokens in flight). A reader implementing from this pseudocode would spin continuously even with an empty pipeline, burning CPU and missing the backpressure-unblock notification path entirely.
   **Fix:** Update the pseudocode to:
   ```
   read_result():
       wait on emitCv until inflight is non-empty (or stop)
       head = inflight.front()
       unlock, busy-wait until steady_clock::now() >= head.exit_time
       re-lock, pop and return head.result, notify injectCv
   ```

---
## Agent A Change Log — Pass 1
- [socket_pipeline.md] Fixed PIMPL diagram: moved `use_deepseek_md_format` from Impl to SocketPipeline. Updated all code references from `impl_->use_deepseek_md_format` to `use_deepseek_md_format_`.
- [socket_pipeline.md] Fixed constructor signature from `SocketPipeline(const SocketConfig& config)` to individual parameters matching source.
- [socket_pipeline.md] Added `impl_->read_buf.fill(0)` before `d2h_socket->read()` in `read_result()`.
- [pipeline_interface.md] Fixed `predicted_token_type` description from "Always SPEC" to accurate description of per-backend behavior.
- [pipeline_simulator.md] Fixed `read_result()` pseudocode to include `emitCv.wait()` step, unlock before busy-wait, re-lock and `injectCv.notify_one()` after pop.

---
# Agent B Review: Chapter 5 — Pass 2 (Final Sweep)

## Pass 1 Fix Verification

1. **socket_pipeline.md PIMPL diagram and `use_deepseek_md_format_` placement**: Confirmed. The PIMPL diagram now shows `bool use_deepseek_md_` on the `SocketPipeline` box (line 18), not on `Impl`. The prose (lines 26-27) correctly states the flag is stored directly on `SocketPipeline`. All code listings (`inject()` line 44, `read_result()` line 59, `shutdown()` lines 117/121/127) use `use_deepseek_md_format_` without `impl_->` prefix, matching `socket_pipeline.cpp` lines 40, 49, 60, 63, 69.

2. **socket_pipeline.md constructor signature**: Confirmed. The constructor (lines 32-35) now shows `SocketPipeline(const std::string& h2d_socket_id, const std::string& d2h_socket_id, uint32_t connect_timeout_ms = 30000, bool use_deepseek_md_format = false)`, matching `socket_pipeline.hpp` lines 21-22 exactly.

3. **socket_pipeline.md `read_result()` missing `read_buf.fill(0)`**: Confirmed. Line 57 now includes `impl_->read_buf.fill(0);` before `d2h_socket->read()`, matching `socket_pipeline.cpp` line 47.

4. **pipeline_interface.md `predicted_token_type` description**: Confirmed. Line 109 now reads "Set to `SPEC` by mock/simulator backends and DeepSeek deserialization. Remains at default (`BASE`) in default wire format results, which do not transmit this field." This accurately reflects: (a) mock sets it to `SPEC` at `mock_pipeline.hpp` line 66, (b) simulator sets it to `SPEC` at `pipeline_simulator.hpp` line 165, (c) DeepSeek deserialization reads it from the page at `wire_format.hpp` line 104, and (d) default deserialization at `wire_format.hpp` lines 109-114 does not touch the field, so it stays at the default `BASE` from `pipeline_types.hpp` line 55.

5. **pipeline_simulator.md `read_result()` pseudocode missing `emitCv.wait()`**: Confirmed. The pseudocode (lines 75-79) now shows `wait on emitCv until inflight is non-empty (or stop)` as the first step, then unlock + busy-wait, then re-lock + pop + notify `injectCv`. This matches `pipeline_simulator.hpp` lines 100-128.

## Final Sweep (all 6 files vs source code)

1. **File:** `socket_pipeline.md`, lines 124-129 (`shutdown()` drain loop)
   **Error:** The `shutdown()` code listing omits `impl_->read_buf.fill(0)` in the drain loop before `d2h_socket->read()`. The actual source (`socket_pipeline.cpp` line 67) calls `impl_->read_buf.fill(0)` inside the `while (true)` loop before each read, identical to the pattern in `read_result()` that was fixed in Pass 1 issue #3. Without the zero-fill, stale data from a previous page could leak into the deserialized sentinel check.
   **Fix:** Add `impl_->read_buf.fill(0);` before `impl_->d2h_socket->read(impl_->read_buf.data(), 1);` in the `shutdown()` drain loop code listing, changing:
   ```cpp
   while (true) {
       impl_->d2h_socket->read(impl_->read_buf.data(), 1);
   ```
   to:
   ```cpp
   while (true) {
       impl_->read_buf.fill(0);
       impl_->d2h_socket->read(impl_->read_buf.data(), 1);
   ```

---
## Agent A Change Log — Pass 2
- [socket_pipeline.md] Added `impl_->read_buf.fill(0)` before `d2h_socket->read()` in the `shutdown()` drain loop, matching the fix already applied to `read_result()` in Pass 1.
