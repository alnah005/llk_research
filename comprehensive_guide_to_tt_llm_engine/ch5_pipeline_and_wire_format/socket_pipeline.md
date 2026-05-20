# SocketPipeline: Real Hardware Backend

*Source: `socket_pipeline.hpp`, `socket_pipeline.cpp`*

## Purpose

`SocketPipeline` is the production backend. It connects the host engine to a Tenstorrent device through tt-metalium's Host-to-Device (H2D) and Device-to-Host (D2H) socket abstraction. Every token injected by the Writer thread (Ch2) is serialized into a 256-byte page and written to the H2D socket; every result read by the Reader thread is a 256-byte page deserialized from the D2H socket. This is the only `PipelineInterface` implementation that touches real hardware.

## PIMPL Pattern

`SocketPipeline` uses the Pointer-to-Implementation (PIMPL) idiom to hide tt-metalium dependencies from the header:

```
socket_pipeline.hpp                socket_pipeline.cpp
+---------------------------+      +------------------------------+
| class SocketPipeline      |      | struct SocketPipeline::Impl  |
|   unique_ptr<Impl> impl_  |----->|   H2DSocket h2d_socket       |
|   bool use_deepseek_md_   |      |   D2HSocket d2h_socket       |
|   inject()                |      |   PageBuffer write_buf       |
|   read_result()           |      |   PageBuffer read_buf        |
|   ...                     |      |   atomic<bool> stop_requested|
+---------------------------+      +------------------------------+
```

PIMPL provides two benefits: (1) **Compilation firewall** -- code including `socket_pipeline.hpp` does not transitively include tt-metalium distributed headers, speeding compilation and reducing coupling. (2) **ABI stability** -- the `Impl` struct can change without affecting the header's binary layout.

The `use_deepseek_md_format_` flag is stored directly on `SocketPipeline`, not on the `Impl` struct, because it is needed by both `inject()` and `shutdown()` without going through the PIMPL indirection.

## Construction

```cpp
SocketPipeline(const std::string& h2d_socket_id,
               const std::string& d2h_socket_id,
               uint32_t connect_timeout_ms = 30000,
               bool use_deepseek_md_format = false);
```

The constructor takes individual parameters, not a `SocketConfig` struct. The `SocketConfig`-to-parameter unpacking happens at the `std::visit` call site (see [index.md](./index.md)). The constructor connects to both sockets (blocking up to `connect_timeout_ms`) and configures both with `set_page_size(PAGE_SIZE_BYTES)`. Socket IDs are opaque strings that identify communication channels -- the device firmware and host must agree on these at deployment time (Ch1).

## inject() Implementation

```cpp
void SocketPipeline::inject(const InjectDescriptor& desc) {
    impl_->write_buf = serialize_inject(desc, use_deepseek_md_format_);
    impl_->h2d_socket->write(impl_->write_buf.data(), 1);
}
```

Two steps: serialize the descriptor into the pre-allocated `write_buf` (see [Wire Format](./wire_format.md)), then write exactly one page to the H2D socket. The buffer is reused across calls -- safe because `inject()` is only called from the Writer thread (Ch2). No heap allocation on the hot path.

## read_result() Implementation

```cpp
ResultDescriptor SocketPipeline::read_result() {
    while (!impl_->stop_requested.load(std::memory_order_acquire)) {
        if (impl_->d2h_socket->has_data()) {
            impl_->read_buf.fill(0);
            impl_->d2h_socket->read(impl_->read_buf.data(), 1);
            return deserialize_result(impl_->read_buf, use_deepseek_md_format_);
        }
    }
    return ResultDescriptor{.slot_id = INVALID_SLOT};
}
```

The Reader thread polls the D2H socket in a tight loop, checking `stop_requested` on every iteration. This polling approach (rather than a blocking `read()`) ensures that `request_stop()` can break the loop without requiring the D2H socket to support cancellation. If the device hangs, the poll loop still exits cleanly during shutdown.

The read buffer is reused across calls -- safe because `read_result()` is only called from the Reader thread.

## reset_kv()

```cpp
void SocketPipeline::reset_kv(uint32_t) { /* no-op */ }
```

Currently a no-op in the codebase, as in all other backends. KV-cache management happens on the device side through the normal token flow.

## Two-Phase Shutdown with Sentinel Exchange

SocketPipeline implements the two-phase shutdown protocol described in [PipelineInterface](./pipeline_interface.md#two-phase-shutdown-protocol). The host-side mechanics (atomic flag → sentinel return → thread join) are identical; SocketPipeline's `shutdown()` adds the device-side sentinel exchange:

```
After thread join:

Main Thread                                           Device
    |                                                    |
    | shutdown()                                         |
    |--- write INVALID_SLOT sentinel ------------------->|
    |                                                    | processes remaining
    |                                                    | echoes sentinel
    |<-- drain until sentinel result --------------------+
    |--- h2d_socket->barrier() ------------------------->|
    |--- d2h_socket->barrier() ------------------------->|
    | [done]                                             |
```

### shutdown()

```cpp
void SocketPipeline::shutdown() {
    if (!use_deepseek_md_format_) {
        // Send sentinel inject to device
        InjectDescriptor sentinel{};
        sentinel.slot_id = INVALID_SLOT;
        impl_->write_buf = serialize_inject(sentinel, use_deepseek_md_format_);
        impl_->h2d_socket->write(impl_->write_buf.data(), 1);

        // Drain until device echoes sentinel result
        while (true) {
            impl_->read_buf.fill(0);
            impl_->d2h_socket->read(impl_->read_buf.data(), 1);
            auto result = deserialize_result(impl_->read_buf, use_deepseek_md_format_);
            if (result.slot_id == INVALID_SLOT) break;
        }
    }

    // Both modes call barrier
    impl_->h2d_socket->barrier();
    impl_->d2h_socket->barrier();
}
```

The sentinel is serialized through `serialize_inject()`, which places `INVALID_SLOT` at the correct word offset for the active format. The drain loop consumes any in-flight results before the device echoes the sentinel back. The barriers flush all buffered data in both directions.

### DeepSeek Mode Exception

When `use_deepseek_md_format == true`, the sentinel exchange is **skipped entirely**. The DeepSeek device firmware does not recognize the in-band `INVALID_SLOT` sentinel -- it uses a separate control channel for lifecycle management. The host calls `barrier()` on both sockets directly.

## DeepSeek Mode Summary

| Aspect | Default | DeepSeek |
|--------|---------|----------|
| Inject serialization | Default layout (`slot_id` at word 0) | DeepSeek layout (`slot_id` at word 6, `token_type` at word 1) |
| Result deserialization | Sparse layout (no `p_indices`/`p_scores`) | Full layout with top-32 probability arrays |
| Shutdown | Sentinel exchange + barrier | Barrier only |

## Thread Safety

`SocketPipeline` has a simpler concurrency model than the other backends because it delegates thread safety to the socket implementation:

- `inject()`: Writer thread only. Accesses `write_buf` and `h2d_socket`.
- `read_result()`: Reader thread only. Accesses `read_buf` and `d2h_socket`.
- Shared state: only `stop_requested` (`std::atomic<bool>`, acquire/release ordering).

There is no mutex. The Writer and Reader threads operate on completely disjoint state, with the atomic flag as the sole synchronization point.

---

| | Navigation | |
|:---|:---:|---:|
| [PipelineSimulator](./pipeline_simulator.md) | [Table of Contents](../index.md) | [Wire Format](./wire_format.md) |
