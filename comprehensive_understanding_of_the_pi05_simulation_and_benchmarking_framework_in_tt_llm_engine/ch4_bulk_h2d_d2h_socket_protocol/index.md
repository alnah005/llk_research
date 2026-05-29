# Chapter 4: Bulk H2D/D2H Socket Protocol

The Pi0.5 pipeline uses a dedicated pair of bulk sockets---separate from the token-level SocketPipeline---to transfer pixel frames from the host to the device and action-output tensors from the device back to the host. Built on tt-metal's experimental socket infrastructure, the protocol introduces its own framing, page-alignment rules, sentinel-based shutdown, and staging-buffer management. This chapter dissects the wire protocol, the host-side channel abstractions, the ordered shutdown handshake, and the concurrency hazards that arise from the interplay between page-aligned DMA, FIFO flow control, and multi-user scheduling.

## Sections

1. [Wire Protocol and Frame Layout](01_wire_protocol.md)
   Header formats, page-alignment arithmetic, H2D/D2H payload asymmetry, the sentinel shutdown marker, and first-turn vs. subsequent-turn payload variation.

2. [Bulk Channel Classes](02_bulk_channel_classes.md)
   `BulkH2DChannel` and `BulkD2HChannel` host-side wrappers: staging buffers, zero-fill security, spin-wait polling, sentinel I/O, and channel lifecycle.

3. [Shutdown Protocol](03_shutdown_protocol.md)
   The ordered sentinel-barrier-echo-stop-destroy teardown sequence, why ordering matters, the `_exit()` workaround, and robustness gaps.

4. [Race Conditions and Concurrency Hazards](04_race_conditions.md)
   Header/payload atomicity, pop-before-D2H ordering, D2H slot mismatch, FIFO sizing deadlock, H2D serialization, echo hang, and missing error propagation.

---

**Previous:** [Chapter 3 -- PipelineSimulator Internals](../ch3_pipelinesimulator_timing_model/index.md)

**Next:** [Chapter 5 -- Device Launcher and Kernel Architecture](../ch5_device_launcher_and_kernel_architecture/index.md)
