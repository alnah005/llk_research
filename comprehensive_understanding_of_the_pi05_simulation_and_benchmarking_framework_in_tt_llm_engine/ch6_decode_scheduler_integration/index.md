# Chapter 6: Decode Scheduler Integration

How Pi0.5 wires its multi-turn robotics inference loop into the tt-llm-engine `DecodeScheduler`. Covers the ALLOCATE/SUBMIT/EVICT request lifecycle, why Pi0.5 uses fresh-context SUBMIT (not CONTINUE), the `skip_eos_writeback` optimization, and per-user session management with staggered submission.

## Sections

1. [Scheduler Request Lifecycle](01_scheduler_request_lifecycle.md)
   DecodeScheduler creation, ALLOCATE/SUBMIT/CONTINUE/EVICT/STOP request types, Pi0.5's upfront-allocate + fresh-SUBMIT-per-turn pattern, SchedulerResponse/OutputMessage structs, SchedulerParams configuration, and the scheduler's internal threading model.

2. [Batch Prefill and skip_eos_writeback](02_batch_prefill_and_skip_eos.md)
   How `batch_prefill` and `skip_eos_writeback` interact inside `DecodeScheduler`; the `stage_eos_writeback` lambda; why Pi0.5 sets both to `true`; what breaks without these flags; cross-reference to Chapter 3 for `PipelineSimulator` batch_prefill internals.

3. [User Session Management](03_user_session_management.md)
   The `UserSession` struct, `slot_to_user` routing map, multi-turn loop with action-history reuse, staggered initial submission formula, prompt seed convention, and simulation vs. socket mode differences.

---

**Previous:** [Chapter 5 -- Device Launcher and Kernel Architecture](../ch5_device_launcher_and_kernel_architecture/index.md)

**Next:** [Chapter 7 -- Metrics, Reporting, and Benchmark Orchestration](../ch7_metrics_reporting_and_benchmark_orchestration/index.md)
