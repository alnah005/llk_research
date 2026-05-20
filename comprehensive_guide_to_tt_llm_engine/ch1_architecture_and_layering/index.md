# Chapter 1: Architecture and Layering

tt-llm-engine is the host-side control-plane scheduling engine for streaming LLM inference on Tenstorrent hardware. It sits between upstream Inference Server stacks and persistent, device-resident model workloads, bridging the gap between millisecond-timescale HTTP/RPC request handling and microsecond-timescale pipeline token injection. This chapter establishes the engine's architectural position, defines the ownership boundaries between components, maps all major subsystems, and introduces the invariants that constrain every design decision in the chapters that follow.

## Why This Chapter First

Every chapter that follows -- threading models, scheduling algorithms, speculative decode, pipeline abstractions, session lifecycle, build system -- depends on understanding four foundational questions:

1. **Where does the engine run?** Between an HTTP-facing Inference Server and a persistent device workload. The engine is the control-plane bridge, not the model runner.
2. **Who owns what?** The IS/PM ownership contract dictates which side stores sessions, which side allocates slots, and where policy versus mechanism live.
3. **What components exist today, and what is planned?** The component map tells you what is implemented versus designed-only, and how future schedulers slot in without restructuring.
4. **What rules are never violated?** Six system invariants constrain every data structure, thread, and queue in the engine. Breaking any one of them produces either pipeline stalls or use-after-free bugs.

## Reading Roadmap

| File | Content |
|------|---------|
| [`system_position.md`](./system_position.md) | Three-layer diagram with timescale annotations. The "bridge" metaphor. Semi-hardened deployment pattern. Repository URL transitional note. |
| [`ownership_contract.md`](./ownership_contract.md) | IS-vs-PM responsibility matrix. What each side must and must never do. `session_id` vs `user_id` mapping protocol. The three-queue boundary. Cancellation walkthrough. |
| [`component_map.md`](./component_map.md) | Table of all engine components with implementation status. Two library artifacts. Namespace hierarchy. Directory layout. Key constants and defaults. Build targets summary. |
| [`key_invariants.md`](./key_invariants.md) | Six key system invariants with enforcement locations. Timescale separation. Hardware deployment path (fixed-size structures, bitmap allocators, bounded queues, no dynamic allocation). |

## Navigation

Every subsequent chapter depends on the concepts introduced here. Use the table below to find the right next step for your role:

| Chapter | Focus | Key Ch1 Dependency | Recommended For |
|---------|-------|--------------------|-----------------|
| [Ch 2 -- Threading & Data Structures](../ch2_threading_and_data_structures/index.md) | Three-thread architecture, fixed-size data structures | Hardware deployment path | All readers |
| [Ch 3 -- Scheduling Algorithm](../ch3_scheduling_algorithm/index.md) | Writer/reader loops | "PM never blocks on IS" invariant | Scheduler developers |
| [Ch 4 -- Speculative Decode](../ch4_speculative_decode/index.md) | MTP speculation protocol | Base scheduling flow | Scheduler developers |
| [Ch 5 -- Pipeline & Wire Format](../ch5_pipeline_and_wire_format/index.md) | PipelineInterface, three backends | Component map, system position | Hardware/transport developers |
| [Ch 6 -- Lifecycle, IS Integration, KV Tiering](../ch6_lifecycle_is_integration_kv_tiering/index.md) | Session state machine, queue integration, KV tiers | Ownership contract | IS integrators |
| [Ch 7 -- Transport & Tools](../ch7_transport_and_tools/index.md) | Mooncake transport, loopback benchmark, DeepSeek runner | System position | Hardware/transport developers |
| [Ch 8 -- Build System & Design Decisions](../ch8_build_system_and_design_decisions/index.md) | CMake build, architectural tradeoffs | Component map | All readers |

**Role-based reading paths:**
- **IS integrators:** Read all of Ch1, then proceed to Ch 6.
- **Scheduler developers:** Read all of Ch1, then proceed sequentially through Ch 2--8.
- **Hardware/transport developers:** Focus on [System Position](./system_position.md) and [Component Map](./component_map.md), then proceed to Ch 5.
