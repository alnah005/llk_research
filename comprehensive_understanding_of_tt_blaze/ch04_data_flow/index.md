# Chapter 4 -- Data Flow: CBHandle Chain and FusedProgram

This chapter covers the composition-API data-flow primitives in TT-Blaze: the CBHandle token that flows between ops, the FusedProgram context that manages CB allocation and graph recording, the OverlappedView mechanism for sub-tensor L1 sharing, and the multi-phase CB reconfiguration system that stretches 64 hardware CB slots across arbitrarily many logical buffers.

**Source files:**
- `blaze/cb_handle.py` -- CBHandle dataclass and CBAccessMode enum
- `blaze/fused_program.py` -- FusedProgram, OverlappedView, MultiOutput, TileInfo, helper functions
- `blaze/cb_reconfig.py` -- CircularBufferIdManager, CBContext, record_cb_metadata, build_cb_reconfig_tensor
- `blaze/cb_reconfig_builder.py` -- prepare_for_build, scratch compaction, multi-phase build pass

## Sections

### [01 -- CBHandle Abstraction](01_cbhandle_abstraction.md)

The CBHandle dataclass: every field, both access modes (FIFO and DIRECT_ADDRESS), integer coercion, the FIFO guard, buffer_address() resolution, shadow graph metadata, and the chaining pattern that connects one op's output to the next op's input. Includes a worked example tracing a matmul output handle through a gather input.

### [02 -- FusedProgram](02_fused_program.md)

The composition context: constructor parameters, grid infrastructure, CB allocation APIs (cb_from_tensor, cb_scratch, cb_from_view, cb_alias, cb_from_shared_l1_views), CT/RT arg declaration with CB position tracking, output() and multi_output() for shadow graph recording, semaphore and named-tensor allocation, internal tracking state, the build/run pipeline, and MeshFusedProgram for multi-device execution. Includes a worked example tracing a 3-op pipeline (mcast, matmul, reduce) through the full allocation and graph-recording sequence.

### [03 -- OverlappedView and Shared L1 Views](03_overlapped_view.md)

The OverlappedView frozen dataclass for sub-tensor addressing in fused L1 buffers: every field, the page_size property with its three computation paths, the duck-typed memory_config() bridge, the from_overlapped_tensor() migration classmethod, the cb_from_view() allocator with three sharing paths, the cb_from_shared_l1_views() direct-address allocator with eight validation checks, the _format_key() sharing bucket, and MultiOutput. Includes a worked example tracing gate+up weight packing through cb_from_view().

### [04 -- CB Reconfig: Multi-Phase Pipeline Reconfiguration](04_cb_reconfig.md)

CircularBufferIdManager and its Context for cross-phase ID allocation, CBContext for manual phase control, record_cb_metadata() and build_cb_reconfig_tensor() for the 264-word per-core reconfig tensor, the full cb_reconfig_builder.py build pass (scratch compaction with spatial/temporal reuse, arena materialization, multi-phase ID assignment, reconfig tensor construction, placeholder patching, descriptor deduplication), and the _remap_cb_ids rewrite pipeline. Includes a worked example tracing a MoE expert format switch through the entire multi-phase build.

## Cross-Chapter References

| Topic | Location | Relationship |
|---|---|---|
| Two-API design | Ch1 S03 | CBHandle is the "currency" of the Composition API introduced there |
| emit() methods | Ch2 S05-06 | emit() functions USE CBHandle; this chapter explains what CBHandle IS |
| CB Engine | Ch3 S02 | Graph-API path CB allocation; this chapter covers composition-API path |
| BlazeCompiler | Ch3 S06 | _split_per_device() consumes FusedProgram; seed_reserved_cb bridges the two paths |

---

[Next: 01 -- CBHandle Abstraction ->](01_cbhandle_abstraction.md)
