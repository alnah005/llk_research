# Abstracting CB Management for PyTorch Developers Using TT-Blaze

A comprehensive research guide for PyTorch developers who want to understand how an abstraction layer (TensorAdapter) can automate the 12+ manual decisions required to configure circular buffers, tile decomposition, padding, data formats, and op fusion when targeting Tenstorrent hardware via TT-Blaze. This guide also serves experienced Blaze developers who want to understand where such an abstraction adds value and where manual control remains necessary.

---

## How to Use This Guide

| Your Goal | Recommended Path | Key Sections |
|-----------|-----------------|--------------|
| Understand why CB abstraction is needed | Ch 1 → Ch 2 | [The Manual Plumbing Burden](ch01_the_gap/03_the_manual_plumbing_burden.md), [Synthesized Design Principles](ch02_design_patterns/05_synthesized_design_principles.md) |
| Learn the TensorAdapter architecture | Ch 3 → Ch 4 → Ch 5 | [Architecture Overview](ch03_tensor_adapter/01_architecture_overview.md), [Three Shape Layers](ch04_tile_decomposition/01_three_shape_layers.md) |
| Understand data format and CB sizing decisions | Ch 6 → Ch 7 | [Format Negotiation](ch06_data_formats/02_format_negotiation_protocol.md), [L1 Memory Model](ch07_cb_sizing/01_l1_memory_model.md) |
| Learn how broadcasting and fusion work | Ch 8 → Ch 9 | [Broadcast Primitives](ch08_broadcasting/02_mapping_to_blaze_broadcast_primitives.md), [Blaze Fusion Model](ch09_op_fusion/01_blaze_fusion_model.md) |
| Start using the API immediately | Ch 10 | [BlazeModule API](ch10_end_to_end/01_the_blaze_module_api.md), [Linear Layer Example](ch10_end_to_end/02_worked_example_linear_layer.md) |
| Migrate existing code | Ch 10 (File 07) | [Migration Guide](ch10_end_to_end/07_migration_guide.md) |
| Override automatic decisions | Ch 10 (File 06) | [Escape Hatches](ch10_end_to_end/06_escape_hatches_for_power_users.md) |

---

## Chapter Index

| Chapter | Title | Description | Key Concepts |
|---------|-------|-------------|--------------|
| [Ch 1 — The Gap](ch01_the_gap/) | The Gap Between PyTorch Tensors and Tenstorrent Hardware | Establishes the impedance mismatch between PyTorch's tensor model and Tenstorrent's tile/CB execution model | PyTorch tensors, CBHandle, tile geometry, 12 manual decisions |
| [Ch 2 — Design Patterns](ch02_design_patterns/) | Existing Framework Patterns and Synthesized Principles | Surveys TT-Symbiote, TVM, Triton, XLA, MLIR, torch.compile and extracts seven design principles (P1-P7) | P1-P7 principles, DP1-DP16 patterns, framework comparison |
| [Ch 3 — TensorAdapter](ch03_tensor_adapter/) | The TensorAdapter Architecture | Three-layer architecture (TensorAdapter, ShapeEngine, CBBackend), the adapt() pipeline, ShapeDescriptor, TypedHandle | adapt() 6-step pipeline, ShapeDescriptor, TypedHandle |
| [Ch 4 — Tile Decomposition](ch04_tile_decomposition/) | Tile Decomposition and Shape Propagation | Three shape layers (logical, padded, tile), automatic decomposition, shape propagation through op chains | logical/padded/tile shapes, shard mapping, core grids |
| [Ch 5 — Padding](ch05_padding/) | Padding Strategy and Propagation | Five padding strategies, the op padding registry, automatic insertion, consumer-determines-padding principle | ZERO, NEG\_INF, MASK, IDENTITY, NONE, PaddingPolicy |
| [Ch 6 — Data Formats](ch06_data_formats/) | Data Formats, Negotiation, and Precision Profiles | BFP format landscape, two-pass format negotiation, cb\_alias/cb\_reconfig, three precision profiles | bfloat16/8\_b/4\_b, FormatPolicy, LoFi/HiFi4 |
| [Ch 7 — CB Sizing](ch07_cb_sizing/) | CB Sizing, Blocking Strategy, and L1 Budgeting | L1 memory model, automatic page count, RT/CT/KT blocking, CB compaction via interval coloring | page\_size, num\_pages, CircularBufferIdManager |
| [Ch 8 — Broadcasting](ch08_broadcasting/) | Broadcasting: From PyTorch Implicit to TT-Blaze Explicit | PyTorch broadcasting rules, 27-entry mapping taxonomy, Mcast physical broadcast, shape tracking | bcast\_rows/cols/scalar, Mcast, BroadcastDetector |
| [Ch 9 — Op Fusion](ch09_op_fusion/) | Op Fusion with Shape, Padding, and Format Tracking | MicroOp/FusedOp composition, fusion context, shape/padding/format tracking across fused boundaries | FusedOp, compose(), FusionContext, fusion analyzer |
| [Ch 10 — End-to-End](ch10_end_to_end/) | End-to-End Developer API, Performance, and Escape Hatches | Complete BlazeModule API, worked examples, performance analysis, three override levels, migration guide | blaze.from\_pytorch(), BlazeModule, escape hatches |

---

## Quick Reference

| Operation | What It Does | Where to Learn More |
|-----------|-------------|-------------------|
| `blaze.from_pytorch(module, sample_input)` | Converts a PyTorch module to a BlazeModule with automatic CB management | [Ch 10, File 01](ch10_end_to_end/01_the_blaze_module_api.md) |
| `TensorAdapter.adapt(tensor, op_hint)` | Decomposes a tensor into tiles, applies padding/format, allocates CBs | [Ch 3, File 01](ch03_tensor_adapter/01_architecture_overview.md) |
| `ShapeDescriptor` | Frozen dataclass carrying logical\_shape, padded\_shape, tile\_shape, data\_format | [Ch 3, File 02](ch03_tensor_adapter/02_shape_metadata_system.md) |
| `TypedHandle` | CBHandle + ShapeDescriptor composite with dual access | [Ch 3, File 02](ch03_tensor_adapter/02_shape_metadata_system.md) |
| `FormatPolicy.select()` | Two-pass graph-walk format negotiation with cost model | [Ch 6, File 02](ch06_data_formats/02_format_negotiation_protocol.md) |
| `CBSizer.compute()` | Automatic page\_size and num\_pages calculation with L1 validation | [Ch 7, File 02](ch07_cb_sizing/02_automatic_page_count_and_page_size.md) |
| `BroadcastDetector.detect()` | Maps PyTorch implicit broadcasts to TT-Blaze primitives | [Ch 8, File 02](ch08_broadcasting/02_mapping_to_blaze_broadcast_primitives.md) |
| `blaze.fuse()` | Context manager for capturing op calls into a fused graph | [Ch 9, File 01](ch09_op_fusion/01_blaze_fusion_model.md) |
| `@blaze.override(...)` | Per-op escape hatch for overriding automatic decisions | [Ch 10, File 06](ch10_end_to_end/06_escape_hatches_for_power_users.md) |

---

## Prerequisites

- **Required:** Python fluency, PyTorch `nn.Module` / `torch.Tensor` API, basic model composition
- **Required:** Understanding of matrix multiplication, broadcasting, shape semantics, mixed precision (float32, bfloat16)
- **Helpful:** General awareness that hardware accelerators require data in specific layouts
- **Not required:** Any Tenstorrent-specific knowledge (this guide teaches it)

---

## Source Code References

Key source files referenced throughout this guide:

| File | Contains |
|------|---------|
| `blaze/blaze_op.py` | `BlazeOp`, `MicroOp`, `FusedOp`, `compose()` |
| `blaze/cb_handle.py` | `CBHandle` dataclass (12 fields) |
| `blaze/cb_engine.py` | `CBEngine`, `DTYPE_BYTES`, `DEFAULT_TILE_SHAPE` |
| `blaze/fused_program.py` | `FusedProgram`, `cb_from_tensor()`, `cb_scratch()`, `emit()` |
| `blaze/_gated_mlp.py` | `build_gated_mlp_graph()` (11-node template) |
| `blaze/ops/swiglu/op.py` | `SwigluOp.compose()` |
| `blaze/cb_reconfig.py` | CB reconfiguration primitives |
| `blaze/cb_reconfig_builder.py` | CB reconfig builder utilities |
| `blaze/context.py` | `ExternalTensor` escape hatch |
| `blaze/l1_profile.py` | `print_cb_stats()`, `print_kernel_stats()` |
