# Output Unpadding

The three previous files defined the padding taxonomy, the per-op registry, and the automatic insertion pass. This final file addresses the last mile: when a padded tensor returns to PyTorch, the abstraction must strip the padding to restore the original logical shape. This section defines the unpadding mechanism, how padding metadata propagates through the chain to inform unpadding, why unpadding must happen on the host without device overhead, how mask tensors must be padded consistently with the data they mask, and diagnostic tools for debugging padding-related issues.

> *Design Principle P1 (Separate Computation Intent from Execution Strategy):* The developer's output tensor must match their expected logical shape exactly. Padding is an execution strategy invisible to the developer. If the developer writes `output = blaze_module(input)` with `input.shape == [1, 5, 64]`, they must receive `output.shape == [1, 5, 64]`, never `[1, 32, 64]`.

**What you will learn:**

- How to strip padding from the final output tensor using logical shape metadata from ShapeDescriptor -- a zero-copy view operation
- How padding metadata propagates from initial `adapt()` through every `emit()` to the final `output()` via the PaddingChainTracker
- Why host-side unpadding (after device-to-host transfer) is preferred over device-side unpadding, with performance analysis
- How attention mask tensors must be padded consistently with data, including the causal mask padding walkthrough
- Stale padding detection for debugging: verifying that padding positions still contain the expected fill values
- Multi-output ops, partial unpadding, and the BlazeModule integration

---

## The Unpadding Operation

### Concept

Unpadding is the inverse of padding: given a tensor with `padded_shape` and the original `logical_shape`, extract the sub-tensor of real data:

```
Padded tensor:  shape [32, 4096]   (height padded from 1 to 32)
Logical shape:  [1, 4096]

Unpadded output = padded_tensor[0:1, 0:4096]  ->  shape [1, 4096]
```

### Implementation

```python
# PROPOSED -- unpadding function
def unpad_output(
    padded_tensor: torch.Tensor,
    logical_shape: tuple[int, ...],
) -> torch.Tensor:
    """Strip padding from a device output tensor to restore logical shape.

    Operates on a host-side torch.Tensor after device-to-host transfer.
    Returns a view (not a copy) of the padded tensor.

    Raises ValueError if padded_tensor's shape is smaller than logical_shape.
    """
    padded_shape = padded_tensor.shape

    # Validate
    if len(padded_shape) != len(logical_shape):
        raise ValueError(
            f"Rank mismatch: padded rank {len(padded_shape)} "
            f"!= logical rank {len(logical_shape)}."
        )
    for i, (p, l) in enumerate(zip(padded_shape, logical_shape)):
        if p < l:
            raise ValueError(
                f"Padded dim {i} ({p}) < logical ({l}). Shape tracking bug."
            )

    # Compute and apply slices -- creates a view, not a copy
    slices = tuple(slice(0, l) for l in logical_shape)
    result = padded_tensor[slices]

    assert result.shape == torch.Size(logical_shape), (
        f"Unpadding produced {result.shape}, expected {logical_shape}"
    )
    return result
```

### The View Optimization

PyTorch's slice operation returns a **view** of the original tensor, not a copy:

```python
padded = torch.zeros(32, 4096)       # 256 KB
unpadded = padded[0:1, 0:4096]       # View into the same 256 KB
unpadded.data_ptr() == padded.data_ptr()  # True: same memory

# No extra memory allocated
# No data copied
# O(1) time complexity
```

> **Warning:** If the unpadded tensor is passed to an operation requiring contiguous memory (C++ extension, CUDA kernel), PyTorch may implicitly call `.contiguous()`, creating a copy. The abstraction should document that `unpad_output()` returns a view.

---

## Tracking Padding Metadata Through the Op Chain

### The Full Pipeline

```
Entry point:
  tensor = torch.randn(1, 5, 64)
  adapted = adapter.adapt(tensor, op_hint="matmul.in0")
  # adapted.shape.logical_shape == (1, 5, 64)
  # adapted.shape.padded_shape  == (1, 32, 64)
  # adapted.shape.has_padding   == True

Mid-chain:
  mm_out = Matmul.emit(f, adapted.handle, weight_handle)
  mm_typed = tracker.propagate_through_op("matmul",
      {"in0": adapted, "in1": weight_typed}, mm_out)
  # mm_typed.shape.logical_shape == (1, 5, 128)  <- logical M preserved
  # mm_typed.shape.padded_shape  == (1, 32, 128) <- padded M preserved

Exit point:
  output_tensor = device_to_host(silu_typed.handle)
  # output_tensor.shape == (1, 32, 128)  <- padded shape from device

  final = unpad_output(output_tensor, silu_typed.shape.logical_shape)
  # final.shape == (1, 5, 128)  <- correct logical shape
```

### Integration with BlazeModule

```python
# PROPOSED -- BlazeModule forward with automatic unpadding
class BlazeModule(torch.nn.Module):
    """Guarantees:
    1. output.shape == expected_logical_shape
    2. output contains only real data (no padding artifacts)
    3. Unpadding is automatic and invisible to the developer
    4. If no padding was needed, output is returned as-is (no copy)
    """

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # Step 1: Execute on device (padding inserted automatically)
        device_output = self._execute(input_tensor)

        # Step 2: Transfer to host
        host_output = ttnn.to_torch(device_output)

        # Step 3: Automatic unpadding
        output_desc = self._shape_tracker.lookup(self._output_handle)
        if output_desc.has_padding:
            return unpad_output(host_output, output_desc.logical_shape)
        return host_output
```

### Inspecting Intermediate Results

In some cases, a developer may want to inspect intermediate padded results for debugging. The ShapeTracker makes this possible for any node in the chain:

```python
# PROPOSED -- intermediate unpadding for debugging
def get_intermediate_result(
    node_id: str,
    shape_tracker: ShapeTracker,
    device_tensor: "ttnn.Tensor",
) -> torch.Tensor:
    """Extract and unpad an intermediate tensor for debugging.

    WARNING: This transfers data from device to host. Use only for debugging.
    In production, intermediate device tensors are not accessible because
    they may be overwritten by subsequent ops sharing the same CB.
    """
    desc = shape_tracker.lookup_by_node(node_id)
    host_tensor = ttnn.to_torch(device_tensor)
    if desc.has_padding:
        return unpad_output(host_tensor, desc.logical_shape)
    return host_tensor
```

This function is the unpadding equivalent of TensorAdapter's `adapt()`: where `adapt()` adds padding at the input boundary, `get_intermediate_result()` removes it at any inspection point.

### PaddingChainTracker Integration

The `PaddingChainTracker` from File 03 provides the information needed for unpadding:

```python
# PROPOSED -- chain tracker output spec
class PaddingChainTracker:
    def get_output_unpad_spec(self, output_node_id: str) -> dict:
        """Get the unpadding specification for a graph output node."""
        entry = self._node_to_entry.get(output_node_id)
        if entry is None:
            return {"needs_unpad": False}

        logical = entry.logical_shape
        padded = entry.padded_shape
        if logical == padded:
            return {"needs_unpad": False}

        logical_elems = 1
        padded_elems = 1
        for l in logical:
            logical_elems *= l
        for p in padded:
            padded_elems *= p

        return {
            "needs_unpad": True,
            "logical_shape": logical,
            "padded_shape": padded,
            "slices": tuple(slice(0, l) for l in logical),
            "num_padded_elements": padded_elems - logical_elems,
            "padding_ratio": 1.0 - (logical_elems / padded_elems),
        }
```

---

## Host-Side vs. Device-Side Unpadding

### Why Host-Side Is Preferred

| Approach | Implementation | Cost |
|----------|---------------|------|
| **Host-side** (after DtH) | `torch.Tensor` slice | Zero compute, view only |
| **Device-side** (before DtH) | Kernel writing logical elements | Reduces transfer; adds kernel overhead |

Host-side is preferred because:

1. **Zero device overhead:** No kernel launch, no L1 consumption, no CB allocation.
2. **Minimal PCIe overhead:** Padding is typically small in absolute terms.
3. **View semantics:** Unpadding itself costs nothing.
4. **Simplicity:** No extra compilation, CB ID, or CT arg wiring.

### PCIe Transfer Overhead Analysis

| Scenario | Logical Size | Padded Size | Overhead | Transfer Time (12 GB/s) |
|----------|-------------|-------------|----------|------------------------|
| LLaMA-7B decode [1, 4096] | 8 KB | 256 KB | 97% | 21 us |
| LLaMA-7B prefill [128, 4096] | 1 MB | 1 MB | 0% | 83 us |
| DeepSeek V3 decode [1, 7168] | 14 KB | 448 KB | 97% | 37 us |
| Custom [1, 4000] | 8 KB | 252 KB | 97% | 21 us |
| Custom [1, 5, 64] | 640 B | 4 KB | 84% | 0.3 us |
| Batch decode [32, 1, 4096] | 256 KB | 256 KB | 0% | 21 us |

**Single-token decode** always has ~97% height-padding overhead, but absolute sizes are small (<500 KB). **Prefill** with tile-aligned batch/seq has zero overhead. Transfer times are in microseconds -- negligible vs. kernel execution (~100+ us for a matmul).

### When Device-Side Unpadding Is Justified

```python
# PROPOSED -- device-side unpadding decision
def should_unpad_on_device(output_desc, transfer_threshold=10_000_000,
                            ratio_threshold=0.5):
    """Returns True if padding overhead justifies device kernel cost."""
    if not output_desc.has_padding:
        return False
    padded_bytes = output_desc.total_tiles * output_desc.page_size
    logical_elements = 1
    for d in output_desc.logical_shape:
        logical_elements *= d
    DTYPE_BYTES = {"bfloat16": 2, "float32": 4, "bfloat8_b": 1}
    logical_bytes = logical_elements * DTYPE_BYTES.get(output_desc.data_format, 2)
    padding_ratio = 1.0 - (logical_bytes / padded_bytes)
    return padded_bytes > transfer_threshold and padding_ratio > ratio_threshold
```

Justified scenarios: multi-device pipelines (avoid inter-device transfer waste), very large tensors (>10 MB) with >50% padding, streaming inference.

> **Warning:** For single-token decode, the ~21 us PCIe penalty is comparable to the ~5 us device kernel cost. Given that device-side unpadding also consumes a CB ID and L1 memory (both scarce), host-side is the better default.

---

## Mask Tensor Padding Consistency

### The Problem

Attention masks must be padded **consistently** with the data they mask:

```
Data:  [1, 5, 64]  padded to [1, 32, 64]   (5 real seq positions)
Mask:  [1, 5, 5]   must pad to [1, 32, 32]  (matching seq dim)

If inconsistent:
  Data: [1, 32, 64]  (32 seq)
  Mask: [1, 8, 8]    (8 positions -- WRONG)
  -> Unmasked padding positions pollute softmax
```

### Causal Mask Padding

```
Logical causal mask for seq_len=5:
  [1, 0, 0, 0, 0]
  [1, 1, 0, 0, 0]
  [1, 1, 1, 0, 0]
  [1, 1, 1, 1, 0]
  [1, 1, 1, 1, 1]

Padded causal mask (tile_width=32):
  Rows 0-4: [real causal pattern, 0, 0, ..., 0]  (27 padding zeros per row)
  Rows 5-31: [0, 0, 0, ..., 0]                    (all-zero padding rows)
```

The padding zeros serve double duty:
1. **Column padding (5:32):** Masks out padded K/V positions ("do not attend")
2. **Row padding (5:32):** Masks out padded Q positions ("no output from these rows")

> **Warning:** The mask padding fill value must be 0 (do not attend), NOT 1. A mask padded with 1s allows attending to padding positions, contaminating the softmax denominator.

### Validation

```python
# PROPOSED -- mask padding consistency check
def validate_mask_padding(data_desc, mask_desc):
    """Validate mask is padded consistently with its data tensor."""
    data_seq_padded = data_desc.padded_shape[-2]
    mask_h_padded = mask_desc.padded_shape[-2]
    mask_w_padded = mask_desc.padded_shape[-1]

    if mask_h_padded != data_seq_padded:
        raise PaddingConsistencyError(
            f"Mask height ({mask_h_padded}) != data seq dim ({data_seq_padded})."
        )
    if mask_w_padded != data_seq_padded:
        raise PaddingConsistencyError(
            f"Mask width ({mask_w_padded}) != data seq dim ({data_seq_padded}). "
            f"Self-attention mask must be square with padded seq length."
        )
```

### Mask in SDPA

```python
# EXISTING -- from SDPA emit()
qk_tiles = PNHt_pg * Sk_chunk_t_cb
cb_mask_in = _scratch("mask_in", qk_tiles)
```

The `qk_tiles` count derives from padded Q and K dimensions, ensuring the mask covers all padded positions. The BRISC kernel fills with `identity_scalar_packed` (1.0) for valid and `zero_scalar_packed` (0.0) for causal/padding positions.

---

## Stale Padding Detection (Debug Mode)

In complex op chains, padding values may be corrupted by intermediate operations. A debug-mode validator checks that padding positions still contain the expected fill:

```python
# PROPOSED -- stale padding detection (debug mode only)
def validate_padding_integrity(
    tensor: torch.Tensor,
    desc: ShapeDescriptor,
    tolerance: float = 1e-6,
) -> list[str]:
    """Check that padding positions contain the expected fill value.

    This is a debug-time check -- too expensive for production.
    """
    if not desc.has_padding or desc.padding_strategy == "NONE":
        return []

    issues = []
    expected = desc.padding_fill
    padded_shape = desc.padded_shape
    logical_shape = desc.logical_shape

    # Check each dimension for padding
    for dim in range(len(logical_shape)):
        if logical_shape[dim] < padded_shape[dim]:
            # Slice the padding region along this dimension
            pad_slice = [slice(None)] * len(logical_shape)
            pad_slice[dim] = slice(logical_shape[dim], padded_shape[dim])
            padding_region = tensor[tuple(pad_slice)]

            if expected == float("-inf"):
                bad_mask = ~torch.isinf(padding_region) | (padding_region > 0)
            elif expected == float("inf"):
                bad_mask = ~torch.isinf(padding_region) | (padding_region < 0)
            else:
                bad_mask = (padding_region - expected).abs() > tolerance

            num_bad = bad_mask.sum().item()
            if num_bad > 0:
                issues.append(
                    f"Dim {dim}: {num_bad} padding positions do not match "
                    f"expected fill {expected}. "
                    f"This may indicate a kernel wrote to padding positions."
                )

    return issues
```

> **Warning:** Stale padding detection is expensive (requires reading device memory) and should only be enabled in debug mode. In production, rely on the registry and compatibility matrix to guarantee correctness statically.

---

## Multi-Output Ops

Some ops produce multiple outputs, each with potentially different padding:

```python
# PROPOSED -- multi-output unpadding
class MultiOutputBlazeModule(torch.nn.Module):
    def forward(self, *args, **kwargs) -> tuple[torch.Tensor, ...]:
        device_outputs = self._execute(*args, **kwargs)
        results = []
        for device_out, desc in zip(device_outputs, self._output_descriptors):
            host_out = ttnn.to_torch(device_out)
            results.append(maybe_unpad(host_out, desc))
        return tuple(results)

def maybe_unpad(tensor, desc):
    """Unpad if needed, return as-is if not."""
    if not desc.has_padding:
        return tensor
    return unpad_output(tensor, desc.logical_shape)
```

The PaddingChainTracker records metadata per output port, not per op.

---

## Walkthrough: Unpadding Through Non-Aligned MLP

Continuing the MLP walkthrough from File 03 with `hidden_dim=4000`:

```
Input: activation [1, 4000]
Tile: (32, 32)
Padded: [32, 4000]  (4000 is already tile-aligned; only height pads 1 -> 32)

Chain execution:
  Gate matmul: [1, 4000] @ [4000, 11008] -> logical [1, 11008], padded [32, 11008]
  Up matmul:   same
  SiLU(gate):  logical [1, 11008], padded [32, 11008]
  gate * up:   logical [1, 11008], padded [32, 11008]
  Down matmul: [1, 11008] @ [11008, 4000] -> logical [1, 4000], padded [32, 4000]

Output from device: shape [32, 4000]

PaddingChainTracker:
  logical_shape = (1, 4000)
  padded_shape  = (32, 4000)
  needs_unpad   = True
  slices        = (slice(0,1), slice(0,4000))

Unpadding:
  host_output = ttnn.to_torch(device_output)  # [32, 4000]
  final = host_output[0:1, 0:4000]             # [1, 4000] (view)

Padding statistics:
  padded_elements = 32 * 4000 = 128,000
  logical_elements = 1 * 4000 = 4,000
  padding_ratio = 1 - 4000/128000 = 96.9%
  transfer_overhead = 128,000 * 2 = 256,000 bytes (250 KB)
  logical_data = 4,000 * 2 = 8,000 bytes (7.8 KB)

  The PCIe transfer sends 250 KB but only 7.8 KB is useful data.
  At 12 GB/s PCIe bandwidth: 250 KB / 12 GB/s = 21 microseconds.
  This overhead is negligible for inference latency.
```

Only height needed unpadding here (32 to 1), since `hidden_dim=4000` is already a multiple of 32. The width slice `[0:4000]` is a no-op but included for uniformity. When the width is truly non-aligned (e.g., `hidden_dim=4001`), the width slice crosses tile boundaries, but PyTorch's slice on the reassembled host tensor handles partial tiles correctly.

---

## Edge Cases

### Partial Unpadding

When only some dimensions are padded, slices are identity for aligned dimensions:

```
Input: [2, 8, 5, 64]  (batch and hidden aligned, seq not)
Padded: [2, 8, 32, 64]
Slices: (slice(0,2), slice(0,8), slice(0,5), slice(0,64))
  -> First, second, fourth slices are full-dimension no-ops
  -> Third slice removes padding rows
```

### Nested Padding (Multiple Re-Padding Levels)

```
Input: [1, 37]
  Tile decomposition: padded to [32, 64] (ZERO)
  Matmul: [32, 128] (ZERO)
  Repad for softmax: [32, 128] (NEG_INF in padding)
  Softmax: [32, 128] (ZERO -- exp(-inf)=0)
  Final: unpad to [1, 128]
```

**Unpadding is determined solely by the logical shape**, not by the padding strategy. The strategy affects the fill value (numerical correctness during computation), but the unpadding slice depends only on which positions contain real data.

### Non-Contiguous Output

When both dimensions are sliced, the result may be non-contiguous:

```python
padded_tensor = torch.randn(32, 64)  # contiguous
sliced = padded_tensor[0:7, 0:37]    # non-contiguous view
sliced.is_contiguous()               # False (strides don't match shape)
```

If the caller passes the unpadded tensor to an operation requiring contiguous memory (C++ extension, CUDA kernel, certain serialization paths), PyTorch implicitly calls `.contiguous()`, creating a copy. The `unpad_output()` function intentionally returns the view, not a contiguous copy, because:

1. Most PyTorch operations accept non-contiguous tensors transparently
2. The caller may immediately consume the tensor without needing contiguity
3. Forcing `.contiguous()` on every unpad would add an O(logical_elements) copy

If a specific downstream path requires contiguity, the caller can explicitly call `.contiguous()` on the result.

### No-Op Unpadding

For production LLMs with tile-aligned dimensions (`logical_shape == padded_shape`), unpadding is a no-op:

```python
if not desc.has_padding:
    return tensor  # No padding -> return unchanged
```

This path is taken for the majority of outputs in production workloads.

---

## Diagnostic Output

```python
# PROPOSED -- padding diagnostic summary
def print_padding_summary(chain_tracker: PaddingChainTracker):
    """Print human-readable summary of padding decisions."""
    print("=== Padding Chain Summary ===")
    for entry in chain_tracker._chain:
        needs_pad = entry.logical_shape != entry.padded_shape
        pad_str = f"PADDED ({entry.padding_strategy.value})" if needs_pad else "ALIGNED"
        repad_str = " [REPADDED]" if entry.was_repadded else ""
        print(
            f"  {entry.op_id} ({entry.op_type}): "
            f"{entry.logical_shape} -> {entry.padded_shape} {pad_str}{repad_str}"
        )
    final = chain_tracker.get_final_unpad_spec()
    if final["needs_unpad"]:
        print(f"\n  Output: {final['padded_shape']} -> {final['logical_shape']}")
        print(f"  Padding ratio: {final['padding_ratio']:.1%}")
    else:
        print("\n  No unpadding needed (all dimensions tile-aligned)")
```

**Example output for the attention block (seq_len=5):**

```
=== Padding Chain Summary ===
  q_proj (matmul): (1, 5, 64) -> (1, 32, 64) PADDED (zero)
  k_proj (matmul): (1, 5, 64) -> (1, 32, 64) PADDED (zero)
  v_proj (matmul): (1, 5, 64) -> (1, 32, 64) PADDED (zero)
  qk_scores (matmul): (1, 5, 5) -> (1, 32, 32) PADDED (zero)
  repad_qk_softmax (repad): (1, 5, 5) -> (1, 32, 32) PADDED (neg_inf) [REPADDED]
  softmax (softmax): (1, 5, 5) -> (1, 32, 32) PADDED (zero)
  attn_output (matmul): (1, 5, 64) -> (1, 32, 64) PADDED (zero)

  Output: (1, 32, 64) -> (1, 5, 64)
  Padding ratio: 84.4%
```

---

## Host-Side Unpadding Cost Summary

| Operation | Cost | When It Occurs |
|-----------|------|----------------|
| Shape comparison | O(1) | Every `unpad_output()` call |
| No-op return | O(1) | When logical == padded (common case) |
| Slice construction | O(rank) | When logical != padded |
| Tensor slicing | O(1) | View creation, zero data movement |
| `.contiguous()` copy | O(logical_elements) | Only if caller requires it |
| Debug padding check | O(padded_elements) | Debug mode only |
| PCIe transfer overhead | O(padded - logical) | Always (padded tensor transferred) |

For the common case (tile-aligned dimensions in production LLMs), unpadding adds exactly one tuple comparison -- effectively zero overhead.

---

## Integration with the Complete Pipeline

```
Complete pipeline:
  1. Developer calls blaze_module(input_tensor)
  2. TensorAdapter.adapt() records logical_shape in ShapeDescriptor
  3. Padding pass inserts PadOps where needed (File 03)
  4. Graph compiled and executed on device
  5. Device output transferred to host (DtH)
  6. ** unpad_output() strips padding **  <- THIS FILE
  7. Developer receives output with correct logical shape

The developer never sees padding artifacts. The P7 (Transparent Interception)
principle is preserved: the abstraction is invisible.
```

---

## Key Takeaways

- Output unpadding strips the padded shape back to the logical shape using a simple slice operation. Because PyTorch's slice returns a **view** (not a copy), unpadding has zero memory overhead and zero compute cost. The developer never sees padding artifacts.
- Padding metadata propagates through the entire op chain via ShapeDescriptor's `logical_shape` field. Every op's shape inference rule preserves the original logical dimensions (matmul's output M is the **logical** M, not the padded M). The PaddingChainTracker records the full chain for diagnostics and final unpadding.
- Host-side unpadding (after device-to-host transfer) is preferred because it adds zero device overhead: no kernel launch, no CB ID, no L1 memory. PCIe transfer penalty for padding bytes is in microseconds -- negligible vs. kernel execution times.
- Mask tensors must be padded consistently with the data they mask. For attention, the causal mask must be padded to the same tile-aligned sequence length as Q and K, with padding positions set to 0 (do not attend). Inconsistent mask padding allows attending to garbage data.
- Stale padding detection (debug mode) verifies that padding positions contain expected fill values. This catches bugs where intermediate kernels inadvertently write to padding positions, corrupting downstream computations.
- For production LLMs with tile-aligned dimensions (4096, 7168, 11008, 2048), unpadding is a no-op because `logical_shape == padded_shape`. The mechanism matters for non-standard sequence lengths, custom architectures, and debugging.

## Source Files

- `blaze/ops/padded_rmsnorm/op.py` -- `PaddedRMSNorm.emit()`: demonstrates padding metadata (`input_bytes`, `padded_bytes`) for compute-side handling
- `blaze/ops/sdpa/op.py` -- `SDPADecode.emit()`: mask CB allocation (`cb_mask_in`) and how mask dimensions derive from padded tensor dimensions
- `blaze/context.py` -- `FusionResult`: graph output node carrying ShapeDescriptor to the unpadding boundary
- `blaze/compiler.py` -- `BlazeCompiler.compile()`: compilation path where output ShapeDescriptors are finalized
- `blaze/utils.py` -- `interpret_tile_padded()`: computes padded width determining how much padding to strip

---

[Previous: File 03 -- Automatic Padding Insertion and Propagation](./03_automatic_padding_insertion_and_propagation.md) | [Next: Chapter 6 -- Data Format Selection](../ch06_data_formats/)
