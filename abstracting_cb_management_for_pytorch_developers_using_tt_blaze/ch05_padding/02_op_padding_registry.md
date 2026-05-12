# Op Padding Registry

File 01 established that each padding strategy has specific mathematical requirements. This section formalizes that knowledge into a machine-readable registry: a `PaddingPolicy` enum, a `PaddingSpec` dataclass, per-op and per-input-port padding declarations, extensions to `TensorPort` and `OpSpec` to carry padding metadata, and default policies for every current Blaze op. The registry is the central lookup that the automatic padding insertion pass (File 03) uses to determine what fill value to apply at each graph edge.

> *Design Principle P4 (Provide Defaults with Per-Decision Overrides):* The registry provides a correct default padding strategy for every registered op and input port. A developer can override the strategy per-tensor, per-op, or globally. The default is always safe; the override is for performance optimization or escape-hatch correctness.

**What you will learn:**

- The `PaddingPolicy` enum with six variants (ZERO, NEG_INF, MASK, IDENTITY, NONE, INHERIT) and the `PaddingSpec` dataclass that associates each with fill values and validation predicates
- The registry design: how each Blaze op declares its required padding strategy per input port, with the asymmetric case where different inputs to the same op require different strategies (SDPA: ZERO for Q/K/V, NEG_INF for mask)
- How `TensorPort` (from `blaze/graph.py`) is extended with a `padding_strategy` field, and `OpSpec` gains `padding_policies` and `output_padding_propagation`
- The complete default policy table for all current Blaze ops, derived from examining each op's kernel semantics
- Format-specific fill value encoding via `encode_fill_value()`, validated against SDPA's `identity_scalar_packed` and `zero_scalar_packed` constants
- The compatibility matrix between producer and consumer padding policies
- Three levels of validation: registration-time, pass-time, and golden-reference testing
- CI integration via `STRICT_PADDING_REGISTRY` mode that rejects unregistered ops

---

## The PaddingPolicy Enum

### Definition

```python
# PROPOSED -- PaddingPolicy enum
from enum import Enum
from dataclasses import dataclass
from typing import Optional

class PaddingPolicy(Enum):
    """Padding strategy for tile-boundary alignment.

    Each variant specifies how to fill positions in the padded shape
    that do not correspond to real data in the logical shape.
    """
    ZERO = "zero"           # Fill with 0.0 (additive identity)
    NEG_INF = "neg_inf"     # Fill with -inf (softmax-safe)
    MASK = "mask"           # Fill with 0.0 + carry a boolean mask
    IDENTITY = "identity"   # Fill with the op's identity element
    NONE = "none"           # No padding (tile-aligned)
    INHERIT = "inherit"     # Pass through producer's padding (data movement ops)
```

The `INHERIT` policy (from V2's analysis) applies to data movement ops (mcast, gather, copy) that do not transform data. These ops should propagate whatever padding their producer established, rather than declaring a fixed strategy.

### The PaddingSpec Dataclass

```python
# PROPOSED -- PaddingSpec dataclass
@dataclass(frozen=True)
class PaddingSpec:
    """Complete padding specification for a single tensor port.

    Separates the type-level (PaddingPolicy) from the instance-level
    (fill_value, identity_op) to enable both static validation and
    runtime configuration.
    """
    policy: PaddingPolicy
    fill_value: Optional[float] = None
    identity_op: Optional[str] = None
    requires_mask_cb: bool = False

    def __post_init__(self):
        # Auto-derive fill values from policy
        if self.policy == PaddingPolicy.ZERO:
            assert self.fill_value is None or self.fill_value == 0.0
            object.__setattr__(self, "fill_value", 0.0)
        elif self.policy == PaddingPolicy.NEG_INF:
            assert self.fill_value is None or self.fill_value == float("-inf")
            object.__setattr__(self, "fill_value", float("-inf"))
        elif self.policy == PaddingPolicy.MASK:
            object.__setattr__(self, "fill_value", 0.0)
            object.__setattr__(self, "requires_mask_cb", True)
        elif self.policy == PaddingPolicy.IDENTITY:
            if self.identity_op == "mul":
                object.__setattr__(self, "fill_value", 1.0)
            elif self.identity_op == "add":
                object.__setattr__(self, "fill_value", 0.0)
            elif self.identity_op == "max":
                object.__setattr__(self, "fill_value", float("-inf"))
            elif self.identity_op == "min":
                object.__setattr__(self, "fill_value", float("inf"))
        elif self.policy == PaddingPolicy.NONE:
            object.__setattr__(self, "fill_value", None)
        elif self.policy == PaddingPolicy.INHERIT:
            object.__setattr__(self, "fill_value", None)

    def validate_format_compatibility(self, data_format: str) -> list[str]:
        """Check if this spec's fill value is representable in the given format.

        Returns a list of warnings (empty if fully compatible).
        """
        warnings = []
        fill = self.fill_value
        if fill is None:
            return warnings

        if data_format in ("bfloat8_b", "bfloat4_b"):
            import math
            if math.isinf(fill):
                warnings.append(
                    f"PaddingSpec({self.policy.value}): fill value {fill} is not "
                    f"exactly representable in {data_format}. The most extreme "
                    f"representable value (~-1e30 for bfloat8_b) will be used."
                )
            if fill == 1.0 and data_format == "bfloat4_b":
                warnings.append(
                    f"PaddingSpec(IDENTITY_MUL): fill value 1.0 in bfloat4_b "
                    f"may have reduced precision."
                )

        if data_format in ("uint8", "uint16", "uint32"):
            import math
            if fill < 0 or math.isinf(fill):
                warnings.append(
                    f"PaddingSpec({self.policy.value}): fill value {fill} is not "
                    f"representable in unsigned format {data_format}."
                )

        return warnings
```

### Convenience Constructors

```python
# PROPOSED -- convenience constructors
ZERO_PAD = PaddingSpec(policy=PaddingPolicy.ZERO)
NEG_INF_PAD = PaddingSpec(policy=PaddingPolicy.NEG_INF)
MASK_PAD = PaddingSpec(policy=PaddingPolicy.MASK)
MUL_IDENTITY_PAD = PaddingSpec(policy=PaddingPolicy.IDENTITY, identity_op="mul")
ADD_IDENTITY_PAD = PaddingSpec(policy=PaddingPolicy.IDENTITY, identity_op="add")
MAX_IDENTITY_PAD = PaddingSpec(policy=PaddingPolicy.IDENTITY, identity_op="max")
MIN_IDENTITY_PAD = PaddingSpec(policy=PaddingPolicy.IDENTITY, identity_op="min")
NO_PAD = PaddingSpec(policy=PaddingPolicy.NONE)
INHERIT_PAD = PaddingSpec(policy=PaddingPolicy.INHERIT)
```

### Why an Enum + Dataclass Instead of Just Strings?

The `padding_strategy` field in `ShapeDescriptor` (Chapter 3) is a string for serialization simplicity. The `PaddingPolicy` enum + `PaddingSpec` dataclass add:

1. **Fill value association:** The string `"NEG_INF"` does not carry `-inf`; the spec does.
2. **Format validation:** `validate_format_compatibility()` catches NEG_INF in bfloat8_b at registration time.
3. **Mask requirement tracking:** MASK is the only strategy requiring a companion tensor.
4. **Exhaustiveness checking:** A `match` on the enum can be checked for completeness.

> **Warning:** The string-based `padding_strategy` in ShapeDescriptor and the `PaddingPolicy` enum must stay synchronized. A `PaddingSpec.policy.value` must equal the ShapeDescriptor string. Any mismatch causes the padding pass to look up the wrong strategy.

---

## Format-Specific Fill Value Encoding

The fill value must be encoded into the target data format's bit representation for NCRISC copy-and-fill operations:

```python
# PROPOSED -- fill value encoding per format
def encode_fill_value(spec: PaddingSpec, data_format: str) -> int:
    """Encode a PaddingSpec's fill value into the target format's bit representation.

    Returns a uint32 containing two packed bfloat16 values (for NCRISC
    paired writes) or a single float32 value.
    """
    if spec.policy in (PaddingPolicy.NONE, PaddingPolicy.INHERIT):
        return 0

    value = spec.fill_value
    if value is None:
        return 0

    if data_format in ("bfloat16", "float16"):
        return _pack_bf16_pair(value)
    elif data_format == "float32":
        return _float_to_bits(value)
    elif data_format in ("bfloat8_b", "bfloat4_b"):
        import math
        if math.isinf(value) and value < 0:
            return _bfp_neg_max(data_format)
        elif math.isinf(value) and value > 0:
            return _bfp_pos_max(data_format)
        else:
            return _float_to_bfp_bits(value, data_format)
    else:
        raise ValueError(f"Cannot encode fill value for format '{data_format}'")


def _pack_bf16_pair(value: float) -> int:
    """Pack two identical bfloat16 values into a uint32.

    Mirrors the pattern from SDPA: identity_scalar_packed = 0x3F803F80.
    """
    import struct
    f32_bits = struct.unpack(">I", struct.pack(">f", value))[0]
    bf16_bits = f32_bits >> 16
    return (bf16_bits << 16) | bf16_bits
```

### Validation Against Known Constants

```
encode_fill_value(ZERO_PAD, "bfloat16"):
  value = 0.0 -> f32 bits: 0x00000000 -> bf16: 0x0000
  packed: 0x00000000
  Matches SDPA's zero_scalar_packed = 0x00000000  [PASS]

encode_fill_value(MUL_IDENTITY_PAD, "bfloat16"):
  value = 1.0 -> f32 bits: 0x3F800000 -> bf16: 0x3F80
  packed: 0x3F803F80
  Matches SDPA's identity_scalar_packed = 0x3F803F80  [PASS]

encode_fill_value(NEG_INF_PAD, "bfloat16"):
  value = -inf -> f32 bits: 0xFF800000 -> bf16: 0xFF80
  packed: 0xFF80FF80
```

---

## Extending TensorPort with Padding Strategy

### Current TensorPort (EXISTING)

```python
# EXISTING -- from blaze/graph.py
@dataclass
class TensorPort:
    name: str
    dtype: Optional[str] = None
    tile_shape: Optional[tuple[int, int]] = None
```

### Extended TensorPort (PROPOSED)

```python
# PROPOSED -- TensorPort with padding
@dataclass
class TensorPort:
    """A typed tensor port on an op (input or output).

    Extended with padding_strategy for developer escape hatches.
    If None, the padding pass looks up strategy from the registry.
    """
    name: str
    dtype: Optional[str] = None
    tile_shape: Optional[tuple[int, int]] = None
    padding_strategy: Optional[PaddingSpec] = None  # NEW
    padding_fill: Optional[float] = None            # NEW
```

The new fields default to `None`, preserving backward compatibility. The padding pass interprets `None` as "use the registry default."

> **Warning:** The `NONE`/`None` distinction matters: `None` means "unspecified, use registry default," while `NO_PAD` means "explicitly no padding." Adding `padding_strategy` to TensorPort is non-breaking, but serialization code must handle the new field.

---

## Extending OpSpec with Padding Metadata

### Current OpSpec (EXISTING)

```python
# EXISTING -- from blaze/graph.py (simplified)
@dataclass
class OpSpec:
    name: str
    kernel_path: str
    op_class: str
    input_ports: list[TensorPort] = field(default_factory=list)
    output_ports: list[TensorPort] = field(default_factory=list)
    # ... existing fields ...
```

### Extended OpSpec (PROPOSED)

```python
# PROPOSED -- OpSpec with padding metadata
@dataclass
class OpSpec:
    """Specification for a registered op type.

    Extended with padding_policies and output_padding_propagation.
    """
    name: str
    kernel_path: str
    op_class: str
    input_ports: list[TensorPort] = field(default_factory=list)
    output_ports: list[TensorPort] = field(default_factory=list)
    # ... existing fields ...

    # NEW: padding policy declarations
    padding_policies: dict[str, PaddingSpec] = field(default_factory=dict)
    """Per-port padding policies. Key is port name, value is PaddingSpec."""

    output_padding_propagation: str = "inherit"
    """How output padding is derived from input padding:
    - 'inherit': output matches the dominant input (default)
    - 'reset_zero': output is always zero-padded regardless of input
    - 'reset_none': output has no padding (op guarantees tile alignment)
    """

    def __post_init__(self):
        if self.padding_policies:
            # Cross-validate port names
            spec_ports = set(self.padding_policies.keys())
            defined_ports = {p.name for p in self.input_ports}
            unknown = spec_ports - defined_ports
            if unknown:
                import warnings
                warnings.warn(
                    f"Op '{self.name}' padding spec references ports "
                    f"{unknown} not in input_ports."
                )
            unregistered = defined_ports - spec_ports
            if unregistered and self.padding_policies:
                import warnings
                warnings.warn(
                    f"Op '{self.name}' ports {unregistered} have no "
                    f"padding policy. They will use default ZERO."
                )
```

> **Warning:** The `__post_init__` cross-validation is critical. If the padding spec references a port name that does not exist (e.g., a typo like `"in_0"` instead of `"in0"`), the policy is registered but never applied. The tensor on that port gets the default ZERO padding, which may be wrong for softmax or reduction ops.

---

## Default Policies for All Current Blaze Ops

### The Complete Registry

```python
# PROPOSED -- complete padding policy registry
PADDING_REGISTRY: dict[str, dict[str, PaddingSpec]] = {
    # === Matmul Family ===
    "matmul":                {"in0": ZERO_PAD, "in1": ZERO_PAD},
    "matmul_cb":             {"in0": ZERO_PAD, "in1": ZERO_PAD},
    "kn_sliced_matmul":      {"in0": ZERO_PAD, "in1": ZERO_PAD},
    "dram_streaming_matmul": {"in0": ZERO_PAD, "in1": ZERO_PAD},

    # === Normalization Family ===
    "rmsnorm":               {"input": NO_PAD, "gamma": NO_PAD},
    "padded_rmsnorm":        {"input": ZERO_PAD, "gamma": ZERO_PAD},
    "broadcast_rmsnorm":     {"input": ZERO_PAD, "gamma": ZERO_PAD},
    "broadcast_rmsnorm_mcast": {"input": ZERO_PAD, "gamma": ZERO_PAD},
    "layernorm":             {"input": MASK_PAD, "gamma": ZERO_PAD, "beta": ZERO_PAD},

    # === Elementwise Family ===
    "add":                   {"in0": ZERO_PAD, "in1": ZERO_PAD},
    "mul":                   {"in0": ZERO_PAD, "in1": ZERO_PAD},
    "eltwise_mul":           {"in0": ZERO_PAD, "in1": ZERO_PAD},
    "residual_add":          {"in0": ZERO_PAD, "in1": ZERO_PAD},
    "sub":                   {"in0": ZERO_PAD, "in1": ZERO_PAD},
    "silu":                  {"input": ZERO_PAD},
    "clamped_silu":          {"input": ZERO_PAD},
    "relu":                  {"input": ZERO_PAD},
    "gelu":                  {"input": ZERO_PAD},

    # === Attention Family ===
    "sdpa_decode":           {"q_in": ZERO_PAD, "k_in": ZERO_PAD, "v_in": ZERO_PAD},
    "softmax":               {"scores": NEG_INF_PAD},

    # === Reduction Family ===
    "reduce":                {"input": ZERO_PAD},  # Default: sum reduction
    "gated_reduce":          {"gate": ZERO_PAD, "up": ZERO_PAD},
    "gated_local_reduce":    {"input": ZERO_PAD},

    # === Data Movement Family (INHERIT from producer) ===
    "mcast":                 {"in0": INHERIT_PAD},
    "gather":                {"in0": INHERIT_PAD},
    "copy":                  {"in0": INHERIT_PAD},

    # === Utility Ops ===
    "retilize":              {"input": ZERO_PAD},
    "argmax":                {"input": NEG_INF_PAD},
}
```

### Registry Lookup Functions

```python
# PROPOSED -- registry API
def register_padding(op_type: str, port_policies: dict[str, PaddingSpec]) -> None:
    """Register padding requirements for an op type."""
    existing = PADDING_REGISTRY.get(op_type)
    if existing is not None and existing != port_policies:
        raise ValueError(
            f"Op '{op_type}' already registered with different policies."
        )
    PADDING_REGISTRY[op_type] = port_policies


def lookup_padding(op_type: str, port_name: str) -> PaddingSpec:
    """Look up the required padding policy for an op's input port.

    Falls back to ZERO if the op is not registered.
    """
    spec = PADDING_REGISTRY.get(op_type)
    if spec is None:
        import warnings
        warnings.warn(
            f"Op '{op_type}' not in padding registry. Defaulting to ZERO."
        )
        return ZERO_PAD
    if port_name in spec:
        return spec[port_name]
    # Wildcard fallback: check for "*" key
    if "*" in spec:
        return spec["*"]
    import warnings
    warnings.warn(
        f"Op '{op_type}' has no policy for port '{port_name}'. Defaulting to ZERO."
    )
    return ZERO_PAD
```

The wildcard `"*"` fallback (from V2) enables ops with many identically-padded ports to avoid enumerating each one.

---

## Per-Input-Port Padding Differences

The SDPA case illustrates why per-port policies are essential:

```
SDPA: q, k, v, mask -> output

q: [B, H, S, D] padded with ZERO
  Zero-padded query positions produce zero attention scores
  These zero scores are masked to -inf by the attention mask

k: [B, H, S, D] padded with ZERO
  Zero-padded key positions produce zero in Q @ K^T

v: [B, H, S, D] padded with ZERO
  Zero-padded value positions contribute 0 to weighted sum

mask: [B, H, S, S] padded with NEG_INF      <-- DIFFERENT
  Padded mask positions must be -inf to prevent attention to padding
```

> **Warning:** If the mask tensor for SDPA is zero-padded instead of NEG_INF-padded, the softmax treats padded positions as having mask value 0 (meaning "do not mask"), and attention weights are computed over padded key positions. The result is attention probability leaking to non-existent positions.

### Elementwise Multiply: ZERO, Not IDENTITY

```
Elementwise multiply (mul, eltwise_mul): ZERO
  a * 0 = 0; padded OUTPUT positions are 0
  This is correct because we want padded output to be zero.

Product REDUCTION (reduce_prod): IDENTITY (1.0)
  prod(..., 1, 1, ..., 1) = prod of real elements only
  This is correct because we want padding to not affect the accumulated product.
```

> **Warning:** Confusing "elementwise multiply" (output = a * b, padded output should be 0) with "product reduction" (output = prod(a_i), padded elements should be 1) is a common error.

---

## Registry Lookup Algorithm

```python
# PROPOSED -- resolution with fallback chain
def resolve_padding_for_port(
    op_type: str,
    port_name: str,
    *,
    override: Optional[PaddingSpec] = None,
    producer_spec: Optional[PaddingSpec] = None,
) -> PaddingSpec:
    """Resolve the padding policy for a specific tensor port.

    Resolution order:
    1. Developer override (Level 1 escape hatch)
    2. PADDING_REGISTRY[op_type][port_name] (registry default)
    3. Producer's output padding (for INHERIT ops)
    4. ZERO_PAD (ultimate fallback)
    """
    # Priority 1: Developer override
    if override is not None:
        return override

    # Priority 2: Registry lookup
    spec = lookup_padding(op_type, port_name)

    # Priority 3: Resolve INHERIT
    if spec.policy == PaddingPolicy.INHERIT:
        if producer_spec is not None:
            return producer_spec
        return ZERO_PAD  # No producer info -> default to ZERO

    return spec
```

### Walkthrough: Lookup for Attention Block

```
Step 1: Q projection (matmul)
  resolve("matmul", "in0") -> ZERO_PAD
  resolve("matmul", "in1") -> ZERO_PAD
  Output: ZERO (from output_padding_propagation="reset_zero")

Step 2: QK score computation (matmul)
  Same: ZERO for both inputs
  Output: QK scores with ZERO padding

Step 3: Softmax on QK scores
  resolve("softmax", "scores") -> NEG_INF_PAD
  ** MISMATCH: producer output is ZERO, consumer needs NEG_INF **
  -> Padding pass inserts re-padding: ZERO -> NEG_INF

Step 4: Attention output (matmul of softmax probs with V)
  resolve("matmul", "in0") -> ZERO_PAD
  probs from softmax output: ZERO (exp(-inf)=0)
  Compatible -> no re-padding needed
```

---

## Compatibility Matrix

Not all padding transitions require re-padding. The compatibility matrix defines when the data can be used as-is:

| Producer \ Consumer | ZERO | NEG_INF | MASK | IDENTITY | NONE | INHERIT |
|---------------------|------|---------|------|----------|------|---------|
| **ZERO** | ok | REPAD | ok* | REPAD | ok** | ok |
| **NEG_INF** | REPAD | ok | REPAD | ok*** | error | ok |
| **MASK** | ok | REPAD | ok | REPAD | error | ok |
| **IDENTITY** | REPAD | ok*** | REPAD | ok | error | ok |
| **NONE** | ok | ok | ok | ok | ok | ok |

\* ZERO -> MASK: compatible if consumer handles count correction internally.
\** ZERO -> NONE: compatible only if no padding was actually applied.
\*** NEG_INF -> IDENTITY(max): compatible because -inf is the max identity.

### Mismatch Diagnostic

```python
# PROPOSED -- mismatch diagnostic
@dataclass
class PaddingMismatch:
    """Diagnostic for a padding strategy mismatch at a graph edge."""
    producer_op: str
    producer_port: str
    producer_policy: PaddingPolicy
    consumer_op: str
    consumer_port: str
    consumer_policy: PaddingPolicy
    resolution: str  # "repad", "compatible", "error"

    def __str__(self):
        return (
            f"Padding mismatch: {self.producer_op}.{self.producer_port} "
            f"({self.producer_policy.value}) -> "
            f"{self.consumer_op}.{self.consumer_port} "
            f"({self.consumer_policy.value}): {self.resolution}"
        )
```

---

## Validation Strategies

### Level 1: Registration-Time Validation

When an `OpSpec` with `padding_policies` is registered, validate port name consistency and format compatibility:

```python
# PROPOSED -- registration-time validation
def _validate_registration(op_type: str, policies: dict[str, PaddingSpec],
                            input_ports: list[TensorPort]) -> None:
    defined_ports = {p.name for p in input_ports}
    spec_ports = set(policies.keys()) - {"*"}
    unknown = spec_ports - defined_ports
    if unknown:
        import warnings
        warnings.warn(f"Op '{op_type}' padding references unknown ports {unknown}.")
    for port_name, spec in policies.items():
        for fmt in ("bfloat16", "bfloat8_b", "bfloat4_b"):
            ws = spec.validate_format_compatibility(fmt)
            for w in ws:
                import warnings
                warnings.warn(f"[{op_type}.{port_name}] {w}")
```

### Level 2: Padding-Pass-Time Validation

After the padding pass, verify no unresolved mismatches remain (see File 03).

### Level 3: Golden-Reference Testing

```python
# PROPOSED -- golden reference test pattern
def test_padding_policy_for_op(op_type, port_name, test_shape, tile_shape=(32,32)):
    """Test that the registered policy produces correct results,
    AND that a wrong policy produces incorrect results."""
    policy = lookup_padding(op_type, port_name)
    x = torch.randn(test_shape)

    ref = compute_op_cpu(op_type, x)
    x_padded = apply_padding(x, policy, tile_shape)
    result = unpad(compute_op_tiled(op_type, x_padded), test_shape)

    assert torch.allclose(ref, result, atol=1e-3, rtol=1e-3)

    # Verify WRONG policy fails
    wrong_policies = [p for p in [ZERO_PAD, NEG_INF_PAD, MUL_IDENTITY_PAD]
                      if p != policy]
    for wrong in wrong_policies:
        x_wrong = apply_padding(x, wrong, tile_shape)
        result_wrong = unpad(compute_op_tiled(op_type, x_wrong), test_shape)
        if torch.allclose(ref, result_wrong, atol=1e-3, rtol=1e-3):
            warnings.warn(
                f"Wrong policy {wrong.policy.value} also correct for "
                f"{op_type}.{port_name}. Try a different test shape."
            )
```

> **Warning:** The "wrong policy" test is essential: it confirms that the correct policy is not merely coincidentally producing the right answer. If the wrong policy also passes, the test shape does not exercise the padding boundary -- try a non-aligned shape like `[1, 37]`.

---

## CI Integration: STRICT_PADDING_REGISTRY Mode

To prevent regressions when new ops are added, the padding registry supports a strict mode for CI:

```python
# PROPOSED -- strict mode for CI
STRICT_PADDING_REGISTRY = os.environ.get("BLAZE_STRICT_PADDING", "0") == "1"

def lookup_padding_strict(op_type: str, port_name: str) -> PaddingSpec:
    """Strict-mode lookup that raises on unregistered ops.

    In CI (BLAZE_STRICT_PADDING=1), any op without a padding
    registry entry is a hard error, not a warning.
    """
    spec = PADDING_REGISTRY.get(op_type)
    if spec is None:
        if STRICT_PADDING_REGISTRY:
            raise PaddingRegistryError(
                f"Op '{op_type}' is not in the padding registry. "
                f"All ops must register padding policies when "
                f"BLAZE_STRICT_PADDING=1 is set."
            )
        import warnings
        warnings.warn(f"Op '{op_type}' not in registry. Defaulting to ZERO.")
        return ZERO_PAD
    return spec.get(port_name, spec.get("*", ZERO_PAD))
```

This ensures that adding a new Blaze op without registering its padding requirements fails CI rather than silently using a possibly-incorrect default.

---

## Override Mechanism

### Three Tiers (Matching P4: Defaults with Per-Decision Overrides)

| Tier | Scope | Mechanism | Use Case |
|------|-------|-----------|----------|
| **Per-tensor** | Single `adapt()` call | `padding=NEG_INF_PAD` kwarg | Developer knows the correct strategy |
| **Per-op** | All ports of one op | `with blaze.padding_override(op=..., policy=...)` | Custom reduction ops |
| **Global** | Entire pipeline | `with blaze.no_auto_padding()` | Debugging, manual control |

```python
# PROPOSED -- per-tensor override
act_typed = adapter.adapt(activation, op_hint="softmax.scores",
                          padding=NEG_INF_PAD)

# PROPOSED -- per-op override
with blaze.padding_override(op="my_custom_reduce", policy=MUL_IDENTITY_PAD):
    result = custom_reduce(input_tensor)

# PROPOSED -- global override (expert escape hatch)
with blaze.no_auto_padding():
    result = manual_pipeline(input_tensor)
```

> **Warning:** Global override disabling automatic padding is an expert escape hatch. Using it incorrectly produces silent numerical errors. The context manager should log a warning when entered.

---

## Complete Default Policy Table

| Op Type | Port | Policy | Fill Value | Rationale |
|---------|------|--------|-----------|-----------|
| matmul | in0, in1 | ZERO | 0.0 | `a * 0 = 0`, `0 * w = 0` |
| matmul_cb | in0, in1 | ZERO | 0.0 | Same as matmul |
| kn_sliced_matmul | in0, in1 | ZERO | 0.0 | Same as matmul |
| dram_streaming_matmul | in0, in1 | ZERO | 0.0 | Same as matmul |
| add / residual_add | in0, in1 | ZERO | 0.0 | `a + 0 = a` |
| mul / eltwise_mul | in0, in1 | ZERO | 0.0 | `a * 0 = 0` (output is 0) |
| sub | in0, in1 | ZERO | 0.0 | `a - 0 = a` |
| silu / relu / gelu | input | ZERO | 0.0 | `f(0) = 0` for all three |
| rmsnorm | input, gamma | NONE | N/A | Only for tile-aligned widths |
| padded_rmsnorm | input, gamma | ZERO | 0.0 | Non-aligned; `0^2 = 0`, scalar correction |
| layernorm | input | MASK | 0.0 | Variance needs correct N |
| layernorm | gamma, beta | ZERO | 0.0 | Parameters pad with identity |
| softmax | scores | NEG_INF | -inf | `exp(-inf) = 0` |
| sdpa | q, k, v | ZERO | 0.0 | Zero scores, zero weights |
| sdpa | mask | NEG_INF | -inf | Mask padded positions |
| reduce (sum) | input | ZERO | 0.0 | `sum + 0 = sum` |
| reduce (max) | input | IDENTITY | -inf | `max(x, -inf) = x` |
| reduce (prod) | input | IDENTITY | 1.0 | `prod * 1 = prod` |
| gated_reduce | gate, up | ZERO | 0.0 | `silu(0) * 0 = 0` |
| mcast / gather / copy | in0 | INHERIT | (producer) | Pass-through |
| argmax | input | NEG_INF | -inf | Padded positions must not win |

---

## Key Takeaways

- The `PaddingPolicy` enum (ZERO, NEG_INF, MASK, IDENTITY, NONE, INHERIT) and `PaddingSpec` dataclass formalize the padding taxonomy from File 01 into a machine-readable format. Each spec carries the policy, fill value, identity operation, mask CB requirement, and format validation.
- The padding registry maps every Blaze op and input port to its required `PaddingSpec`. Matmul family uses ZERO; softmax uses NEG_INF; LayerNorm uses MASK; reductions use IDENTITY parameterized by reduction op; data movement ops use INHERIT. Missing or mistyped port names silently fall back to ZERO.
- `TensorPort` gains `padding_strategy` and `OpSpec` gains `padding_policies` and `output_padding_propagation`. Both extensions are backward-compatible. The `__post_init__` cross-validation catches port name mismatches at registration time.
- The registry lookup follows a four-level priority: developer override, registry per-port, wildcard `"*"` fallback, producer inheritance for INHERIT ops. This mirrors P4 (Provide Defaults with Per-Decision Overrides).
- The compatibility matrix between producer and consumer policies determines when the padding pass (File 03) must insert re-padding. The most common mismatch is ZERO (matmul output) -> NEG_INF (softmax input) at the attention boundary.
- Three validation levels catch errors at different stages: registration-time (port names, format compatibility), pass-time (producer-consumer mismatches), and test-time (golden reference with intentional wrong-policy tests). CI integration via `STRICT_PADDING_REGISTRY` mode rejects unregistered ops.

## Source Files

- `blaze/blaze_op.py` -- `Input`, `Output`, `Internal` descriptors; `OpSpec` (extended with `padding_policies`)
- `blaze/graph.py` -- `TensorPort` (extended with `padding_strategy`), `OpSpec`, `Edge`
- `blaze/ops/padded_rmsnorm/op.py` -- `PaddedRMSNorm`: canonical ZERO padding with scalar correction
- `blaze/ops/sdpa/op.py` -- `SDPADecode`: implicit NEG_INF via `identity_scalar_packed = 0x3F803F80`, `zero_scalar_packed = 0x00000000`
- `blaze/cb_engine.py` -- `DTYPE_BYTES`: format-specific byte sizes for fill value validation

---

← Previous: [Chapter 4: Automatic Tile Decomposition](../ch04_tile_decomposition/) | Next: [Chapter 6: Data Format Selection](../ch06_data_formats/) →

[Previous file: 01 -- Padding Strategy Taxonomy](./01_padding_strategy_taxonomy.md) | [Next file: 03 -- Automatic Padding Insertion](./03_automatic_padding_insertion_and_propagation.md)
