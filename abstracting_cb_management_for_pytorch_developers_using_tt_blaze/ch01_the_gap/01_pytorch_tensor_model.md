# The Gap Between PyTorch Tensors and Tenstorrent Hardware

PyTorch's programming model gives developers a remarkably frictionless experience: define a module, call forward, and shapes just work. This section establishes the implicit contracts that PyTorch developers rely on -- contracts that Tenstorrent hardware does not honor by default, creating the gap that an abstraction layer must bridge.

**What you will learn:**

- What a PyTorch tensor actually is: contiguous memory, arbitrary shapes, strides, and dtypes
- The four implicit contracts PyTorch upholds: shape inference, automatic broadcasting, type promotion, and transparent memory management
- How PyTorch developers think about computation at the module level
- What happens under the hood in a standard `torch.nn.Linear(512, 1024)` call

---

## What Is a PyTorch Tensor?

At its core, a `torch.Tensor` is a view over a contiguous (or strided) block of memory, annotated with:

| Property | Description | Example |
|----------|-------------|---------|
| **Storage** | A flat, one-dimensional array of typed elements | 524,288 `float32` values |
| **Shape** | N-dimensional size tuple | `(batch=8, seq=128, hidden=4096)` |
| **Strides** | Step sizes in elements per dimension | `(524288, 4096, 1)` for C-contiguous |
| **Dtype** | Element data type | `torch.float32`, `torch.bfloat16` |
| **Device** | Where the data lives | `cpu`, `cuda:0` |

A developer can create tensors of any shape -- `[1, 37]`, `[7, 2048]`, `[3, 5, 11, 13]` -- with no alignment constraints. The storage is a flat 1-D buffer; the shape and strides provide the multidimensional interpretation. **There is no concept of tiles, alignment boundaries, or fixed-size memory regions in PyTorch's model.**

The strides determine how indices map to linear memory offsets. For a row-major tensor with shape `[B, S, D]`:

$$\text{offset}(i, j, k) = i \cdot \text{stride}[0] + j \cdot \text{stride}[1] + k \cdot \text{stride}[2]$$

For a row-major tensor with shape `[2, 128, 4096]` and `float32` dtype:
- `strides = (524288, 4096, 1)` (elements, not bytes)
- Total memory: $2 \times 128 \times 4096 \times 4 = 4{,}194{,}304$ bytes (4 MB)

The shape can be any tuple of non-negative integers. There is no requirement that dimensions be powers of two, multiples of 32, or aligned to any hardware boundary:

```python
# All of these are valid PyTorch tensors
a = torch.randn(7, 37)        # Odd dimensions -- no problem
b = torch.randn(1)            # Scalar-like -- just works
c = torch.randn(3, 5, 768)    # 3D -- fine
d = torch.randn(1, 1, 1, 1)   # Four degenerate dimensions -- allowed
```

Strides enable views (transposes, slices, reshapes) without copying data. A transposed view of a `[512, 1024]` tensor has shape `(1024, 512)` and strides `(1, 1024)` -- the same underlying storage, just reinterpreted. PyTorch tracks contiguity automatically and materializes copies only when operations require contiguous memory.

---

## The Four Implicit Contracts

PyTorch developers operate under four assumptions that "just work." Understanding these is essential because each represents a responsibility that moves from the framework to the developer when targeting Tenstorrent hardware directly.

### Contract 1: Shape Inference

When composing operations, PyTorch automatically determines output shapes from input shapes. The developer never specifies them:

```python
# The developer writes:
linear = torch.nn.Linear(4096, 11008)
x = torch.randn(1, 4096)         # [1, 4096]
y = linear(x)                     # [1, 11008]  -- shape inferred automatically

# Under the hood, PyTorch computes:
#   y.shape = (x.shape[0], linear.out_features)
# No manual calculation needed.
```

For a matrix multiplication `[M, K] @ [K, N]`, PyTorch verifies that the inner dimensions match ($K = K$), produces output shape `[M, N]`, and raises a descriptive error if they do not match. The developer never writes code to compute output dimensions.

### Contract 2: Automatic Broadcasting

When operand shapes do not match, PyTorch applies NumPy-style broadcasting rules:

1. Right-align the shapes
2. Dimensions of size 1 are expanded to match the other operand
3. Missing dimensions are prepended as size 1

```python
# Bias addition in a transformer: [B, D] + [1, D]
activation = torch.randn(32, 1024)   # [32, 1024]
bias = torch.randn(1024)             # [1024]
result = activation + bias            # [32, 1024] -- bias broadcast along batch

# Attention mask application: [1, 1, S, S] * [B, H, S, S]
mask = torch.ones(1, 1, 128, 128)
scores = torch.randn(8, 32, 128, 128)
masked = scores * mask                # [8, 32, 128, 128] -- mask broadcast
```

The developer writes `a + b` or `a * b` and expects it to work regardless of compatible shape combinations. No explicit replication or expansion code is needed. Broadcasting is implemented through stride manipulation and does not allocate new memory for the expanded dimensions.

### Contract 3: Type Promotion

When operands have different dtypes, PyTorch promotes both to a common type:

```python
a = torch.randn(4096, dtype=torch.float32)
b = torch.randn(4096, dtype=torch.bfloat16)
c = a + b  # c.dtype == torch.float32 (promoted to the higher precision)
```

The rules follow a well-defined hierarchy: `bfloat16` < `float16` < `float32` < `float64`. Mixed-precision training relies on this: gradients may be `float32` while activations are `bfloat16`, and PyTorch handles the conversion at every operation boundary. The developer does not manually cast.

### Contract 4: Transparent Memory Management

PyTorch's memory allocator handles all allocation, deallocation, and data movement:

```python
# Memory is allocated when tensors are created
x = torch.randn(8, 4096)            # ~131 KB allocated on CPU

# Memory is freed when the tensor goes out of scope (reference counting + GC)
del x                                 # storage freed

# Device transfer is a single call
x_gpu = x.to('cuda:0')               # allocates GPU memory, copies data, frees on GC
```

The developer never computes buffer sizes, never decides where in a memory hierarchy to place data, and never coordinates producer/consumer synchronization between processing units. Intermediate results in a forward pass are allocated on demand and freed by the autograd engine when no longer needed for backward. There is no manual `malloc` or `free`, no pre-allocation of output buffers, and no size calculations.

---

## How PyTorch Developers Think About Computation

The standard development workflow has three steps:

1. **Define a module** by subclassing `torch.nn.Module`
2. **Implement `forward()`** using tensor operations
3. **Call the module** with input tensors -- shapes, types, and memory "just work"

Here is a minimal transformer MLP layer -- the kind of code a PyTorch developer writes daily:

```python
import torch
import torch.nn as nn

class GatedMLP(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj   = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, hidden_dim]
        gate = torch.nn.functional.silu(self.gate_proj(x))   # [batch, seq, intermediate]
        up = self.up_proj(x)                                   # [batch, seq, intermediate]
        return self.down_proj(gate * up)                       # [batch, seq, hidden_dim]
```

Notice what is absent from this code:

- No mention of tile shapes, padding, or alignment
- No buffer allocation or sizing
- No data format negotiation between layers
- No core placement or memory budgeting
- No synchronization primitives
- No explicit output shape calculation

The developer trusts PyTorch to handle all of this. When moving to Tenstorrent hardware, every one of these absent concerns becomes an explicit responsibility -- unless an abstraction layer bridges the gap.

---

## Under the Hood: `torch.nn.Linear(512, 1024)`

To make the implicit contracts concrete, let us trace what happens when a PyTorch developer calls a linear layer:

```python
linear = torch.nn.Linear(512, 1024)
x = torch.randn(1, 512)
y = linear(x)
```

### Step 1: Module Initialization

`nn.Linear(512, 1024)` creates:
- A weight parameter `self.weight` of shape `[1024, 512]` (note the transposition -- PyTorch stores weights as `[out_features, in_features]`)
- A bias parameter `self.bias` of shape `[1024]`, initialized to zero
- Both are initialized with Kaiming uniform initialization
- Total parameters: $1024 \times 512 + 1024 = 525{,}312$

The shapes are arbitrary integers. There is no padding to 32-element boundaries, no tile decomposition, no consideration of hardware data paths. PyTorch would handle `nn.Linear(513, 1023)` identically.

### Step 2: Forward Pass

`linear(x)` calls `F.linear(x, self.weight, self.bias)`, which computes:

$$y = x W^T + b$$

where $x$ has shape `[1, 512]`, $W^T$ has shape `[512, 1024]`, and $b$ has shape `[1024]`.

The sequence of implicit operations:

1. **Shape check:** Verifies `x.shape[-1] == weight.shape[1]` (both 512)
2. **Matrix multiply:** Computes `x @ weight.T` -- output shape is `[1, 1024]`
3. **Bias addition with broadcasting:** Adds bias `[1024]` via broadcasting (bias is broadcast from `[1024]` to `[1, 1024]`)
4. **Memory allocation:** A new tensor `y` of shape `[1, 1024]` is allocated (8,192 bytes in float32)
5. **Returns:** `y` with shape `[1, 1024]`, dtype `float32`

### Step 3: What the Developer Did NOT Specify

| Concern | PyTorch Handles It |
|---------|-------------------|
| How to decompose the matmul into blocks | Dispatched to BLAS (MKL, cuBLAS) |
| Output tensor size and allocation | Inferred from input shapes |
| Weight memory layout for the multiply | Transposed view via strides |
| Bias broadcast expansion | Implicit broadcasting rules |
| Intermediate precision | Follows dtype promotion rules |
| Memory cleanup of intermediates | Reference counting / GC |
| Parallelism across compute units | BLAS library handles threading/GPU blocks |

This is the developer experience that an abstraction layer must preserve when targeting Tenstorrent hardware: the developer writes `linear(x)` and gets `y`, with all the hardware-specific plumbing handled transparently.

### The Expectations That Transfer to Hardware

When a PyTorch developer approaches hardware acceleration, they carry these expectations:

| Expectation | PyTorch behavior |
|-------------|-----------------|
| Arbitrary shapes | Any `[M, K]` works for matmul as long as inner dimensions match |
| No alignment constraints | Dimensions need not be multiples of any hardware constant |
| Implicit broadcasting | `[1, D] + [B, D]` works without explicit replication |
| Automatic type handling | Mixed-precision ops "just work" via promotion rules |
| Zero memory management | Intermediates are allocated and freed automatically |
| Shape inference | Output shapes follow from operation semantics, never manually specified |

Every one of these expectations breaks when targeting Tenstorrent hardware directly. The next section explains why.

---

## Key Takeaways

- A PyTorch tensor is a view over contiguous typed memory with shape, strides, and dtype -- no tile alignment, no fixed-size regions, no hardware constraints.
- PyTorch upholds four implicit contracts: shape inference, automatic broadcasting, type promotion, and transparent memory management. Developers rely on all four without conscious thought.
- The developer's mental model is: define a module, call `forward()`, get a correctly shaped result. All plumbing is invisible.
- A single `nn.Linear(512, 1024)` call involves shape checking, matrix multiplication, bias broadcasting, and memory allocation -- all handled transparently.
- Every implicit contract becomes an explicit developer responsibility when targeting Tenstorrent hardware without an abstraction layer. This is the gap that motivates the TensorAdapter.

## Source Files

- PyTorch source: `torch/nn/modules/linear.py` (nn.Linear definition)
- PyTorch source: `torch/_refs/__init__.py` (broadcasting and type promotion rules)
- PyTorch documentation: [Broadcasting Semantics](https://pytorch.org/docs/stable/notes/broadcasting.html)

---

**Next:** [`02_tenstorrent_tile_cb_model.md`](./02_tenstorrent_tile_cb_model.md)
