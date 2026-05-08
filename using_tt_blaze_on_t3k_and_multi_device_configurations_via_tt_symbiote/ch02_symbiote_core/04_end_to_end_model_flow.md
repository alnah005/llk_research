# 04 -- End-to-End Model Flow

This file walks through a complete single-device (or mesh-device) model acceleration example using the `test_glm_4_7.py` pattern. It explains how a HuggingFace model is loaded, how modules are replaced and initialized, how inference runs transparently through `model.generate()`, how the `compose_transforms` pipeline processes inputs at each stage, how `past_key_value` kwargs are handled, how timing data is collected, and provides a complete minimal runnable example. Prerequisites: all previous files in this chapter.

## The GLM-4 Test Pattern

The file [test_glm_4_7.py:test_glm] provides the canonical end-to-end example. The full flow consists of eight stages:

```
+---------------------------+
| 1. Load model + tokenizer |    AutoModelForCausalLM.from_pretrained("zai-org/GLM-4.7")
+---------------------------+
            |
            v
+---------------------------+
| 2. Define replacement map |    {nn.Linear: TTNNLinearLLama, nn.SiLU: TTNNSilu}
+---------------------------+
            |
            v
+---------------------------+
| 3. Replace modules        |    register_module_replacement_dict(model, nn_to_ttnn, ...)
+---------------------------+
            |
            v
+---------------------------+
| 4. Set device             |    set_device(model, mesh_device)
+---------------------------+
            |
            v
+---------------------------+
| 5. Preprocess weights     |    for module in all_modules: module.preprocess_weights()
+---------------------------+
            |
            v
+---------------------------+
| 6. Move weights to device |    for module in all_modules: module.move_weights_to_device()
+---------------------------+
            |
            v
+---------------------------+
| 7. Inference              |    model.generate(**inputs, max_new_tokens=128)
+---------------------------+
            |
            v
+---------------------------+
| 8. Profile                |    DispatchManager.save_stats_to_file("timing.csv")
+---------------------------+
```

## Stage 1: Load Model and Tokenizer

```python
# [test_glm_4_7.py]
tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-4.7")
model = AutoModelForCausalLM.from_pretrained("zai-org/GLM-4.7")
```

At this point, the model is a standard HuggingFace `PreTrainedModel` containing only `nn.Module` layers. No TTNN involvement yet.

### Preparing Inputs

```python
messages = [{"role": "user", "content": "What is your favorite condiment? ..."}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)
```

The inputs are standard PyTorch tensors (`input_ids`, `attention_mask`) on CPU. They will be automatically wrapped into `TorchTTNNTensor` during the first module call.

## Stage 2: Module Replacement

```python
nn_to_ttnn = {
    nn.Linear: TTNNLinearLLama,
    nn.SiLU: TTNNSilu,
}
persistent_weights = set()  # No exclusions in this test
modules1 = register_module_replacement_dict(
    model, nn_to_ttnn, model_config=None,
    exclude_replacement=persistent_weights
)
```

The map keys are PyTorch classes, values are `TTNNModule` subclasses. Each value class must implement `from_torch()` and `forward()`. The granularity of the map determines the acceleration boundary: only `Linear` and `SiLU` are replaced here; attention mechanisms, layer norms, and other operations remain as PyTorch code but automatically dispatch individual `aten` ops to TTNN via `__torch_dispatch__` (when a `TorchTTNNTensor` is involved).

### After Replacement

```
  model (AutoModelForCausalLM)
    |
    +-- model.embed_tokens (nn.Embedding)           <-- Unchanged
    +-- model.layers[0] (GLMBlock)
    |     +-- self_attn
    |     |     +-- q_proj (TTNNLinearLLama)         <-- Replaced
    |     |     +-- k_proj (TTNNLinearLLama)         <-- Replaced
    |     |     +-- v_proj (TTNNLinearLLama)         <-- Replaced
    |     |     +-- o_proj (TTNNLinearLLama)         <-- Replaced
    |     +-- mlp
    |     |     +-- gate_proj (TTNNLinearLLama)      <-- Replaced
    |     |     +-- up_proj (TTNNLinearLLama)        <-- Replaced
    |     |     +-- down_proj (TTNNLinearLLama)      <-- Replaced
    |     |     +-- act_fn (TTNNSilu)                <-- Replaced
    |     +-- input_layernorm (RMSNorm)              <-- Unchanged
    |     +-- post_attention_layernorm (RMSNorm)     <-- Unchanged
    +-- model.layers[1] ... [N-1]                    <-- Same pattern
    +-- model.norm (RMSNorm)                         <-- Unchanged
    +-- lm_head (nn.Linear)                          <-- Replaced (unless excluded)
```

## Stage 3: Device Initialization

```python
set_device(model, mesh_device)
```

`set_device(model, mesh_device)` recursively binds all TTNNModules to hardware -- see [File 01, "set_device() -- Binding Modules to Hardware"](./01_module_replacement_engine.md) for details on the four operations it performs (device assignment, bypass flag setting, timing hooks, and graph visualization).

## Stage 4: Weight Preprocessing and Device Transfer

```python
all_modules = {**modules1}
for k, v in tqdm(all_modules.items()):
    v.preprocess_weights()
    if not isinstance(v, TTNNLinearLLama):
        v.move_weights_to_device()
```

### Why Two Stages?

In the GLM test, `TTNNLinearLLama` modules have their `move_weights_to_device()` deferred -- it is not called eagerly in this loop. Instead, `module_run()` calls `move_weights_to_device()` idempotently before the first `forward()`, so the weights are transferred synchronously at that point. This is functionally equivalent to calling it in the loop above, but skipping it here means the `tqdm` progress bar does not reflect device transfer progress for linear layers. Non-linear modules (like `TTNNSilu`) have their weights moved eagerly in the loop, giving visibility via `tqdm`.

> **Warning:** The explicit `preprocess_weights()` loop before inference is required. Without it, the first `model.generate()` call would trigger lazy preprocessing inside `module_run()`, which works but adds significant latency to the first token. Pre-processing in a separate loop with `tqdm` gives the user visibility into progress. In traced mode, the `preprocess_weights()` and `move_weights_to_device()` methods will **assert** (not perform) during trace execution. Forgetting this setup step produces assertion errors.

## Stage 5: Inference

```python
model.eval()
torch.set_grad_enabled(False)
outputs = model.generate(**inputs, max_new_tokens=1, use_cache=True)
DispatchManager.clear_timings()
outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
```

### The Two-Inference Pattern

The test runs `model.generate()` twice:

1. **First run** (`max_new_tokens=1`): Warm-up. Triggers JIT compilation, weight lazy loading, KV cache allocation, and trace warm-up (if TRACED mode). Timing data from this run is discarded via `DispatchManager.clear_timings()`.

2. **Second run** (`max_new_tokens=128`): Production run. Timing data reflects steady-state performance.

### How model.generate() Works Transparently

The HuggingFace `model.generate()` method is completely unmodified. It works because:

1. **TorchTTNNTensor masquerades as torch.Tensor**: `TorchTTNNTensor` is a `torch.Tensor` subclass. All standard tensor operations (`isinstance(t, torch.Tensor)`, `t.shape`, `t.dtype`, `t.device`) work normally. HuggingFace's generation loop checks these properties to manage KV caches, attention masks, and token selection.

2. **Module calls are transparent**: When `model.generate()` calls `model(input_ids, ...)`, the call propagates through the HuggingFace model's `forward()` method, which calls child modules. Replaced modules route through `TTNNModule.__call__()` -> `module_run()` -> `forward()`, while unreplaced modules run on PyTorch as usual.

3. **The dispatch layer handles transitions**: When a PyTorch module (e.g., `nn.Embedding`) produces a `torch.Tensor` and passes it to a `TTNNModule`, the `module_run()` method's transform pipeline wraps it into a `TorchTTNNTensor` and converts to TTNN. When a `TTNNModule` produces output that flows to a PyTorch module, the `__torch_dispatch__` layer handles conversion back to PyTorch.

> **Warning:** Not all HuggingFace operations will automatically dispatch to TTNN. Operations like `torch.where`, `torch.cat`, or complex indexing may fall back to CPU if the default dispatcher lacks a handler for that specific `aten` op. Check the timing CSV for operations labeled with backend `"Torch"` to identify fallback hotspots.

### What Happens During Each Forward Pass

For each token generation step, `model.generate()` calls `model.forward()`, which triggers this cascade:

```
  model.forward(input_ids, attention_mask, ...)
       |
       v
  model.embed_tokens(input_ids)         -- nn.Embedding (PyTorch)
       |                                    Returns torch.Tensor
       v
  model.layers[0](hidden_states, ...)   -- GLMBlock (nn.Module)
       |
       +-- input_layernorm(hidden_states)  -- RMSNorm (PyTorch)
       |     Returns torch.Tensor
       |
       +-- self_attn.q_proj(hidden_states)  -- TTNNLinearLLama
       |     module_run() called:
       |       1. compose_transforms(wrap, to_ttnn, set_device)
       |       2. preprocess_weights() [idempotent]
       |       3. move_weights_to_device() [idempotent]
       |       4. forward() -> ttnn.linear(...)
       |       5. post_process -> wrap result in TorchTTNNTensor
       |     Returns TorchTTNNTensor
       |
       +-- [Attention computation with PyTorch ops]
       |     __torch_dispatch__ routes each op:
       |       torch.matmul(q, k.T) -> can_dispatch_to_ttnn? -> TTNN or CPU
       |       torch.softmax(scores) -> can_dispatch_to_ttnn? -> TTNN or CPU
       |
       +-- self_attn.o_proj(attn_output)    -- TTNNLinearLLama
       +-- mlp.gate_proj / up_proj / down_proj / act_fn
       |
       v
  model.layers[1]...[N-1]               -- Same pattern
       |
       v
  model.norm(hidden_states)              -- RMSNorm (PyTorch)
       |
       v
  lm_head(hidden_states)                -- TTNNLinearLLama (or nn.Linear if excluded)
       |
       v
  logits -> model.generate() selects next token
```

## The compose_transforms Pipeline in Detail

The three-stage pipeline (`wrap_to_torch_ttnn_tensor` -> `to_ttnn_wrap` -> `set_device_wrap`) from [File 02, "The compose_transforms Pipeline"](./02_dispatch_and_tensor_wrapping.md#the-compose_transforms-pipeline) applies here. For the GLM-4 model, a typical activation tensor traces through as follows: a `[1, 128, 4096]` float32 `torch.Tensor` is wrapped into a `TorchTTNNTensor`, converted via `ttnn.from_torch` to `ttnn.float32`, and placed on the target device via `ttnn.to_device`. After `forward()` completes, `post_process_ttnn_module_output` re-wraps the result into a `TorchTTNNTensor` for compatibility with downstream PyTorch code.

In bypass mode (child `TTNNModule` with `_bypass_tensor_wrapping = True`), the input is already a `ttnn.Tensor` on device, so `fast_unwrap_to_device` passes it through as a no-op -- no wrapping, no `from_torch` conversion, no distributed config assignment. See [File 02, "fast_unwrap_to_device"](./02_dispatch_and_tensor_wrapping.md#fast_unwrap_to_device) for the implementation.

## The past_key_value Exclusion

A notable implementation detail in `module_run()`: keyword arguments containing `"past_key_value"` in their key name are excluded from the transform pipeline:

```python
# [run_config.py:NormalRun.module_run] (simplified)
other_kwargs = {k: v for k, v in kwds.items() if "past_key_value" not in k}
func_kwargs = _map(transform, other_kwargs)
func_kwargs.update({k: v for k, v in kwds.items() if "past_key_value" in k})
```

This is because HuggingFace's KV-cache objects are complex nested structures (containing non-tensor members such as cache metadata, sequence length counters, and index pointers) that cannot be meaningfully converted to TTNN tensors. They are passed through as-is to the module's `forward()`.

> **Warning:** If you write a custom `TTNNModule` that accepts keyword arguments with `"past_key_value"` in the name, those arguments will bypass all tensor transformation. Your `forward()` implementation must handle them in their original PyTorch form.

## Stage 6: Timing and Profiling

The timing infrastructure (three recording levels, CSV format, `DisableTiming()`) is covered in [File 02, "DispatchManager -- Timing and Context Tracking"](./02_dispatch_and_tensor_wrapping.md#dispatchmanager----timing-and-context-tracking). In the GLM test, profiling is collected via:

```python
DispatchManager.save_stats_to_file("glm_timing_stats.csv")
```

This produces two output files: `glm_timing_stats.csv` (raw per-operation rows) and `glm_timing_stats_pivot.csv` (aggregated pivot table). For programmatic analysis:

```python
df = DispatchManager.get_timing_entries_stats()
# df is a pandas DataFrame with columns: attrs, module_name, func_name, duration, backend
ttnn_ops = df[df["backend"] == "TTNN"]
top_slow = ttnn_ops.groupby("func_name")["duration"].sum().sort_values(ascending=False)
```

## Device Parameterization for Multi-Topology Testing

The test fixture in `test_glm_4_7.py` demonstrates how to parameterize device selection:

```python
@pytest.mark.parametrize(
    "mesh_device",
    [{
        "N150": (1, 1),
        "N300": (1, 2),
        "T3K": (1, 8),
        "TG": (8, 4),
        "P150": (1, 1),
        "P300": (1, 2),
        "P150x4": (1, 4),
        "P150x8": (1, 8),
        "BHGLX": (8, 4),
    }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
def test_glm(mesh_device):
    ...
```

The `MESH_DEVICE` environment variable selects the topology, and the fixture maps it to a mesh shape tuple. The `device_params` fixture configures trace region size and command queue count:

```python
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 50000000, "num_command_queues": 1}],
    indirect=True,
)
```

The `trace_region_size` parameter reserves DRAM for trace buffers. A value of 50 MB is typical for medium-sized models; larger models or deeper trace stacks may require more.

## Complete Minimal Example

Combining all components, here is the minimal code to accelerate a model with TT-Symbiote:

```python
import os
os.environ["TT_SYMBIOTE_RUN_MODE"] = "NORMAL"
os.environ["TT_SYMBIOTE_DISPATCHER"] = "DEFAULT"

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.experimental.tt_symbiote.core.run_config import DispatchManager
from models.experimental.tt_symbiote.modules.linear import TTNNLinearLLama
from models.experimental.tt_symbiote.modules.activation import TTNNSilu
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict
from models.experimental.tt_symbiote.utils.device_management import set_device

import ttnn

# 1. Load
tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-4.7")
model = AutoModelForCausalLM.from_pretrained("zai-org/GLM-4.7")

# 2. Replace
modules = register_module_replacement_dict(
    model,
    {nn.Linear: TTNNLinearLLama, nn.SiLU: TTNNSilu},
)

# 3. Device setup
mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
set_device(model, mesh_device)

# 4. Weight pipeline
for m in modules.values():
    m.preprocess_weights()
    m.move_weights_to_device()

# 5. Inference
model.eval()
torch.set_grad_enabled(False)
inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(outputs[0]))

# 6. Profile
DispatchManager.save_stats_to_file("timing.csv")
ttnn.close_mesh_device(mesh_device)
```

## Key Takeaways

1. **Module replacement is the entry point**: All acceleration begins with `register_module_replacement_dict()`. The mapping dict determines which layers run on TTNN.

2. **Transparency is the core design goal**: `TorchTTNNTensor` masquerading as `torch.Tensor` allows HuggingFace's `model.generate()` and all standard PyTorch code to work without modification.

3. **The pipeline is: wrap -> convert -> place**: Inputs flow through `wrap_to_torch_ttnn_tensor` -> `to_ttnn_wrap` -> `set_device_wrap`. Bypass mode collapses this to a single `fast_unwrap_to_device`.

4. **past_key_value kwargs are passed through as-is**: HuggingFace KV-cache objects bypass the tensor transform pipeline.

5. **Profiling is built in**: Every operation is timed. `save_stats_to_file()` provides actionable data for identifying bottlenecks and fallback operations. Use `DispatchManager.DisableTiming()` in production.

6. **The two-run pattern is standard**: Warm up with a short generation, clear timings, then measure the production run.

---

**Next:** [Chapter 3 -- TT-Symbiote Multi-Device Support](../ch03_symbiote_multi_device/index.md)
