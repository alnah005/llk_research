# Agent B Review: Chapter 2 -- Pass 1

1. **File:** `02_dispatch_and_tensor_wrapping.md`, lines 246-256 and 461-474; also `04_end_to_end_model_flow.md`, lines 243-274.
   **Error:** `to_ttnn_wrap` is described as returning the `TorchTTNNTensor` wrapper ("Returns the TorchTTNNTensor (now with both `.elem` and `.ttnn_tensor` set)"). In the source (`run_config.py` lines 345-350), the function does `e = e.to_ttnn` which reassigns `e` to the result of the `.to_ttnn` property -- and that property returns `self.ttnn_tensor`, a raw `ttnn.Tensor`, not the `TorchTTNNTensor` wrapper. Consequently, the compose_transforms pipeline description is wrong about its intermediate and output type: the pipeline yields a raw `ttnn.Tensor` on device, not a `TorchTTNNTensor`. This same error appears in the stage-by-stage trace in file 04 (Step 2 output description and Step 3 input assumption). A reader implementing a similar pipeline would build code that expects `TorchTTNNTensor` wrappers after the transform, when in fact `forward()` receives raw `ttnn.Tensor` objects via this path.
   **Fix:** State that `to_ttnn_wrap` returns a raw `ttnn.Tensor` (the value of `.ttnn_tensor`). Update the compose_transforms diagram and the file 04 trace to reflect that steps 2 and 3 operate on `ttnn.Tensor`, and that the overall pipeline output is a `ttnn.Tensor` on device -- consistent with the bypass path yielding the same type. Note that `post_process_ttnn_module_output` re-wraps the result into `TorchTTNNTensor` after `forward()`.

2. **File:** `02_dispatch_and_tensor_wrapping.md`, lines 261-268.
   **Error:** The `set_device_wrap` code snippet shows only the `TorchTTNNTensor` branch (`isinstance(e, TorchTTNNTensor) and e.ttnn_tensor is not None`). The actual source (`run_config.py` lines 362-374) has a preceding `isinstance(e, ttnn.Tensor)` branch that handles bare `ttnn.Tensor` inputs. This branch is the one actually exercised in the standard compose_transforms pipeline (because `to_ttnn_wrap` returns a `ttnn.Tensor`, per issue 1 above). A reader copying this code would produce a function that silently passes `ttnn.Tensor` through without moving it to the target device.
   **Fix:** Include the `isinstance(e, ttnn.Tensor)` branch in the code snippet, and note that this is the primary path in the compose_transforms pipeline.

3. **File:** `02_dispatch_and_tensor_wrapping.md`, lines 233-236.
   **Error:** The `wrap_to_torch_ttnn_tensor` code snippet only shows the `isinstance(e, torch.Tensor) and not isinstance(e, TorchTTNNTensor)` condition. The actual source (`run_config.py` lines 336-342) has a second condition: `if not isinstance(e, TorchTTNNTensor) and isinstance(e, ttnn.Tensor): result = TorchTTNNTensor(e)`. This branch wraps bare `ttnn.Tensor` objects into `TorchTTNNTensor`. A reader implementing a custom version of this function would fail to handle `ttnn.Tensor` inputs.
   **Fix:** Add the second condition to the code snippet.

4. **File:** `01_module_replacement_engine.md`, lines 375-377.
   **Error:** The text states that both `named_modules()` and `named_children()` "handle `TTNNModule`, `nn.Module`, dict, list, and tuple containers." In the source (`module.py` lines 237-268), only `named_children()` handles dict, list, and tuple containers. `named_modules()` iterates `self.__dict__.items()` and only yields items that are direct `TTNNModule` or `nn.Module` instances -- it does not descend into dict, list, or tuple containers. A reader relying on `named_modules()` to discover modules stored in container attributes would miss them.
   **Fix:** Clarify that `named_modules()` only recurses into direct `TTNNModule` and `nn.Module` attributes in `__dict__`, while `named_children()` additionally handles dict, list, and tuple containers.

5. **File:** `04_end_to_end_model_flow.md`, lines 133-143.
   **Error:** The text implies that in the GLM test, `TTNNLinearLLama` modules intentionally skip `move_weights_to_device()` in the setup loop so that device transfer happens lazily inside `module_run()` to "overlap weight transfer with computation." This is presented as an optimization design. However, the actual `NormalRun.module_run()` in the source (`run_config.py` lines 624-625) calls both `preprocess_weights()` and `move_weights_to_device()` synchronously before `forward()` with timing instrumentation between them -- there is no overlap with computation. The lazy path is functionally equivalent to the eager path, not an optimization for overlapping transfer and compute. A reader may incorrectly design their weight pipeline around an assumed overlap that does not exist.
   **Fix:** Remove the claim that lazy device transfer enables overlap with computation. Instead, state that the GLM test defers `move_weights_to_device()` for `TTNNLinearLLama` modules because `module_run()` will call it idempotently before the first `forward()` -- functionally equivalent but without the `tqdm` progress visibility for that specific module type.

---

# Agent B Review: Chapter 2 -- Pass 2

1. **File:** `04_end_to_end_model_flow.md`, line 259.
   **Error:** The concrete pipeline trace states: "Converts `hidden_states` to bfloat16 via `torch_dtype_to_ttnn_dtype()`." The example input is a `torch.Tensor` with `dtype float32`. However, the `TORCH_TO_TTNN` mapping in `utils.py` maps `torch.float32` to `ttnn.float32`, not `ttnn.bfloat16`. Only `torch.float16` maps to `ttnn.bfloat16`. A reader following this trace for a float32 model would incorrectly believe that the pipeline introduces a precision-reducing float32-to-bfloat16 conversion, when in fact float32 is preserved as ttnn.float32. This affects any reader reasoning about numerical precision through the pipeline.
   **Fix:** Change "Converts `hidden_states` to bfloat16 via `torch_dtype_to_ttnn_dtype()`" to "Converts `hidden_states` to `ttnn.float32` via `torch_dtype_to_ttnn_dtype()`" (or state the general rule: the dtype is mapped according to the `TORCH_TO_TTNN` table, where float32 stays float32 and float16 becomes bfloat16).

2. **File:** `03_run_modes.md`, lines 256-261.
   **Error:** The `_compute_tensor_signature` code snippet shows only the `isinstance(tensor, ttnn.Tensor)` and `isinstance(tensor, torch.Tensor)` branches. The actual source (`run_config.py` lines 875-884) has an intermediate branch: `if hasattr(tensor, "ttnn_tensor") and tensor.ttnn_tensor is not None`, which extracts the underlying `ttnn_tensor` and computes the signature from its shape, dtype, **and layout**. This branch fires for `TorchTTNNTensor` inputs (which are the standard inputs in non-bypass mode) and produces a different signature than the `torch.Tensor` fallback (which omits layout). A reader implementing custom trace logic or debugging cache key mismatches would incorrectly conclude that `TorchTTNNTensor` inputs fall through to the `torch.Tensor` branch (since `TorchTTNNTensor` is a `torch.Tensor` subclass), producing signatures without layout information -- when in reality the `ttnn_tensor` branch fires first and includes layout. Two inputs with the same shape and dtype but different ttnn layouts would NOT share a cache key, contrary to what the simplified snippet implies.
   **Fix:** Add the `TorchTTNNTensor` branch to the code snippet and note that for `TorchTTNNTensor` inputs, the signature is computed from the underlying `ttnn_tensor`'s properties (shape, dtype, and layout), not the wrapper's torch properties.

---

# Agent B Review: Chapter 2 — Pass 3

No feedback — chapter approved.
