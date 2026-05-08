# Chapter 2 -- TT-Symbiote Core Architecture

This chapter dissects the internals of TT-Symbiote: the framework that transparently intercepts PyTorch model execution and reroutes operations to Tenstorrent TTNN hardware. It covers the module replacement engine that swaps PyTorch layers for TTNN equivalents, the tensor wrapping and dispatch machinery that routes individual `aten` operations to device, the suite of run modes that control execution semantics (from pure-CPU debugging through traced zero-overhead replay), and finally an end-to-end walkthrough of accelerating a real model. By the end of this chapter you will understand the complete data path from `model.forward()` through to device-side TTNN kernel execution.

## Contents

1. [`01_module_replacement_engine.md`](./01_module_replacement_engine.md) -- Module Replacement Engine
   - `register_module_replacement_dict()` API, `TTNNModule` base class lifecycle, weight preprocessing pipeline, `exclude_replacement`, module name assignment, `set_device()`, and `DeviceInit`.

2. [`02_dispatch_and_tensor_wrapping.md`](./02_dispatch_and_tensor_wrapping.md) -- Dispatch and Tensor Wrapping
   - `TorchTTNNTensor` subclass, `__torch_dispatch__` routing, pluggable dispatcher system, `DistributedTensorConfig`, bypass tensor wrapping optimization, `fast_unwrap_to_device`, and `DispatchManager`.

3. [`03_run_modes.md`](./03_run_modes.md) -- Run Modes
   - All eight run modes (NORMAL through TRACED), `TracedRun` three-phase lifecycle, `TTNNLayerStack`, trace decorators, Ring topology requirement for traced CCL, pre/post trace hooks, and development workflow guidance.

4. [`04_end_to_end_model_flow.md`](./04_end_to_end_model_flow.md) -- End-to-End Model Flow
   - Complete single-device acceleration walkthrough using the GLM-4 test pattern, HuggingFace `model.generate()` transparency, the `compose_transforms` pipeline, `past_key_value` handling, timing/profiling, and a minimal runnable example.

## Prerequisites

- Familiarity with PyTorch `nn.Module`, `named_modules()`, and `torch.Tensor` subclassing.
- Chapter 1 (Device Topologies) for understanding of `MeshDevice`, `DeviceArch`, and mesh shapes.
