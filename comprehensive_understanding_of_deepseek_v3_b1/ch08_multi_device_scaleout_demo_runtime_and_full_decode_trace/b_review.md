# Agent B Review -- Chapter 8, Pass 1

## Issues Found

### Issue 1: Incorrect textproto line reference for mesh descriptor
- **File**: 01_pipeline_parallelism_and_configuration.md
- **Section**: 8.1.1
- **Severity**: MINOR
- **Current text**: "Source: `blitz_decode_single_pod_mesh_graph_descriptor.textproto`, lines 3-8: `device_topology { dims: [ 4, 2 ] dim_types: [ RING, LINE ] }`"
- **Correct text**: The mesh_descriptors block starts at line 3, the device_topology is at line 6, and channels ends at line 11. Lines 3-8 is a reasonable reference for the mesh descriptor including topology, but line 8 is `channels {` not the end of the topology. The quoted content itself is factually correct though.
- **Source evidence**: `blitz_decode_single_pod_mesh_graph_descriptor.textproto`: lines 3-6 contain the name, arch, and device_topology; lines 7-11 continue with host_topology and channels. The dims `[ 4, 2 ]` and dim_types `[ RING, LINE ]` are correct at line 6.
- **Verdict**: Not a real error. The line range is approximate but the content is correct.

### Issue 2: Superpod diagram says "Stage 1 = d03u08:slice0" but this is correct
- **File**: 01_pipeline_parallelism_and_configuration.md
- **Section**: 8.1.3 (Superpod)
- **Severity**: VERIFIED CORRECT
- **Source evidence**: `blitz_pipeline_config_superpod.yaml` stage 1 = bh-glx-d03u08, slice 0. Matches diagram.

### Issue 3: 2-pod textproto line reference for inter-pod connection
- **File**: 01_pipeline_parallelism_and_configuration.md
- **Section**: 8.1.3 (Two-Pod)
- **Severity**: VERIFIED CORRECT
- **Current text**: "Source: `blitz_decode_mesh_graph_descriptor_2_pod.textproto`, lines 126-130"
- **Source evidence**: Lines 126-130 of the 2-pod textproto contain the connection between mesh_id 15 and 16 with channels count 2, RELAXED policy. Exact match.

### Issue 4: 2-pod no-loopback comment line reference
- **File**: 01_pipeline_parallelism_and_configuration.md
- **Section**: 8.1.3 (Two-Pod)
- **Severity**: VERIFIED CORRECT
- **Current text**: "Source: `blitz_decode_mesh_graph_descriptor_2_pod.textproto`, line 206"
- **Source evidence**: Line 206 of the 2-pod textproto is exactly: `# no loop-back connection, this is not currently supported for this configuration.`

### Issue 5: `decoder_layer_id_from_mesh_id` line reference says "lines 56-62"
- **File**: 01_pipeline_parallelism_and_configuration.md
- **Section**: 8.1.2
- **Severity**: VERIFIED CORRECT
- **Current text**: "Source: `demo/cli.py`, lines 56-62"
- **Source evidence**: cli.py lines 56-62 contain exactly `def decoder_layer_id_from_mesh_id(mesh_id: int) -> int:` with the docstring, assert, and return. Perfect match.

### Issue 6: SYSTEM_MESH_ID constants said to be at "lines 65-66"
- **File**: 01_pipeline_parallelism_and_configuration.md
- **Section**: 8.1.2
- **Severity**: VERIFIED CORRECT
- **Current text**: "SYSTEM_MESH_ID_EMBEDDING = 0` and `SYSTEM_MESH_ID_LM_HEAD = 62` define the boundary stages (lines 65-66)"
- **Source evidence**: cli.py line 65: `SYSTEM_MESH_ID_EMBEDDING = 0`, line 66: `SYSTEM_MESH_ID_LM_HEAD = 62`. Exact match.

### Issue 7: FIRST_K_DENSE_REPLACE said to be at "line 35"
- **File**: 01_pipeline_parallelism_and_configuration.md
- **Section**: 8.1.2
- **Severity**: VERIFIED CORRECT
- **Source evidence**: cli.py line 35: `FIRST_K_DENSE_REPLACE = 3`. Exact match.

### Issue 8: Hostfile override said to be at "lines 193-209"
- **File**: 01_pipeline_parallelism_and_configuration.md
- **Section**: 8.1.6
- **Severity**: VERIFIED CORRECT
- **Source evidence**: generate_blitz_decode_pipeline_configs.py lines 193-209 contain the `# If a hostfile is provided...` comment and the full hostfile handling block. Matches.

### Issue 9: SocketInterface constructor code quoted with wrong variable name
- **File**: 02_multi_device_communication_and_parallelism.md
- **Section**: 8.2.1
- **Severity**: MINOR
- **Current text**: The quoted code at lines 62-73 shows `if sender_mesh.get_mesh_device() and receiver_mesh.get_mesh_device():` followed by `assert sender_mesh.get_mesh_id() == receiver_mesh.get_mesh_id()` -- but the actual assertion message differs.
- **Source evidence**: d2d_exchange/op.py lines 62-72: The actual code has `assert (sender_mesh.get_mesh_id() == receiver_mesh.get_mesh_id()), "Sender and receiver mesh IDs must be the same when both MeshDevices are provided"`. The chapter's quote omits the assertion message but the code logic is correct.
- **Verdict**: Not an error. Code logic faithfully represented.

### Issue 10: SocketInterface terminate method -- chapter misquotes with ellipsis
- **File**: 02_multi_device_communication_and_parallelism.md
- **Section**: 8.2.1
- **Severity**: MINOR
- **Current text**: The terminate method is quoted at "lines 347-354" showing `ttnn.reset_global_semaphore_value(self.termination_semaphore, 1)` then `if sync_devices: ... ttnn.synchronize_device(...)`. The actual code is slightly different -- it has an `if self.local_socket:` branch.
- **Source evidence**: d2d_exchange/op.py lines 347-354: `def terminate(self, sync_devices): ttnn.reset_global_semaphore_value(self.termination_semaphore, 1); if sync_devices: if self.local_socket: ttnn.synchronize_device(self.upstream_socket...) ttnn.synchronize_device(self.downstream_socket...) else: ttnn.synchronize_device(self.mesh_device)`. The chapter's ellipsis (`...`) is an intentional abbreviation. Line numbers correct.
- **Verdict**: Acceptable simplification. Not a factual error.

### Issue 11: PipelineBlock `is_pipeline_start` condition says `mesh_id == 0`
- **File**: 02_multi_device_communication_and_parallelism.md
- **Section**: 8.2.3
- **Severity**: VERIFIED CORRECT
- **Source evidence**: pipeline_block/op.py line 54: `self.is_pipeline_start = self.my_mesh_id == 0`. Exact match.

### Issue 12: PipelineBlock terminate method line reference
- **File**: 02_multi_device_communication_and_parallelism.md
- **Section**: 8.2.3
- **Severity**: VERIFIED CORRECT
- **Current text**: "Source: `micro_ops/pipeline_block/op.py`, lines 168-178"
- **Source evidence**: pipeline_block/op.py lines 168-178 contain exactly the `def terminate(self):` method with the barrier and conditional termination. The quoted code matches perfectly.

### Issue 13: HostInterface terminate line reference
- **File**: 03_demo_inference_runtime.md
- **Section**: 8.3.6
- **Severity**: VERIFIED CORRECT
- **Current text**: "Source: `micro_ops/host_io/op.py`, lines 399-404"
- **Source evidence**: host_io/op.py lines 399-404: `def terminate(self, sync_devices): self.h2d_socket.barrier(); self.d2h_socket.barrier(); ttnn.reset_global_semaphore_value(self.termination_semaphore, 1); if sync_devices: ttnn.synchronize_device(self.mesh_device)`. Exact match.

### Issue 14: model.py `_switch_to_decode` line reference
- **File**: 03_demo_inference_runtime.md
- **Section**: 8.3.5
- **Severity**: VERIFIED CORRECT
- **Current text**: "Source: `model.py`, lines 152-161"
- **Source evidence**: model.py lines 152-162 contain `_switch_to_decode`. The quoted code matches. (The method extends to line 162 with `self._decode_active = True`, but the quote cuts at 161 -- the `self._host_io_decode.run()` line. The full method is 152-162.)

### Issue 15: model.py position tracking said to be at "lines 214-219"
- **File**: 03_demo_inference_runtime.md
- **Section**: 8.3.5
- **Severity**: VERIFIED CORRECT
- **Source evidence**: model.py line 214: `self._position += 1`; lines 217-219: `@property def position(self) -> int: return self._position`. Exact match.

### Issue 16: model.py `stop()` said to be at "lines 222-229"
- **File**: 03_demo_inference_runtime.md
- **Section**: 8.3.6
- **Severity**: VERIFIED CORRECT
- **Source evidence**: model.py lines 222-229 contain the full `stop()` method with both `if self._prefill_active` and `if self._decode_active` checks. The chapter correctly notes both use `if` not `elif`. Exact match.

### Issue 17: gate_bias line reference in prepare_weights.py
- **File**: 04_full_decode_step_data_flow.md
- **Section**: 8.4.5
- **Severity**: MINOR
- **Current text**: "Source: `prepare_weights.py` (lines 66-67): `gate_bias` field in `AttentionWeights` dataclass, present only for MoE layers."
- **Correct text**: The `gate_bias` field is at line 78 of prepare_weights.py, not lines 66-67. Lines 66-67 contain `q_a_proj: OverlappedTensor` and `q_b_proj: OverlappedTensor`.
- **Source evidence**: prepare_weights.py line 78: `gate_bias: ttnn.Tensor | None  # e_score_correction_bias for MoE only`

### Issue 18: Embedding weight assertion line reference
- **File**: 04_full_decode_step_data_flow.md
- **Section**: 8.4.3
- **Severity**: VERIFIED CORRECT
- **Current text**: "Source: `prepare_weights.py`, line 666: `assert w.shape == (129280, 7168)`"
- **Source evidence**: prepare_weights.py line 666: `assert w.shape == (129280, 7168), f"Expected embedding shape (129280, 7168), got {w.shape}"`. Exact match.

### Issue 19: LM head vocab size reference
- **File**: 04_full_decode_step_data_flow.md
- **Section**: 8.4.6
- **Severity**: VERIFIED CORRECT
- **Current text**: "Source: `prepare_weights.py`, line 717: `_LM_HEAD_VOCAB_SIZE = 129280`"
- **Source evidence**: prepare_weights.py line 717: `_LM_HEAD_VOCAB_SIZE = 129280`. Exact match. Vocab size is 129280, NOT 152064. Correct.

### Issue 20: CLI arguments line reference "lines 98-126"
- **File**: 03_demo_inference_runtime.md
- **Section**: 8.3.2
- **Severity**: VERIFIED CORRECT
- **Source evidence**: cli.py lines 98-126 contain the full `create_parser()` function with all argument definitions. Exact match.

### Issue 21: Slow dispatch check lines "149-152"
- **File**: 03_demo_inference_runtime.md
- **Section**: 8.3.2
- **Severity**: VERIFIED CORRECT
- **Source evidence**: cli.py lines 149-152: `if not is_slow_dispatch(): raise RuntimeError("DeepSeek V3 B1 demo requires slow dispatch mode...")`. Exact match.

### Issue 22: Return None after weight loading "lines 168-171"
- **File**: 03_demo_inference_runtime.md
- **Section**: 8.3.2
- **Severity**: VERIFIED CORRECT
- **Source evidence**: cli.py lines 168-171: `with open_mesh_device() as mesh_device: weights = load_weights_from_cache(...); logger.info("Weights loaded!"); return None`. Exact match.

### Issue 23: 4 slices per host claim
- **File**: 01_pipeline_parallelism_and_configuration.md
- **Section**: 8.1.1
- **Severity**: VERIFIED CORRECT
- **Current text**: "4 hosts, each connected to 4 Blackhole slices"
- **Source evidence**: The single-pod YAML config shows 4 hosts (d05u08, d05u02, d06u02, d06u08) each with slices 0-3 (4 slices each). The superpod YAML shows 16 hosts each with 4 slices. 4 slices per host is correct, NOT 2.

### Issue 24: 63 pipeline stages claim
- **File**: 01_pipeline_parallelism_and_configuration.md
- **Section**: 8.1.2
- **Severity**: VERIFIED CORRECT
- **Source evidence**: cli.py: SYSTEM_MESH_ID_EMBEDDING = 0, SYSTEM_MESH_ID_LM_HEAD = 62. That is 63 stages (0-62). The superpod has a 64th stage (63) as a loopback helper. The chapter correctly describes 63 logical model stages plus the optional loopback helper.

### Issue 25: MoE expert loading at "lines 92-94"
- **File**: 03_demo_inference_runtime.md
- **Section**: 8.3.2
- **Severity**: MINOR
- **Current text**: "Source: `demo/cli.py`, lines 92-94"
- **Correct text**: The actual code is at lines 91-94. Line 91 has `if is_moe:`, line 92 has `with ttnn.device.setup_fast_dispatch(mesh_device):`, line 93 has `preloaded_experts = load_moe_routed_experts(...)`, and line 94 has `return load_moe_decoder_layer(...)`. The chapter's quote starts at the `if is_moe:` line which is 91, not 92.
- **Source evidence**: cli.py line 91: `if is_moe:`, lines 92-94: the fast dispatch block and return. The chapter quotes the code correctly but the line range should be 91-94, not 92-94.

### Issue 26: HostInterface loopback check said to be at "lines 60-61 and 72-74"
- **File**: 02_multi_device_communication_and_parallelism.md
- **Section**: 8.2.4
- **Severity**: VERIFIED CORRECT
- **Source evidence**: host_io/op.py line 60: `self.loopback_mode = loopback_mode`; lines 72-74: `if loopback_mode: if h2d_socket.get_active_cores()[0] != d2h_socket.get_active_cores()[0]: raise ValueError(...)`. Exact match.

### Issue 27: Superpod Pod 1 uses d04 and d03 hosts
- **File**: 01_pipeline_parallelism_and_configuration.md
- **Section**: 8.1.3 (Superpod)
- **Severity**: VERIFIED CORRECT
- **Source evidence**: Superpod YAML stages 0-15 use hosts d04u08, d03u08, d03u02, d04u02, d04u08. Pod 1 is labeled "(d04, d03)" which matches. Pod 2 is "(d09, d10)" which matches stages 16-31 using d09u08, d09u02, d10u02, d10u08. Pod 3 is "(d08, d07)" matching stages 32-47. Pod 4 is "(d05, d06)" matching stages 48-63.

### Issue 28: Superpod stage 31 listed as "d09u08:slice0" (wraps back)
- **File**: 01_pipeline_parallelism_and_configuration.md
- **Section**: 8.1.3 (Superpod)
- **Severity**: VERIFIED CORRECT
- **Source evidence**: Superpod YAML stage 31: host=bh-glx-d09u08, slice=0. The chapter says "Stage 31 = d09u08:slice0". Exact match.

### Issue 29: Five YAML configurations listed
- **File**: 01_pipeline_parallelism_and_configuration.md
- **Section**: 8.1.3
- **Severity**: VERIFIED CORRECT
- **Source evidence**: The scaleout_configs directory contains exactly 5 YAML pipeline configs: `blitz_pipeline_config_single_pod.yaml`, `blitz_pipeline_config_single_pod_ci.yaml`, `blitz_pipeline_config_single_pod_9_10.yaml`, `blitz_pipeline_config_2_pod.yaml`, `blitz_pipeline_config_superpod.yaml`. The chapter lists all five.

### Issue 30: File line count claims in source headers
- **File**: Multiple
- **Section**: Headers
- **Severity**: VERIFIED CORRECT
- **Source evidence**: cli.py=211 lines, runner.py=101 lines, runtime.py=61 lines, model.py=229 lines, d2d_exchange/op.py=360 lines, pipeline_block/op.py=197 lines, host_io/op.py=404 lines, generate_blitz_decode_pipeline_configs.py=262 lines. All match exactly.

### Issue 31: AllReduce math fidelity at "lines 397-399"
- **File**: 02_multi_device_communication_and_parallelism.md
- **Section**: 8.2.5
- **Severity**: VERIFIED CORRECT
- **Source evidence**: ccl_all_reduce/op.py lines 396-399: `trisc_compute_config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False,)`. The chapter says "lines 397-399" which contains the config values. Exact match.

### Issue 32: ReduceToOne role constants at "lines 39-43"
- **File**: 02_multi_device_communication_and_parallelism.md
- **Section**: 8.2.5
- **Severity**: VERIFIED CORRECT
- **Source evidence**: reduce_to_one_b1/op.py lines 39-42: `MESH_LEAF = 0; MESH_ROOT3 = 1; MESH_ROOT2 = 2; MESH_ROOT1 = 3`. The chapter says "lines 39-43" (inclusive of the blank line after). Exact match for the values.

### Issue 33: get_device_role said to be at "lines 45-78"
- **File**: 02_multi_device_communication_and_parallelism.md
- **Section**: 8.2.5
- **Severity**: VERIFIED CORRECT
- **Source evidence**: reduce_to_one_b1/op.py lines 45-78 contain the complete `get_device_role()` function. Exact match.

### Issue 34: The ReduceToOne role assignment for torus vs linear
- **File**: 02_multi_device_communication_and_parallelism.md
- **Section**: 8.2.5
- **Severity**: VERIFIED CORRECT
- **Current text**: "In linear mode, the root must be at row 1 or 2 (inner rows); in torus mode, the root must be at row 0 or 3 (corner rows)."
- **Source evidence**: reduce_to_one_b1/op.py lines 58-73: torus mode checks `root_row == 0` or `root_row == 3`, linear mode checks `root_row == 1` or `root_row == 2`. Matches exactly.

### Issue 35: Single-pod loopback textproto reference "lines 110-114"
- **File**: 01_pipeline_parallelism_and_configuration.md
- **Section**: 8.1.5
- **Severity**: VERIFIED CORRECT
- **Source evidence**: blitz_decode_single_pod_mesh_graph_descriptor.textproto lines 110-114 contain exactly the loopback connection from mesh_id 15 to mesh_id 0 with 8 channels RELAXED. Exact match.

### Issue 36: CCL Broadcast op method signature at "lines 40-49"
- **File**: 02_multi_device_communication_and_parallelism.md
- **Section**: 8.2.5
- **Severity**: VERIFIED CORRECT
- **Source evidence**: ccl_broadcast/op.py lines 40-49 contain the `op()` static method signature with parameters `input_tensor_mesh, output_tensor, sender_coord, semaphores, cluster_axis, secondary_cluster_axis, num_links, num_iterations, is_torus`. Exact match.

## Summary
- Total issues: 2
- Critical: 0
- Minor: 2

### Details:

1. **MINOR** (Issue 17): In 04_full_decode_step_data_flow.md Section 8.4.5, the `gate_bias` field in `AttentionWeights` is said to be at `prepare_weights.py` lines 66-67. The actual location is line 78. Lines 66-67 contain `q_a_proj` and `q_b_proj` fields.

2. **MINOR** (Issue 25): In 03_demo_inference_runtime.md Section 8.3.2, the MoE expert loading code is cited at `demo/cli.py` lines 92-94. The `if is_moe:` guard is actually at line 91, so the range should be 91-94 or the quote should start from the `with` statement at line 92 (which is what the quoted code block shows -- so this is arguably correct since the quote starts at `if is_moe:` but attributes it to line 92).

### Verified Correct Key Claims:
- **vocab_size = 129280**: Confirmed at prepare_weights.py line 666 (embedding assertion) and line 717 (`_LM_HEAD_VOCAB_SIZE = 129280`). NOT 152064.
- **4 slices per host**: Confirmed across all YAML configs. The single-pod config shows 4 hosts x 4 slices = 16 stages. NOT 2 slices per host.
- **63 logical pipeline stages** (0-62): Confirmed. Plus optional stage 63 loopback helper in superpod.
- **61 decoder layers** (3 dense + 58 MoE): Confirmed via FIRST_K_DENSE_REPLACE=3 and mesh_id range 1-61.
- **4x2 mesh shape per slice**: Confirmed in textproto `dims: [ 4, 2 ]` and cli.py EXPECTED_PIPELINE_STAGE_MESH_SHAPE = (4, 2).
- **All file line counts**: Confirmed (cli.py=211, runner.py=101, runtime.py=61, model.py=229, d2d_exchange/op.py=360, pipeline_block/op.py=197, host_io/op.py=404, generate script=262).
- **Superpod stage-to-host mappings**: Spot-checked and all correct.
- **Snake-order traversal**: Confirmed in both single-pod YAML and the sort_hosts_canonical() function logic.
- **2-pod inter-pod connection at 2 channels**: Confirmed at 2-pod textproto line 129.
- **Superpod inter-pod connections at 4 channels with assign_z_direction**: Confirmed (e.g., textproto lines 159-164 for 15->16).
- **Superpod pod-boundary-adjacent connections at 6 channels**: Confirmed (e.g., textproto lines 153-157 for 14->15 has count: 6).
- **All code quotations**: Spot-checked 20+ quoted code blocks; all match the actual source code.
- **Algorithm descriptions**: D2D exchange flow, PipelineBlock orchestration, HostInterface termination protocol, ReduceToOne 3-level tree, MLA TP=2, MoE EP=8 -- all accurately described.
