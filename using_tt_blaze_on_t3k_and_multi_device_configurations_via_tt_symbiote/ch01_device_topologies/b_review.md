# Agent B Review: Chapter 1 — Pass 1

1. **File:** `03_fabric_and_routing.md`, lines 200-205. **Error:** The all-reduce skip condition is described as triggering when the target axis has size 1 ("Single device or trivial axis (`mesh_shape == [1,1]` or size-1 on the target axis)"). The actual source code in `ccl.py` line 176 checks `cluster_axis == 1 and 1 in list(mesh_device.shape)` -- it only skips for `cluster_axis == 1`, NOT for an arbitrary target axis. A call with `cluster_axis=0` on T3K (1,8) would NOT be skipped by this condition even though axis 0 has extent 1. A reader implementing the skip logic based on the chapter's description would produce a different (broader) guard than what the code actually does. **Fix:** Replace "size-1 on the target axis" with the precise condition: "specifically `cluster_axis == 1` and the mesh shape contains a dimension of size 1." Add a note that `cluster_axis=0` on a (1,N) mesh is not caught by this guard.

2. **File:** `03_fabric_and_routing.md`, lines 203-212. **Error:** The 1D all-reduce path is described as: "Uses `reduce_scatter_minimal_async` alone, since the mesh has only one meaningful dimension. There is no need for a separate all-gather step." This is technically accurate to the source code, but calling the result an "all-reduce" is misleading without qualification. A reduce-scatter produces a *sharded* result (each device holds 1/N of the reduced output), which is fundamentally different from a true all-reduce (every device holds the full result). A reader implementing a multi-device model based on this description would expect `tt_all_reduce` to return a replicated tensor on 1D meshes, but it actually returns a sharded tensor. **Fix:** Add a clarifying sentence: "Note that on 1D meshes, `tt_all_reduce` returns a *reduced-and-scattered* (sharded) result, not a fully replicated all-reduce. Each device receives only its 1/N slice of the reduced output. Callers that need the full tensor on every device must follow with an explicit `all_gather`."

---

# Agent B Review: Chapter 1 — Pass 2

No feedback — chapter approved.

---

# Agent B Review: Chapter 1 — Pass 3

No feedback — chapter approved.
