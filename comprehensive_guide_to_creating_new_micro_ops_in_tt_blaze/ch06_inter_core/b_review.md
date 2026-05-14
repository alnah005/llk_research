# Agent B Review -- Chapter 6: Inter-Core and Inter-Device Operations

Reviewer: Agent B (independent critic)
Date: 2026-05-13

Files reviewed:
- ch06_inter_core/index.md
- ch06_inter_core/01_sender_receiver_pattern.md
- ch06_inter_core/02_mcast_gather_scatter.md
- ch06_inter_core/03_ccl_and_fabric.md

Source files checked:
- blaze/ops/mcast/op.py (129 lines)
- blaze/ops/mcast/kernels/op.hpp (326 lines)
- blaze/ops/gather/op.py (183 lines)
- blaze/ops/gather/kernels/op.hpp (135 lines)
- blaze/ops/scatter/op.py (161 lines)
- blaze/ops/scatter/kernels/op.hpp (125 lines)
- blaze/ops/scatter_raw/op.py (168 lines)
- blaze/ops/copy/op.py (148 lines)
- blaze/ops/copy/kernels/op.hpp (163 lines)
- blaze/ops/ccl_broadcast/op.py (252 lines)
- blaze/ops/ccl_broadcast/kernels/op.hpp (302 lines)
- blaze/ops/all_reduce/op.py (293 lines)
- blaze/ops/all_reduce/kernels/op.hpp (258 lines)
- blaze/ops/barrier_sender/op.py (115 lines)
- blaze/ops/barrier_sender/kernels/op.hpp (121 lines)
- blaze/ops/barrier_receiver/op.py (93 lines)
- blaze/ops/barrier_receiver/kernels/op.hpp (84 lines)
- blaze/ops/dm_risc_handshake/op.py (66 lines)
- blaze/ops/embed_mcast/op.py (57 lines)
- blaze/ccl.py (73 lines)
- blaze/blaze_op.py (614 lines)
- blaze/fused_program.py (lines 418-453, 560-610, 1590-1625)

---

## Summary

Chapter 6 is a substantial (3,385-line) treatment of inter-core and inter-device operations in TT-Blaze. It covers four intra-device primitives (Mcast, Gather, Scatter, Copy), two CCL ops (CclBroadcast, AllReduce), barrier primitives, and the shared `setup_fabric()` utility. The chapter is thorough and well-structured.

Overall quality is high. The Python code snippets match the source almost verbatim, the C++ kernel excerpts are accurate, and the architectural explanations are sound. I found 2 minor inaccuracies, 2 omissions, and 2 cosmetic issues. No critical errors.

---

## Issues

### Issue 1 (Minor inaccuracy) -- CclBroadcast routing example shows incorrect receiver hop counts

Section 3.4.1 (03_ccl_and_fabric.md, lines 329-335) provides an example:

```
Row 0: is_receiver=True,  num_fwd=0, num_bwd=0
Row 1: is_sender=True,    num_fwd=2, num_bwd=1
Row 2: is_receiver=True,  num_fwd=0, num_bwd=0
Row 3: is_receiver=True,  num_fwd=0, num_bwd=0
```

This is incorrect for receivers. The `compute_routing()` function at `blaze/ops/ccl_broadcast/op.py` lines 62-64 computes `num_fwd = mesh_rows - row - 1` and `num_bwd = row` for **all devices**, not just senders. A receiver at row 0 gets `num_fwd=3, num_bwd=0`; a receiver at row 2 gets `num_fwd=1, num_bwd=2`; a receiver at row 3 gets `num_fwd=0, num_bwd=3`. While receivers do not open fabric connections (because `num_connections=0` for non-senders at line 67-69), the `BroadcastRouting` dataclass stores these non-zero values. The values are also passed to the kernel as CT args (`num_targets_forward_direction`, `num_targets_backward_direction`) even for receivers.

Correct example:
```
Row 0: is_receiver=True,  num_fwd=3, num_bwd=0, num_connections=0
Row 1: is_sender=True,    num_fwd=2, num_bwd=1, num_connections=2
Row 2: is_receiver=True,  num_fwd=1, num_bwd=2, num_connections=0
Row 3: is_receiver=True,  num_fwd=0, num_bwd=3, num_connections=0
```

### Issue 2 (Minor inaccuracy) -- BarrierSender compose() has typo in source but chapter shows clean version

Section 3.6.1 (03_ccl_and_fabric.md, line 1079) shows the BarrierSender emit signature with `send_to_brisc: bool = False`. The actual source at `blaze/ops/barrier_sender/op.py` line 28 in `compose()` has a typo: `send_to_brsic=user_args.get("send_to_brsic", False)` (misspelling "brisc" as "brsic"). The chapter does not call this out. The `emit()` method itself (line 43) has the correct spelling `send_to_brisc`. This is a real bug in the source code that the chapter silently papers over. While the chapter is not wrong per se (it accurately shows `emit()`'s signature), it misses an opportunity to flag a real source-code defect.

### Issue 3 (Omission) -- AllReduce emit() calls input.buffer_address() which would fail on CBHandle

Section 3.5.2 (03_ccl_and_fabric.md, line 795) shows `(f"{prefix}.sender_input_tensor_address", input.buffer_address())` in the NCRISC CT args. The AllReduce emit signature accepts `input: ttnn.Tensor | CBHandle`, but the code at `blaze/ops/all_reduce/op.py` line 223 calls `input.buffer_address()` unconditionally. If `input` is a `CBHandle`, this would call `CBHandle.buffer_address()` which may or may not exist/work the same way as `ttnn.Tensor.buffer_address()`. The chapter does not discuss this potential mismatch. Similarly, line 253 calls `intermediate.buffer_address()` which could be a CBHandle. The chapter should note whether `buffer_address()` is safe to call on CBHandles in this context.

### Issue 4 (Omission) -- Chapter does not mention the `noc_async_writes_flushed()` vs `noc_async_posted_writes_flushed()` distinction

The CclBroadcast kernel uses `noc_async_writes_flushed()` (line 241, 271 of the HPP), while Mcast and Gather use `noc_async_posted_writes_flushed()`. These are different APIs -- `noc_async_posted_writes_flushed()` is specifically for posted writes, while `noc_async_writes_flushed()` handles both posted and non-posted. The chapter does not explain this distinction despite covering both types of ops. The `posted` flag discussion in Section 2.1.3 mentions `noc_async_posted_writes_flushed()` as the required flush for posted writes but does not compare it to the broader `noc_async_writes_flushed()` used in fabric ops.

### Issue 5 (Cosmetic) -- Section 1 describes `emit()` as `@staticmethod` but references `FusedProgram.__init__()` code that is not emit

Section 1.4 (01_sender_receiver_pattern.md, lines 358-390) shows code attributed to `FusedProgram.__init__()` for grid precomputation. This is correct code from the source (`fused_program.py` lines 424-453), but the code block is not labeled as coming from `__init__()`. It appears under "Sender Grid vs. Receiver Grid" without attribution, though the text does say "The FusedProgram precomputes several grid definitions."

### Issue 6 (Cosmetic) -- Inconsistent terminology: `init_src` vs `skip_src_init`

Section 1.4 (01_sender_receiver_pattern.md, line 352) says Scatter uses `skip_src_init`. Section 2.3.1 (02_mcast_gather_scatter.md, lines 690-695) then shows code for this flag. However, the actual Scatter source (scatter/op.py line 112) has a more complex conditional: `f.flag(f"{prefix}.skip_src_init", skip_src) if src_is_cb else (f"{prefix}.skip_src_init", {dest_core_ranges: 0})`. The chapter's snippet at line 693 is a simplified paraphrase that elides the `else` branch. This is not incorrect, but the simplification may confuse a reader trying to match the code to the source.

---

## Verified correct (spot-checked claims)

1. **SemProtocol enum** (01, line 144-150): Matches `blaze/blaze_op.py` lines 96-103 exactly. Four values: SENDER, RECEIVER, NOC0_RECEIVER, NOC1_RECEIVER.

2. **Mcast.emit() signature** (02, lines 30-41): Matches `blaze/ops/mcast/op.py` lines 26-34. Parameters, types, and defaults all correct.

3. **Mcast source resolution** (02, lines 59-63): Matches `mcast/op.py` lines 44-47. `isinstance(src, CBHandle)` check is accurate.

4. **Mcast data size computation** (02, lines 69-71): Matches `mcast/op.py` lines 49-51. `num_pages * tile_size = data_size_bytes`.

5. **Mcast dst CB allocation with `balanced=False`** (02, lines 79-88): Matches `mcast/op.py` lines 63-75. Comment about conservative balanced=False is even present in source.

6. **Mcast semaphore allocation** (02, lines 97-100): Matches `mcast/op.py` lines 77-79. Conditional sender_sem allocation and receiver_sem allocation.

7. **Mcast role flags** (02, lines 108-114): Matches `mcast/op.py` lines 82-87. Four flags: is_sender, is_receiver, pop_src, init_src.

8. **Mcast NCRISC CT args** (02, lines 126-133): Matches `mcast/op.py` lines 91-100. Six args including src, src_num_pages, src_is_tensor_backed, receiver_semaphore, dst, dst_num_pages.

9. **Mcast BRISC CT args** (02, lines 139-156): Matches `mcast/op.py` lines 102-123. All 14 args including linked=1, posted=1, loopback=0 defaults.

10. **Mcast C++ CoreCTArgs** (02, lines 174-179): Matches `mcast/kernels/op.hpp` lines 174-179. Four Flag fields.

11. **Mcast C++ ReaderCTArgs** (02, lines 182-189): Matches `mcast/kernels/op.hpp` lines 182-189. Six fields.

12. **Mcast C++ WriterCTArgs** (02, lines 192-209): Matches `mcast/kernels/op.hpp` lines 192-209. Fifteen fields including linked, posted, loopback.

13. **Mcast init() BRISC path** (02, lines 220-238): Matches `mcast/kernels/op.hpp` lines 220-239. init_persistent_sender call, INVALID/VALID semaphore cycle.

14. **Mcast operator() BRISC sender path** (02, lines 250-258): Matches `mcast/kernels/op.hpp` lines 243-258. cb_wait_front, send_data_mcast, send_with_state, flush, pop pattern.

15. **Mcast operator() NCRISC receiver path** (02, lines 267-277): Matches `mcast/kernels/op.hpp` lines 259-268. cb_reserve_back, semaphore_wait VALID, set INVALID, cb_push_back.

16. **Mcast run_as<A>() template** (02, lines 287-307): Matches `mcast/kernels/op.hpp` lines 271-301. Uses dm1_cta for persistent state, w/r for new args.

17. **Mcast teardown** (02, lines 345-355): Matches `mcast/kernels/op.hpp` lines 313-321. teardown_persistent_sender with sender_semaphore redirection.

18. **Mcast send_data_mcast chunking** (01, lines 430-443): Matches `mcast/kernels/op.hpp` lines 128-140. Template recursion with NOC_MAX_BURST_SIZE.

19. **Mcast get_noc_multicast_addr NOC1 swap** (01, lines 111-126): Matches `mcast/kernels/op.hpp` lines 28-40. NOC0 vs NOC1 coordinate swap.

20. **Gather.emit() signature** (02, lines 389-403): Matches `gather/op.py` lines 38-51. All parameters correct.

21. **Gather receiver core selection** (02, lines 419-429): Matches `gather/op.py` lines 58-65. recv_grid/recv_noc/recv_logical resolution.

22. **Gather dual-NOC semaphores** (02, lines 469-471): Matches `gather/op.py` lines 121-122. noc0_sem and noc1_sem allocation.

23. **Gather per-core sender_idx** (02, lines 482-486): Matches `gather/op.py` lines 131-138. Dict comprehension for sender_idx per core.

24. **Gather C++ sender path** (02, lines 522-556): Matches `gather/kernels/op.hpp` lines 74-108. core_index computation, offset, noc_async_write_one_packet, semaphore_inc.

25. **Gather C++ receiver path** (02, lines 568-585): Matches `gather/kernels/op.hpp` lines 109-127. Counting semaphore wait, noc1 conditional.

26. **Scatter.emit() signature** (02, lines 617-627): Matches `scatter/op.py` lines 46-54. Parameters correct.

27. **Scatter C++ sender loop** (02, lines 703-724): Matches `scatter/kernels/op.hpp` lines 73-102. Loop over dests, rt_args::get, noc_async_write, semaphore_inc.

28. **Scatter C++ receiver** (02, lines 733-748): Matches `scatter/kernels/op.hpp` lines 104-117. semaphore_wait, set 0, setup_sharded_buffer.

29. **Copy.emit() signature** (02, lines 780-793): Matches `copy/op.py` lines 60-74. All parameters correct.

30. **Copy C++ RISC selection** (02, lines 881-887): Matches `copy/kernels/op.hpp` lines 150-156. use_ncrisc guards.

31. **Copy unified_ct_args** (02, lines 833-849): Matches `copy/op.py` lines 129-145. All 15 unified args correct.

32. **CclBroadcast BroadcastRouting dataclass** (03, lines 286-296): Matches `ccl_broadcast/op.py` lines 26-38. Nine fields.

33. **CclBroadcast compute_routing() role determination** (03, lines 300-312): Matches `ccl_broadcast/op.py` lines 49-57. is_sender and is_secondary_sender conditions.

34. **CclBroadcast torus hop calculation** (03, lines 319-324): Matches `ccl_broadcast/op.py` lines 59-61. (mesh_rows-1)//2 forward, remainder backward.

35. **setup_fabric() signature** (03, lines 137-145): Matches `blaze/ccl.py` lines 18-25. All parameters correct.

36. **setup_fabric() empty destination guard** (03, lines 169-172): Matches `blaze/ccl.py` lines 40-42. Returns early with empty RT args slot.

37. **setup_fabric() semaphore allocation** (03, lines 180-189): Matches `blaze/ccl.py` lines 49-58. teardown_sems and buf_idx_sems with program_semaphore=True.

38. **setup_fabric() fabric core tracking** (03, lines 246-247): Matches `blaze/ccl.py` line 72. Exact dict nesting: prefix -> risc -> [cores].

39. **setup_fabric() return value** (03, lines 252-253): Matches `blaze/ccl.py` line 73. Dict lookup by risc.

40. **AllReduce two-core architecture** (03, lines 698-704): Matches `all_reduce/op.py` lines 134-143. data_core, sender_core adjacency logic.

41. **AllReduce semaphore swap** (03, lines 766-769): Matches `all_reduce/op.py` lines 155-156. First chip: send on sem0, receive on sem1. Reversed for second.

42. **AllReduce fabric on BRISC** (03, lines 837-844): Matches `all_reduce/op.py` lines 270-277. Risc.BRISC, link_indices=[0 if is_first_chip else 1].

43. **AllReduce per_core_unified_ct_args for fabric_arg_idx** (03, lines 854-858): Matches `all_reduce/op.py` lines 279-281. Uses unified, not brisc.

44. **AllReduce C++ TRISC static_assert** (03, line 968): Matches `all_reduce/kernels/op.hpp` line 170. `static_assert(receiver_num_tiles <= 8)`.

45. **AllReduce BRISC receiver barrier signal** (03, lines 1013-1021): Matches `all_reduce/kernels/op.hpp` lines 241-249. BARRIER_NOC=0, noc_semaphore_inc, noc_async_atomic_barrier.

46. **BarrierSender.emit() signature** (03, lines 1067-1079): Matches `barrier_sender/op.py` lines 35-47. All parameters correct.

47. **BarrierSender emit_injected_fabric_barrier_sender** (03, lines 1179-1195): Matches `barrier_sender/op.py` lines 97-115. Set comprehension for sender_cores, bool() for execute flags.

48. **BarrierReceiver.emit() signature** (03, lines 1209-1219): Matches `barrier_receiver/op.py` lines 33-43. All parameters correct.

49. **BarrierReceiver C++ receiver_impl** (03, lines 1239-1245): Matches `barrier_receiver/kernels/op.hpp` lines 53-57. wait + set(0).

50. **DMRiscHandshake.emit() structure** (03, lines 1260-1278): Matches `dm_risc_handshake/op.py` lines 29-53. semaphore, is_active_core flag, execute_signalling_logic_on_ncrisc.

51. **EmbedMcast composition** (02, lines 920-933): Matches `embed_mcast/op.py` lines 39-56. Embedding.emit -> Mcast.emit with child_prefix.

52. **FusedProgram.flag() method** (01, lines 300-309): Matches `fused_program.py` lines 1595-1602. Returns (name, {cores: int(enabled)}).

53. **FusedProgram grid precomputation** (01, lines 360-378): Matches `fused_program.py` lines 424-453. sender, sender_grid, mcast_range, all_cores, num_mcast_cores, mcast_receiver_grid, noc_sender, noc_start, noc_end all verified.

54. **CclBroadcast three semaphores** (03, lines 405-407): Matches `ccl_broadcast/op.py` lines 181-183. out_sem, bar_sem, sec_sem.

55. **CclBroadcast fabric on NCRISC** (03, lines 477-479): Matches `ccl_broadcast/op.py` line 237. `risc=Risc.NCRISC`.

---

## Verdict

**PASS with minor corrections needed.**

The chapter is an excellent, thorough treatment of TT-Blaze's inter-core and inter-device operations. The 55 spot-checked claims across all four files overwhelmingly match the source code. The Python code snippets are near-verbatim reproductions of the actual source, and the C++ kernel excerpts faithfully represent the real implementations.

The two minor inaccuracies are:
1. The CclBroadcast routing example shows receivers with zeroed hop counts, when in reality receivers have non-zero hop counts in the BroadcastRouting dataclass (they just don't open connections). This is a factual error that should be corrected.
2. A compose() typo in the BarrierSender source (`send_to_brsic`) is silently corrected in the chapter's presentation without mention.

The two omissions are documentation gaps rather than errors: the chapter could note the AllReduce CBHandle/buffer_address() interaction and the `noc_async_writes_flushed()` vs `noc_async_posted_writes_flushed()` API distinction.

The code quality, architectural explanations, and pedagogical structure are strong. The summary tables, composition guidelines, and common pitfalls sections add practical value beyond bare API documentation.
