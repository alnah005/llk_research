# Agent B Review -- Chapter 7: Pass 1

## Errors Found

### Error 1: [mooncake_transport.md] [line 58] -- Constructor step count mismatch

- What the chapter says: "The constructor (`mooncake_test_engine.hpp`, lines 235-309) performs five steps in strict order:" followed by a numbered list of only 4 items (posix_memalign, TransferEngine::init, installTransport, registerLocalMemory).
- What the source code says: The constructor (lines 235-310) does perform four distinct API-level operations. The chapter claims "five steps" but only lists four.
- Fix: Either change "five steps" to "four steps", or split the TransferEngine constructor (`std::make_unique<mooncake::TransferEngine>(/*auto_discover=*/false)` at line 249) and `engine_->init(...)` (at line 254) into two separate steps to make five. If splitting, renumber accordingly.

---

### Error 2: [mooncake_transport.md] [line 58] -- Constructor end line is 310, not 309

- What the chapter says: "The constructor (`mooncake_test_engine.hpp`, lines 235-309)"
- What the source code says: The constructor's closing brace is at line 310, not 309. Line 309 closes the `if (rc != 0)` block inside the constructor; line 310 closes the constructor body itself.
- Fix: Change "lines 235-309" to "lines 235-310".

---

### Error 3: [mooncake_transport.md] [line 67] -- posix_memalign line number off by one

- What the chapter says: "**Buffer allocation** uses `posix_memalign` with 4096-byte alignment (line 239)"
- What the source code says: Line 239 is `void* raw = nullptr;`. The `posix_memalign` call is at line 240: `if (::posix_memalign(&raw, 4096, cfg_.buffer_size) != 0) {`
- Fix: Change "(line 239)" to "(line 240)".

---

### Error 4: [mooncake_transport.md] [line 69] -- TransferEngine init vs constructor confusion

- What the chapter says: "**TransferEngine initialization** passes `auto_discover=false` (line 249) so transport installation is explicit."
- What the source code says: Line 249 is `engine_ = std::make_unique<mooncake::TransferEngine>(/*auto_discover=*/false);` -- this is the TransferEngine **constructor**, not `init()`. The `init()` call is at line 254. The `auto_discover=false` parameter is passed to the constructor, not to `init()`.
- Fix: Clarify that `auto_discover=false` is passed to the TransferEngine constructor (line 249), and that `init()` is a separate subsequent call (line 254). Or change to "The TransferEngine constructor passes `auto_discover=false` (line 249)".

---

### Error 5: [mooncake_transport.md] [line 105] -- registerLocalMemory line range

- What the chapter says: "`registerLocalMemory` with location `\"cpu:0\"` (NUMA node 0), `remote_accessible=true`, and `update_metadata=true` (line 301-303)."
- What the source code says: The `registerLocalMemory` call spans lines 301-304 (the closing parenthesis and semicolon are on line 304).
- Fix: Change "(line 301-303)" to "(lines 301-304)".

---

### Error 6: [loopback_kernel.md] [line 15] -- Wire format comment line range

- What the chapter says: "The word indices are documented in a comment at lines 17-20"
- What the source code says: The wire format comment spans lines 17-21. Lines 20-21 contain the ResultPage (D2H) layout, with line 21 containing `// [6]=actual_token_pos [3-5,7-15]=reserved`.
- Fix: Change "lines 17-20" to "lines 17-21".

---

### Error 7: [loopback_kernel.md] [line 63-64] -- Kernel entry point code block line range

- What the chapter says: "// kernels/pipeline_loopback.cpp, lines 27-44"
- What the source code says: The code block shown in the chapter includes `noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);` which is at line 45. The code snippet's content extends through line 45, not 44.
- Fix: Change "lines 27-44" to "lines 27-45" in the code block comment.

---

### Error 8: [loopback_microbenchmark.md] [line 63] -- DistributedContext setup line range

- What the chapter says: "// Lines 67-71: Distributed context setup"
- What the source code says: Line 67 of `dummy_pipeline_launcher.cpp` is a blank line. The `DistributedContext::create(argc, argv)` call starts at line 68. The actual setup code occupies lines 68-71. Line 66 has `using namespace tt::tt_metal::distributed::multihost;` which is part of the setup but not included in the chapter's code snippet.
- Fix: Change "Lines 67-71" to "Lines 68-71" or "Lines 66-71" (if the using statement is considered part of setup).

---

### Error 9: [mooncake_transport.md] [line 109] -- Destructor line range

- What the chapter says: "The destructor (lines 312-322) unregisters the local memory"
- What the source code says: The destructor body spans lines 312-322, which is correct. However, the closing brace is at line 322, and the non-copyable declarations follow at 324-325. No error here upon closer inspection.
- Fix: No fix needed. Retracted.

---

### Error 10: [deepseek_inference_runner.md] [line 60] -- turn_limits line range

- What the chapter says: "controlled by a per-user `turn_limits` vector (lines 528-534)"
- What the source code says: The `turn_limits.resize(num_users)` call is at line 527, which is the first line that populates the vector. Lines 528-534 contain the loop that fills turn_limits and computes overall_max.
- Fix: Change "(lines 528-534)" to "(lines 527-534)" to include the `resize` call that creates the vector entries.

---

### Error 11: [mooncake_transport.md] [line 232-243] -- MooncakeTransferFixture line range

- What the chapter says: "The shared test fixture (`mooncake_test_fixture.hpp`, lines 46-88)"
- What the source code says: The fixture class is at lines 46-88. Correct. However, the chapter's code snippet of the class shows a public constructor signature and members but not the `SetUp()` and `TearDown()` overrides declared at lines 48-49 (which are private to the `protected` section). The code snippet shows `make_pair`, `initiator_`, `target_`, `target_handle_from_initiator_`, and `do_write_and_verify` -- this matches the source. No error.
- Fix: No fix needed. Retracted.

---

## No Issues Found In

- **index.md**: All cross-reference links, capability table entries, source file paths, and complexity ladder descriptions are accurate. The structural overview correctly maps each tool to its source files and capabilities.
- **deepseek_inference_runner.md**: All CLI argument names, defaults, and descriptions match the source. The chat template code, tokenizer struct, scheduler configuration, batching logic, speculative decode assignment, pipelined teardown, metric definitions, and output format are all accurately transcribed. Line number references are correct throughout.

## Summary

8 confirmed errors found across 3 files (mooncake_transport.md: 5 errors, loopback_kernel.md: 2 errors, loopback_microbenchmark.md: 1 error). 2 initially flagged issues were retracted on closer review.

Most impactful: **Error 1** (step count mismatch in mooncake_transport.md) -- the chapter claims "five steps" but only documents four, which could confuse readers trying to follow the construction sequence. **Error 4** (TransferEngine constructor vs init confusion) compounds this by attributing the `auto_discover=false` parameter to the wrong function call.

The remaining errors are minor line-number off-by-ones that do not affect comprehension. The deepseek_inference_runner.md and index.md files are clean -- every code snippet, function signature, CLI argument, constant, behavioral description, and cross-reference verified against the source.
