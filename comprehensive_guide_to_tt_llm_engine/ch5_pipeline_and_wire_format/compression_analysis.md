# Compression Analysis -- Chapter 5: Pipeline and Wire Format

## Summary

All four CRUCIAL items from Pass 1 have been resolved. Config structs now live in one place, the token model table is a cross-reference, two-phase shutdown host-side logic is no longer re-explained in socket_pipeline.md, and the writer/reader dispatch in index.md is shortened with a link.

No CRUCIAL issues found.

---

## CRUCIAL Suggestions

None.

---

## MINOR Suggestions

The following MINOR items from Pass 1 remain present and could still be tightened in a future pass:

- **index.md lines 1-5**: Opening paragraph still contains rhetorical hedging ("The scheduler never knows -- and never needs to know --") and a 40-word lead sentence that could be halved.
- **mock_pipeline.md lines 1-7**: "It exists for one reason" followed by three things; the "faithfully implements the full contract" clause restates what PipelineInterface already guarantees.
- **mock_pipeline.md line 100**: Forward reference to SocketPipeline asymmetry before the reader has encountered that page in navigation order.
- **pipeline_simulator.md lines 62-82**: "No Tick Thread" section spends 20 lines on a design-not-taken rationale that could be condensed to about 8.
- **socket_pipeline.md lines 9-27**: PIMPL explanation includes textbook rationale that most C++ readers already know.
- **wire_format.md lines 182-191**: "Design Rationale" section restates points already made inline earlier in the file.
- **socket_pipeline.md lines 129-136**: DeepSeek Mode Summary table partially overlaps the cross-layout comparison in wire_format.md.
- **mock_pipeline.md lines 60-67**: "Why +1 and +3?" point (2) ("Predictable") adds no value given the offsets are fixed constants.
- **pipeline_interface.md lines 109-114**: Paragraph after the ResultDescriptor table restates what the table rows already say ("DeepSeek only").

---

## Load-Bearing Evidence

- `pipeline_interface.md` line ~143: "See the full struct definitions and `std::visit` dispatch code in the [chapter overview](./index.md#selecting-a-backend-the-pipelineconfig-variant)." -- load-bearing because it confirms config structs are no longer duplicated here; index.md is the single source.
- `pipeline_simulator.md` line ~99: "The simulator uses the [same token model as MockPipeline](./mock_pipeline.md#token-model)" -- load-bearing because it replaces the previously restated table with a cross-reference.
- `socket_pipeline.md` line ~80: "SocketPipeline implements the two-phase shutdown protocol described in [PipelineInterface](./pipeline_interface.md#two-phase-shutdown-protocol). The host-side mechanics (atomic flag ... sentinel return ... thread join) are identical" -- load-bearing because it confirms host-side logic is referenced, not re-explained.
- `index.md` line ~7: "The Writer thread calls `inject()` and the Reader thread calls `read_result()` (details in [PipelineInterface](./pipeline_interface.md))." -- load-bearing because it confirms the shortened form with cross-reference replaces the previous duplication.

---

## VERDICT

**PASS.** Crucial updates: no (none needed -- all four were already resolved).
