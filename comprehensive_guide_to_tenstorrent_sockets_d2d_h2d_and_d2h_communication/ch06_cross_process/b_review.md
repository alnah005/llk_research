# Agent B -- Technical Review: Chapter 6 (Cross-Process Socket Sharing)

## Verdict: PASS WITH MINOR FIXES

---

## Scores

| Criterion                        | Score (1-10) | Notes |
|----------------------------------|:------------:|-------|
| Factual accuracy                 | 8            | One confirmed error in flow-control counter write attribution for H2D; one ambiguous phrase about multiple connectors |
| Completeness                     | 9            | Covers protocol, lifecycle, security, and operations thoroughly; minor gap in fork-safety detail |
| Source alignment                 | 9            | Consistent with header files, tech report, and pipeline_block source; one conflict traced to source ambiguity |
| Internal consistency             | 8            | "One or more connectors" in 6.2.1 tension with single-writer invariant; resolved by Failure 5 but wording is misleading |
| Clarity and structure            | 10           | Decision trees, ASCII diagrams, failure catalogs, diagnostic flowcharts all excellent |
| Production relevance             | 10           | Three-barrier shutdown, barrier-gated export, stale descriptor mitigations are directly deployable |

**Weighted Average: 8.8 / 10**

---

## Issues

### Issue 1 -- MEDIUM: Incorrect H2D bytes_acked write attribution in Section 6.3.5

**Location:** `03_security_and_operational_considerations.md`, line 192

**Text:** "For H2D: both `bytes_sent` and `bytes_acked` reside in device L1. Host writes both via TLB write; device reads directly from L1."

**Problem:** The host writes `bytes_sent` to device L1 via TLB write, but the host does NOT write `bytes_acked`. For H2D, the device writes `bytes_acked` (it is the consumer acknowledging receipt). In HOST_PUSH mode, the device additionally mirrors `bytes_acked` to host-pinned memory via NOC write so the host can poll without PCIe reads. The source tech report confirms: "Device polls for new pages, acks via NOC write of bytes_acked to pinned host memory." The H2D flatbuffer descriptor in Section 6.1.4 itself correctly lists `pinned_host_addr` as "Ack mirror (HOST_PUSH)" -- which only makes sense if the device is the writer of bytes_acked.

**Impact:** A developer debugging H2D flow-control stalls could incorrectly look for host-side bugs when the device kernel is the entity responsible for updating bytes_acked.

**Source evidence:** tech_report line 107 (HOST_PUSH: "Device polls for new pages, acks via NOC write of bytes_acked to pinned host memory"); h2d_socket.hpp: `bytes_acked_ptr_` and `pinned_memory_` members suggest device writes ack to host-pinned memory.

---

### Issue 2 -- LOW: "One or more connectors" vs single-writer invariant

**Location:** `02_owner_connector_lifecycle.md`, line 1

**Text:** "Every cross-process socket has exactly one owner and one or more connectors."

**Problem:** Invariant 3 (line 60) states "At most one active writer (or reader) per socket." Failure 5 (line 127) explicitly documents multiple connectors as a failure mode. Saying "one or more connectors" in the opening sentence implies multi-connector is a supported topology, when in practice it requires careful coordination and is flagged as a hazard. The sentence is technically defensible (the protocol does not prevent it), but it sets a misleading expectation.

**Impact:** Low -- Failure 5 and the invariants correct the impression, but a reader skimming 6.2.1 could design a fan-in architecture that the system does not safely support.

---

### Issue 3 -- LOW: Missing clarification on D2H bytes_acked counter location ambiguity

**Location:** `01_export_connect_protocol.md`, lines 99-101 and `03_security_and_operational_considerations.md`, lines 194-195

**Text (01):** D2H descriptor field `bytes_acked_l1_addr` -- "Device L1 address where host writes `bytes_acked` via TLB write"
**Text (03):** "`bytes_acked` is written by host to device L1 via TLB write (device reads via cache invalidation)"

**Problem:** The source tech report flow-control counter summary (line 111) states "D2H: both in host-pinned memory" and "host writes bytes_acked to host RAM." However, the D2H socket header file contains the field `bytes_acked_l1_addr`, and the main D2H description (source line 109) says "writes bytes_acked back to device L1." There is an internal conflict within the source material. The chapter follows the header file and the primary D2H description, which is the correct choice. However, a brief note acknowledging that the canonical location is device L1 (with the header field name as evidence) would strengthen the claim and preempt confusion from readers who encounter the conflicting summary.

**Impact:** Low -- the chapter's position is correct per the header, but the source ambiguity is worth a footnote.

---

### Issue 4 -- INFORMATIONAL: No mention of descriptor versioning or format evolution

**Location:** `01_export_connect_protocol.md`, Section 6.1.4

**Problem:** The flatbuffer descriptor fields are documented, but there is no mention of schema versioning. If the flatbuffer schema changes between TT-Metalium versions, a connector built against a different version than the owner could deserialize incorrectly. This is a real operational concern for rolling upgrades.

**Impact:** Informational -- the source code does not appear to expose versioning, so the chapter cannot document what does not exist. A note flagging this as a gap in the protocol would be valuable for production readers.

---

## Recommended Fixes

### Fix 1 (Issue 1)

In `03_security_and_operational_considerations.md`, replace:

```
For H2D: both `bytes_sent` and `bytes_acked` reside in device L1. Host writes both via TLB write; device reads directly from L1.
```

With:

```
For H2D: both `bytes_sent` and `bytes_acked` reside in device L1. Host writes `bytes_sent` via TLB write. Device writes `bytes_acked` after consuming pages; in HOST_PUSH mode, device also mirrors `bytes_acked` to host-pinned memory via NOC write so the host can poll locally.
```

### Fix 2 (Issue 2)

In `02_owner_connector_lifecycle.md`, replace the opening sentence:

```
Every cross-process socket has exactly one owner and one or more connectors.
```

With:

```
Every cross-process socket has exactly one owner and typically one connector. Multiple connectors can call `connect()` on the same descriptor, but Invariant 3 (single active writer/reader) means only one should be active at a time -- see Failure 5 for the consequences of violating this.
```

### Fix 3 (Issue 3)

In `01_export_connect_protocol.md`, after the D2H descriptor table, add a brief note:

```
> **Note:** The `bytes_acked_l1_addr` field name in the header confirms that `bytes_acked` resides in device L1 for D2H, not in host-pinned memory. The host writes this value to device L1 via TLB write through the PCIe BAR, and the device reads it via L1 cache invalidation.
```

---

## Critical Verification Points -- Results

| # | Verification Point | Result | Location |
|---|-------------------|--------|----------|
| 1 | D2D does NOT use export_descriptor/connect | PASS | index.md:9, 01:104 |
| 2 | export_descriptor serializes to flatbuffer in /dev/shm | PASS | 01:63-64 |
| 3 | connect() reconstructs handle without MetalContext | PASS | 01:66, index.md:45 |
| 4 | H2D connector uses PCIeCoreWriter | PASS | 01:110-132 |
| 5 | vIOMMU required for all H2D/D2H modes | PASS | 03:56, index.md:36 |
| 6 | D2H bytes_acked: host writes to device L1 via TLB write; device reads via cache invalidation | PASS | 01:101, 03:194-195 |
| 7 | /dev/shm is local per host | PASS | 03:126 |
| 8 | Owner must export before connector connects | PASS | 01:155, 02:56 |

All eight critical verification points pass. Issue 1 is a separate factual error about H2D (not D2H) counter write attribution.

---

## Top 3 Strengths

1. **Failure mode catalog with root cause analysis (Section 6.2.5).** Five failure modes, each with trigger, symptom, root cause, and resolution. The "What breaks" callouts are particularly effective -- they describe the insidious silent-corruption cases that would be hardest to debug in production. This is the kind of documentation that prevents outages.

2. **Three-barrier shutdown pattern (Section 6.2.6).** The barrier-gated lifecycle code in DeepSeek V3 PipelineBlock is a directly copy-pasteable production pattern. The chapter correctly identifies that `timeout_ms` is a safety net, not the primary synchronization mechanism, and provides an alternative file-based pattern for non-SPMD deployments.

3. **Security threat model with practical hardening (Section 6.3.1).** The chapter does not pretend the protocol is secure -- it plainly states "no authentication, authorization, encryption, integrity checking, replay protection, or audit logging." Then it provides layered mitigations (umask, udev rules, container isolation, early descriptor cleanup) with decision trees for single-tenant vs multi-tenant. This is honest, actionable security documentation.
