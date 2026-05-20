# KV Cache Tiering

[< IS Integration Patterns](is_integration_patterns.md)

> **STATUS: PLANNED / NOT YET IMPLEMENTED**
>
> The KV cache tiering system described in this section is a planned feature that has not been implemented in the current codebase. This document describes the design intent and architectural direction based on the system's structure and the problems it is positioned to solve. No code references are provided because no implementation exists. Implementation details may change significantly.

## Motivation

The current system has a fixed KV cache budget: each user slot has a statically allocated region of device memory sized to `max_seq_len`. This creates two problems:

1. **Memory waste for short conversations.** A user generating 50 tokens occupies the same device memory as a user at position 4000. With $\text{max\_users} = 64$ and a $\text{max\_seq\_len} = 131072$, this represents a substantial fraction of HBM regardless of actual utilization.

2. **Hard context limit.** When a user hits `max_seq_len`, the only option is to discard the entire context and start over. There is no mechanism to selectively evict older KV entries to make room for new ones (see [Context Exhaustion](context_exhaustion.md)).

KV cache tiering addresses both problems by introducing a multi-tier memory hierarchy for KV data, analogous to CPU cache hierarchies.

## Three-Tier Model

```
+-------------------------------------------------------------+
|  Tier 0: Device HBM (Hot)                                    |
|  Active decode positions. Fastest access.                    |
|  Capacity: Limited by device memory.                         |
|  Managed by: Pipeline Manager (PM)                           |
|                                                              |
|      ^ promote          | demote                             |
|      |                  v                                    |
|  Tier 1: Host DRAM (Warm)                                    |
|  Recently used KV entries not currently in active decode.    |
|  Capacity: Host memory (typically 10-100x device memory).    |
|  Managed by: PM with IS policy guidance                      |
|                                                              |
|      ^ promote          | demote                             |
|      |                  v                                    |
|  Tier 2: Disk / Network Storage (Cold)                       |
|  Archived KV entries for dormant sessions.                   |
|  Capacity: Effectively unlimited.                            |
|  Managed by: IS                                              |
+-------------------------------------------------------------+
```

### Tier Characteristics

| Tier | Medium | Latency | Capacity | Owner | Use Case |
|------|--------|---------|----------|-------|----------|
| T0 | Device HBM | ~ns | Small (GB) | PM | Active decode: positions being read by attention kernels |
| T1 | Host DRAM | ~us | Medium (tens of GB) | PM | Paused sessions, recently completed turns, overflow from T0 |
| T2 | Disk/NFS | ~ms | Large (TB) | IS | Long-dormant sessions, checkpoint/restore across restarts |

A user must have its KV in T0 to participate in decode. PREFILL always writes directly to T0. The tiering system manages demotion (T0 -> T1 -> T2) and promotion (T2 -> T1 -> T0) transparently. Direct T0 <-> T2 migration is not supported; all paths go through T1 as an intermediate stage.

## IS/PM Ownership Split

The tiering system preserves the fundamental ownership boundary established in Ch1:

| Aspect | Owner | Responsibility |
|--------|-------|---------------|
| **Policy ("why/when")** | IS | Idle-time thresholds, SLA classes, priority ordering, demotion triggers, admission budgets |
| **Execution ("can/how")** | PM | Tier tracking, capacity accounting, DMA execution, LRU metadata, migration state machine |

The IS never performs DMA or accesses device memory. The PM never decides *which* user to demote -- it only reports capacity and executes migration commands from the IS.

## TierCommand API (Planned)

The envisioned API extends the existing `ISRequest` / `SchedulerResponse` protocol with tier-management commands:

```cpp
// PLANNED -- not implemented
enum class TierOp : uint32_t {
    ENSURE_TIER = 1,  // Promote to target tier (no-op if already there)
    EVICT       = 2,  // Demote to target tier (free T0 capacity)
    PIN         = 3,  // Prevent automatic demotion
    UNPIN       = 4   // Allow automatic demotion
};

struct TierCommand {
    uint32_t  user_id;
    TierOp    op;
    uint8_t   target_tier;       // 0, 1, or 2
    uint8_t   priority_class;    // IS-defined priority (0 = highest)
    uint32_t  deadline_ms;       // Max time to wait for migration
    bool      allow_auto_evict;  // If promoting, allow PM to evict others
};
```

### TierCommandResult

```cpp
enum class PMStatus : uint32_t {
    OK            = 0,
    NO_SPACE      = 1,  // Target tier is full; no evictable candidates
    BUSY          = 2,  // Migration already in flight for this user
    INVALID_USER  = 3,  // user_id not allocated
    INVALID_STATE = 4,  // User in wrong state for this operation
    IO_ERROR      = 5   // Storage backend failure (T2 only)
};

struct TierCommandResult {
    uint32_t  user_id;
    TierOp    op;
    PMStatus  status;
};
```

The PM responds immediately (non-blocking). If promotion requires DMA, the PM returns `OK` and the actual migration completes asynchronously. The IS can issue an `ENSURE_TIER` and then poll readiness before issuing CONTINUE.

## PM Execution Mechanics (Planned)

### Per-User Residency Tracking

```cpp
struct KVResidency {
    uint8_t   tier;                // Current tier: 0, 1, or 2
    uint64_t  bytes_hot;           // Bytes in T0
    uint64_t  bytes_warm;          // Bytes in T1
    uint64_t  bytes_cold;          // Bytes in T2
    uint64_t  last_access_ts;      // Timestamp of last decode activity
    bool      pinned;              // Exempt from automatic demotion
    bool      migration_in_flight; // DMA or I/O currently active
};
```

### Capacity Tracking

```cpp
struct PMCapacity {
    uint64_t hot_capacity_bytes;   // Total T0 capacity
    uint64_t hot_used_bytes;       // Currently occupied T0 bytes
    uint64_t warm_capacity_bytes;  // Total T1 capacity
    uint64_t warm_used_bytes;      // Currently occupied T1 bytes
};
```

The PM exposes capacity snapshots to the IS on request so the IS can make informed demotion/promotion decisions.

### `can_admit_decode`

Before a CONTINUE can proceed, the PM checks whether the user's KV is in T0:

```cpp
bool can_admit_decode(uint32_t uid) {
    return residency[uid].tier == 0 && !residency[uid].migration_in_flight;
}
```

If the KV is not in T0, the IS must issue `ENSURE_TIER(uid, target_tier=0)` and wait for the promotion to complete before issuing CONTINUE.

### DMA and Pipeline Coordination

Migration within the PM would be coordinated with the pipeline's batch execution:

1. **DMA scheduling.** T0 <-> T1 transfers use device DMA engines, scheduled between pipeline steps to avoid contention with attention kernel reads.
2. **Pipeline barrier.** Before a `DEMOTE_TO_T1`, the PM must ensure no pipeline step is currently reading from the slot's KV region.
3. **Atomic visibility.** After a `PROMOTE_TO_T0`, the slot's KV must be fully resident before the Writer includes that user in a batch.
4. **Partial residency.** For `EVICT_RANGE`, the PM would need to track which position ranges are resident in T0 for each slot.

## Migration Paths

### Scenario 1: User Pauses Mid-Conversation

```
Active decode (T0) --EVICT(target=1)--> Warm storage (T1)
                                             |
                    (user returns)           |
                                             |
Active decode (T0) <--ENSURE_TIER(0)---- (T1)
```

When a user completes a turn but does not immediately send a follow-up, the IS may decide to demote that user's KV to T1 to free device memory for other users. When the user returns, an `ENSURE_TIER(uid, 0)` brings the KV back before the `CONTINUE`.

### Scenario 2: Session Goes Dormant

```
Warm storage (T1) --EVICT(target=2)--> Cold storage (T2)
                                             |
                    (user returns            |
                     hours later)            |
                                             |
Warm storage (T1) <--ENSURE_TIER(1)----- (T2)
    |
    +--ENSURE_TIER(0)--> Active decode (T0)
```

Restoration requires two hops: T2 -> T1 -> T0.

### Scenario 3: Context Window Extension via Partial Eviction

```
T0 KV cache: [pos 0] [pos 1] ... [pos 4095]  (full)
                |
    EVICT_RANGE(0, 2048)
                |
                v
T0 KV cache: [pos 2048] ... [pos 4095] [    free    ]
T1 backup:   [pos 0] ... [pos 2047]
```

By evicting older positions, the system frees device memory for new tokens. This is the most complex migration path and would require attention kernel support for non-contiguous KV.

## Eviction Strategy

The IS would implement the eviction policy, using signals such as:

| Signal | Weight | Rationale |
|--------|--------|-----------|
| Time since last activity | High | Idle users are unlikely to return soon |
| Current position / context usage | Medium | Users with more context are more expensive to restore |
| Client priority / SLA tier | Variable | Premium users may be kept hot longer |
| Device memory pressure | Trigger | Eviction is triggered when T0 utilization exceeds a threshold |

The IS makes the final eviction decision, not the PM. The PM only provides sorted candidate lists.

### PM-Side Candidate Selection

```cpp
// PM-side: select eviction candidates sorted by LRU
std::vector<uint32_t> select_eviction_candidates(uint32_t count) {
    std::vector<std::pair<uint64_t, uint32_t>> candidates;
    for (uint32_t uid = 0; uid < max_users; uid++) {
        if (residency[uid].tier == 0
            && !residency[uid].pinned
            && !residency[uid].migration_in_flight
            && state[uid] != PREFILL
            && state[uid] != DECODE) {
            candidates.push_back({residency[uid].last_access_ts, uid});
        }
    }
    std::sort(candidates.begin(), candidates.end());  // Oldest first
    std::vector<uint32_t> result;
    for (size_t i = 0; i < std::min((size_t)count, candidates.size()); i++) {
        result.push_back(candidates[i].second);
    }
    return result;
}
```

Eviction candidates are filtered to exclude pinned users, users with migration in flight, and active users (in PREFILL or DECODE state -- their KV is being read/written by the pipeline).

## Admission Failure

If the PM cannot admit a user to T0 (no capacity, no evictable candidates), it returns `NO_SPACE` immediately. The PM never blocks waiting for capacity. The IS decides what to do:

- **Wait and retry** (with exponential backoff)
- **Evict a lower-priority session** (issue EVICT, then retry ENSURE_TIER)
- **Reject the request** (return 503 to the client)
- **Queue internally** (IS-side waiting queue with SLA ordering)

This mirrors the PM's general philosophy: the PM provides mechanisms, the IS provides policy.

## IS-Side Tier Controller

The IS runs a periodic control loop (100--1000ms cadence) that manages tier policy:

```cpp
// IS-side: periodic tier management (NOT on the token hot path)
void tier_controller_tick() {
    auto now = current_time_ms();

    for (auto& [session, slot] : session_to_slot) {
        auto& info = session_info[session];

        // Demotion: idle sessions in T0 -> T1
        if (info.state == COMPLETE
            && info.tier == 0
            && (now - info.last_activity) > idle_threshold_ms) {
            push_tier_command(TierCommand{
                .user_id     = slot,
                .op          = TierOp::EVICT,
                .target_tier = 1
            });
        }

        // Spill: long-idle sessions in T1 -> T2
        if (info.tier == 1
            && (now - info.last_activity) > spill_threshold_ms) {
            push_tier_command(TierCommand{
                .user_id     = slot,
                .op          = TierOp::EVICT,
                .target_tier = 2
            });
        }
    }
}
```

This runs at IS cadence (milliseconds), not pipeline cadence (microseconds). Tier management is a control-plane operation that does not interfere with the hot path.

## Interaction with the Session Lifecycle

| Lifecycle Event | Tier Implication |
|----------------|-----------------|
| ALLOCATE | KV slot reserved in T0 (no data yet) |
| SUBMIT | Prefill writes KV to T0 |
| DECODE | KV read/written in T0 every tick |
| COMPLETE | KV idle in T0; eligible for demotion |
| CONTINUE | KV must be in T0; promotion if demoted |
| CANCEL | KV freed from all tiers |

> **Key constraint:** CONTINUE on a demoted user requires explicit promotion. The IS must issue `ENSURE_TIER(uid, 0)`, wait for `OK`, and only then issue CONTINUE. If the IS issues CONTINUE while KV is in T1 or T2, the PM will reject it with `INVALID_STATE`.

## Current State and Path Forward

As of the current codebase:
- **No tiering code exists.** KV cache is statically allocated per slot at initialization.
- **`pipeline_->reset_kv(uid)` is the only KV management operation.** It clears the entire slot; there is no partial eviction or migration.
- **The IS/PM ownership split is already established** (see Ch1), providing a natural integration point for tier commands.
- **The `CONTINUE` mechanism already preserves KV across turns**, which is a prerequisite for tiering (without it, there would be nothing to tier).

The existing architecture is well-positioned for tiering: the slot-based allocation, the generation counter for staleness detection, and the deferred cancellation protocol all provide foundations that a tiering system would build on.

---

*[< IS Integration Patterns](is_integration_patterns.md) | [Back to Chapter Overview](index.md)*
