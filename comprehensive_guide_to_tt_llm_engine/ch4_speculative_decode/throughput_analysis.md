# Throughput Analysis

Speculative decode trades pipeline bandwidth (2 slots per user per round) for the possibility of higher per-user throughput (2 output tokens per round on match). This section quantifies the tradeoff for individual users and mixed-population systems, derives the break-even analysis, documents the diagnostic counters, and connects the formulas back to the worked example from [`result_classification.md`](./result_classification.md).

## Regular vs Speculative: Per-User Comparison

Let $D$ denote the pipeline depth (number of stages), $\tau$ the tick period, and $\alpha$ the accept rate (fraction of BASE results that match the prediction).

### Regular Decode

Each user consumes 1 pipeline slot per round. Each round takes $D$ ticks. One output token is produced per round.

$$\text{TPOT}_{\text{regular}} = D \cdot \tau$$

### Speculative Decode

Each user consumes 2 pipeline slots per round (BASE + SPEC). The output depends on whether the prediction is accepted:

- **Match** (probability $\alpha$): 2 output tokens.
- **Mismatch** (probability $1 - \alpha$): 1 output token.

$$\text{Expected tokens per round} = 2\alpha + 1(1 - \alpha) = 1 + \alpha$$

$$\text{TPOT}_{\text{spec}} = \frac{D \cdot \tau}{1 + \alpha}$$

At $\alpha = 1.0$: $\text{TPOT} = \frac{D \cdot \tau}{2}$ (2x improvement).

At $\alpha = 0.5$: $\text{TPOT} = \frac{D \cdot \tau}{1.5}$ (1.5x improvement, but at 2x slot cost).

At $\alpha = 0.0$: $\text{TPOT} = D \cdot \tau$ (no improvement, 2x slot cost).

## Comparison Table

For $D = 64$, $\tau = 44\,\mu\text{s}$:

| Metric | Regular | Spec ($\alpha = 0.8$) | Spec ($\alpha = 1.0$) |
|--------|---------|----------------------|----------------------|
| Slots per user per round | 1 | 2 | 2 |
| Output tokens per round | 1 | 1.8 | 2 |
| TPOT | 2.816 ms | 1.564 ms | 1.408 ms |
| Per-user throughput | 355 tok/s | 639 tok/s | 710 tok/s |

| Accept Rate ($\alpha$) | Speedup | Effective TPOT ($D=64$, $\tau=44\mu s$) |
|---|---|---|
| 100% | 2.0x | 1.408 ms |
| 90% | 1.9x | 1.482 ms |
| 80% | 1.8x | 1.564 ms |
| 50% | 1.5x | 1.877 ms |
| 20% | 1.2x | 2.347 ms |
| 0% | 1.0x | 2.816 ms |

Even at 0% accept rate, per-user TPOT does not degrade below regular decode -- the user still gets 1 token per round. The cost is entirely in pipeline bandwidth.

## System-Level Analysis

With $D$ pipeline slots total, $M$ spec-decode users, and $R$ regular users, the slot constraint is:

$$2M + R \leq D$$

System-wide token output per round:

$$T_{\text{system}} = R + M(1 + \alpha)$$

For a fully loaded pipeline ($R + 2M = D$), substituting $R = D - 2M$:

$$T_{\text{system}} = D - 2M + M(1 + \alpha) = D + M(\alpha - 1)$$

This yields a critical insight:

- **If $\alpha = 1$**: $T_{\text{system}} = D$. System throughput equals the all-regular case.
- **If $\alpha < 1$**: $T_{\text{system}} = D - M(1 - \alpha) < D$. Each speculative user that mismatches wastes a pipeline slot.

### Break-Even Analysis

A speculative user is individually "worth it" (produces more tokens than the two regular users displaced) when:

$$1 + \alpha > 2 \cdot 1 \implies \alpha > 1$$

This is never achievable -- individual speculative throughput can at most equal two regular users' combined throughput, when $\alpha = 1$. Therefore, **speculative decode at maximum pipeline utilization never improves aggregate system throughput**. It can only improve per-user throughput at the cost of serving fewer users.

The value of speculative decode is not in aggregate throughput but in **individual latency**: a speculative user with $\alpha = 0.8$ sees $\text{TPOT} = \frac{D \cdot \tau}{1.8} \approx 0.56 \cdot D \cdot \tau$, a 44% reduction in per-token latency. This benefits interactive use cases where per-user response speed matters more than aggregate token throughput.

## Accept Rate Threshold

The per-user slot efficiency of speculative decode is:

$$\eta_{\text{spec}} = \frac{1 + \alpha}{2}$$

For regular decode, efficiency is $\frac{1}{1} = 1.0$. Speculative decode matches regular efficiency only at $\alpha = 1.0$.

| Accept rate $\alpha$ | Slot efficiency $\eta$ | Individual speedup | System throughput impact |
|---|---|---|---|
| 1.00 | 1.00 | 2.00x | None (equal to regular) |
| 0.90 | 0.95 | 1.90x | -5% per spec user slot |
| 0.80 | 0.90 | 1.80x | -10% per spec user slot |
| 0.70 | 0.85 | 1.70x | -15% per spec user slot |
| 0.50 | 0.75 | 1.50x | -25% per spec user slot |
| 0.00 | 0.50 | 1.00x | -50% per spec user slot |

**Guidance:** Enable speculative decode when the accept rate exceeds approximately 80% ($\alpha > 0.8$). At this threshold, the per-user latency improvement (1.8x) is substantial, and the system-level throughput loss is moderate (10% per spec user's slot pair). Below 80%, the wasted pipeline bandwidth grows significant relative to the latency benefit.

## Worked Example: Revisiting the 10-Token Trace

Connecting back to the trace from [`result_classification.md`](./result_classification.md):

| Round | Case | Tokens | Slots |
|---|---|---|---|
| 1 (INITIAL) | -- | 1 | 2 |
| 2 | ACCEPT+CONTINUE | 2 | 2 |
| 3 | REJECT+STALE | 1 | 2 |
| 4 | ACCEPT+CONTINUE | 2 | 2 |
| 5 | ACCEPT+CONTINUE | 2 | 2 |
| 6 | ACCEPT+CONTINUE | 2 | 2 |
| **Total** | | **10** | **12** |

**Observed accept rate**: 4 accepts / 5 verification rounds = 80% (the INITIAL round has no verification).

**Duration**: 6 rounds = $6D$ ticks.

**Comparison with regular decode**: 10 tokens at 1 token/round = 10 rounds = $10D$ ticks.

**Speedup**: $10D / 6D = 1.67\text{x}$.

**Expected speedup at $\alpha=0.8$**: $1 + 0.8 = 1.8\text{x}$.

The observed 1.67x is slightly below the expected 1.8x because the INITIAL round contributes only 1 token (no verification, no bonus). For longer generations, the INITIAL overhead is amortized and the actual speedup converges to $1 + \alpha$.

## Worked Example: Mixed Deployment

**Setup:** $D = 64$, $\tau = 44\,\mu s$, accept rate $\alpha = 0.85$.

| Configuration | Users | Slots used | Tokens/round | Aggregate throughput | Per-spec-user TPOT |
|---|---|---|---|---|---|
| 64 regular | 64 regular | 64 | 64 | $\frac{64}{64 \times 44\mu s} = 22{,}727$ tok/s | N/A |
| 20 spec + 24 regular | 44 total | $40 + 24 = 64$ | $20 \times 1.85 + 24 = 61$ | $\frac{61}{64 \times 44\mu s} = 21{,}662$ tok/s | $\frac{2.816\text{ms}}{1.85} = 1.52\text{ms}$ |
| 32 spec | 32 spec | 64 | $32 \times 1.85 = 59.2$ | $\frac{59.2}{64 \times 44\mu s} = 21{,}023$ tok/s | $1.52\text{ms}$ |

The all-regular configuration achieves the highest aggregate throughput. The mixed configuration sacrifices 5% aggregate throughput to give 20 users a 46% per-token latency reduction ($2.816\text{ms} \to 1.52\text{ms}$). This is the typical production sweet spot: a subset of latency-sensitive users get speculative acceleration while the remaining slots serve regular traffic.

## Pipeline Utilization Under Mixed Workloads

When both regular and speculative users are active, the Writer interleaves their staging entries from the same `DecodeStaging` FIFO. A speculative entry produces two injections; a regular entry produces one. The Writer's decode priority (see [Chapter 3 -- Writer Priority](../ch3_scheduling_algorithm/writer_priority_and_decode.md)) does not distinguish between spec and regular entries.

The pipeline utilization is:

$$U = \frac{N_r + 2N_s}{D}$$

With $N_r + 2N_s = D$, utilization is 100% and no bandwidth remains for prefill. With $N_r + 2N_s < D$, the remaining $D - (N_r + 2N_s)$ ticks per period are available for prefill. The IS controls the mix per-user via `GenerationParams::spec_decode`, set in the SUBMIT request.

## Diagnostic Counters: `spec_accepts` and `spec_rejects`

The `UserTable` tracks per-slot accept and reject counts (defined in `user_table.hpp`, lines 43-44):

```cpp
std::vector<uint32_t> spec_accepts;   // incremented on ACCEPT (line 549)
std::vector<uint32_t> spec_rejects;   // incremented on REJECT (line 559)
```

These are plain (non-atomic) `uint32_t` counters, written only by the Reader thread:

- `spec_accepts[uid]++` on ACCEPT (line 549).
- `spec_rejects[uid]++` on REJECT (line 559), conditioned on `has_unverified` being true (avoids counting the INITIAL BASE result as a reject when there was no prior speculation).

### Exposed via Public API

```cpp
uint32_t DecodeScheduler::get_spec_accepts(uint32_t slot_id) const;
uint32_t DecodeScheduler::get_spec_rejects(uint32_t slot_id) const;
```

The inference server can poll these counters to compute a running accept rate:

$$\alpha_{\text{observed}} = \frac{\text{spec\_accepts}}{\text{spec\_accepts} + \text{spec\_rejects}}$$

Since they are plain `uint32_t` and the Reader is the sole writer, there is a potential for torn reads from the IS thread. In practice, this is acceptable for diagnostics -- the counters are monotonically increasing and a torn read would at worst show a stale value, not an incorrect one.

### Counter Lifecycle

The counters are **not** reset by `UserTable::reset(uid)`. Inspection of the `reset()` function in `user_table.hpp` (lines 77-95) confirms that `spec_accepts` and `spec_rejects` are absent from the reset body -- they are initialized to 0 at construction time (line 65-66) but never individually zeroed thereafter. When a slot is freed and reallocated to a new user, the counters carry over from the previous occupant.

| Event | `spec_accepts` | `spec_rejects` |
|---|---|---|
| Construction | 0 | 0 |
| `UserTable::reset(uid)` (ALLOCATE) | **NOT reset** | **NOT reset** |
| SUBMIT | unchanged | unchanged |
| ACCEPT | `++` | unchanged |
| REJECT (with prior prediction) | unchanged | `++` |
| CONTINUE / STALE | unchanged | unchanged |
| CANCEL / cleanup | unchanged | unchanged |

For per-session monitoring, the IS should snapshot the counters at SUBMIT time and compute deltas at completion:

$$\alpha_{\text{session}} = \frac{\Delta\text{spec\_accepts}}{\Delta\text{spec\_accepts} + \Delta\text{spec\_rejects}}$$

---

**Next:** [`thinking_phase_and_relaxed_acceptance.md`](./thinking_phase_and_relaxed_acceptance.md)
