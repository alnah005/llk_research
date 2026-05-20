# Multi-Turn Continue

[< Session State Machine](session_state_machine.md) | [Deferred Cancellation >](deferred_cancellation.md)

## Overview

Multi-turn conversation is one of the most performance-critical features in a serving engine. A naive implementation would discard the KV cache after each generation and re-prefill the entire conversation history on every turn. tt-llm-engine avoids this through the `CONTINUE` request type, which reuses the existing KV cache and only prefills the new tokens (the user's latest message plus any system tokens).

There are two continuation paths, selected based on whether the deployment uses disaggregated prefill:

| Path | Entry Function | New State | KV Reuse | Use Case |
|------|---------------|-----------|----------|----------|
| Local continue | `handle_local_continue` | PREFILL | Yes -- KV from prior turns preserved | Standard single-node or co-located deployment |
| Disaggregated continue | `handle_disaggregated_continue` | DECODE | Partial -- only the migrated token position | Disaggregated prefill/decode with separate nodes |

## GenerationParams

Every `SUBMIT` and `CONTINUE` request carries a `GenerationParams` struct that controls the generation behavior for that turn:

```cpp
struct GenerationParams {
    uint32_t max_new_tokens = 0;
    bool     spec_decode = false;
    bool     ignore_eos = false;
    float    temperature = 1.0f;
    float    top_p = 1.0f;
    int32_t  top_k = 1;
    bool     disaggregated_decode = false;
    float    relaxed_acceptance_threshold = 0.0f;
};
```

**Per-turn semantics:** Generation parameters are reset on every `SUBMIT` and `CONTINUE`. The IS can change sampling parameters (temperature, top-p, top-k), toggle speculative decode, or adjust `max_new_tokens` between turns. The scheduler does not carry forward any generation parameters from a prior turn.

## Local Continue: `handle_local_continue`

This is the standard multi-turn path. The key insight is that the KV cache from all prior turns is still resident in device memory, so the new turn only needs to prefill the delta tokens.

### Implementation

```cpp
void handle_local_continue(const ISRequest& req) {
    uint32_t uid = req.slot_id;
    uint32_t start_pos = user_table.current_position[uid].load(
        std::memory_order_relaxed);

    prompt_table.store(uid, req.tokens.data(), ...);

    user_table.state[uid].store(UserState::PREFILL,
        std::memory_order_release);
    user_table.prefill_pos[uid] = start_pos;
    user_table.prefill_start_pos[uid] = start_pos;
    // ... reinitialize gen params from req.gen ...
    // ... reset tokens_generated to 0 ...
    // ... reset spec_state ...

    prefill_queue.push(uid);
}
```

### How KV Reuse Works

The critical difference between `SUBMIT` and local `CONTINUE` is the position initialization:

| Field | SUBMIT | Local CONTINUE |
|-------|--------|----------------|
| `current_position` | 0 | **Preserved** (read from prior turn) |
| `prefill_pos` | 0 | `start_pos` (= prior `current_position`) |
| `prefill_start_pos` | 0 | `start_pos` |
| `tokens_generated` | 0 | 0 (reset) |

By setting `prefill_pos = prefill_start_pos = start_pos`, the Writer knows to begin chunked prefill at the position where the last generation ended. All KV entries from positions $[0, \text{start\_pos})$ are already populated in the device's KV cache and do not need to be recomputed.

### Memory Order Reasoning

The `state` store uses `memory_order_release`:

```cpp
user_table.state[uid].store(UserState::PREFILL, std::memory_order_release);
```

This guarantees that the Writer thread, when it loads the state with `memory_order_acquire`, will observe all the preceding writes (`prefill_pos`, `prefill_start_pos`, generation params, etc.) as fully committed. This is the same release-acquire pattern used in `SUBMIT` (see Ch2 for the UserTable memory ordering contract).

## Worked Example: Slot 7, Turn 2

Continuing from the [session state machine example](session_state_machine.md), slot 7 has just completed its first turn at position 192 (128 prompt + 64 generated). The IS sends a CONTINUE with 32 new tokens.

### IS Sends CONTINUE

```cpp
ISRequest cont_req;
cont_req.type       = RequestType::CONTINUE;
cont_req.request_id = 1003;
cont_req.slot_id    = 7;
cont_req.tokens     = {cont_tok_0, ..., cont_tok_31};  // 32 new tokens
cont_req.gen        = GenerationParams{
    .max_new_tokens       = 48,
    .temperature          = 0.6f,
    .top_p                = 0.95f,
    .top_k                = 40,
    .disaggregated_decode = false,
};
push_request(cont_req);
```

### KV Layout After Turn-2 Prefill

Because `prefill_pos = 192`, the prefill pipeline writes the new 32 tokens into KV cache positions 192--223, seamlessly extending the existing context:

```
KV Cache:  0         128       192     224       272
           +----------+---------+--------+---------+
Turn 1:    [prompt 128 tokens  ][gen 64 ]
Turn 2:                                  [usr2 32 ][gen2 48]
```

Positions 0--191 are untouched -- the KV entries from Turn 1 remain resident in device memory.

### Trace Timeline

| Time | Thread | Action | Position |
|------|--------|--------|----------|
| T+1.520ms | IS | `push_request(CONTINUE, 1003, slot=7, 32 toks)` | 192 |
| T+1.522ms | Scheduler | `handle_local_continue`: set PREFILL, prefill_pos=192 | 192 |
| T+1.600ms | Writer | Prefill 32 tokens at positions 192--223, transition to DECODE | 224 |
| T+1.620ms | Reader | Generate token 1 of turn 2 | 225 |
| ... | Reader | Tokens 2--48 | 226--272 |
| T+2.560ms | Reader | Token 48 (or EOS), set COMPLETE | 272 |

## Disaggregated Continue: `handle_disaggregated_continue`

In a disaggregated deployment, prefill and decode run on separate physical nodes. The prefill node handles prompt processing and then migrates the final KV state to the decode node. For a `CONTINUE` in this mode, the decode scheduler receives a single "migrated token" and transitions directly into `DECODE`, bypassing the local prefill pipeline entirely.

### Implementation

```cpp
void handle_disaggregated_continue(const ISRequest& req) {
    uint32_t uid = req.slot_id;
    uint32_t n = static_cast<uint32_t>(req.tokens.size() - 1);
    uint32_t migrated_token = req.tokens.back();

    user_table.state[uid].store(UserState::DECODE,
        std::memory_order_release);
    user_table.current_position[uid].store(n + 1,
        std::memory_order_release);
    // ... reinitialize gen params from req.gen ...

    decode_staging.stage(uid, migrated_token, n, EMPTY_TOKEN, 0);
}
```

### Key Differences from Local Continue

| Aspect | Local Continue | Disaggregated Continue |
|--------|---------------|----------------------|
| New state | PREFILL | DECODE (directly) |
| KV source | Reuses local KV cache | KV migrated from prefill node |
| Token handling | All new tokens go through chunked prefill | Last token staged directly into decode |
| Position calculation | `start_pos` from prior `current_position` | $n + 1$ where $n = \text{tokens.size()} - 1$ |
| Pipeline involvement | Writer runs prefill chunks | Writer skipped; decode path starts immediately |

### Token Convention

The `req.tokens` vector in a disaggregated continue has a specific convention:
- Tokens $[0, n)$ represent the context that was prefilled on the remote prefill node.
- Token $[n]$ (the last element, `req.tokens.back()`) is the "migrated token" -- the first token that the decode node will use to begin autoregressive generation.

The position is set to $n + 1$ because the KV cache on the decode side is assumed to contain entries for all $n + 1$ positions ($[0, n]$ inclusive), populated by the KV migration from the prefill node.

### Staging Into Decode

Instead of going through the prefill queue and Writer, the migrated token is placed directly into `decode_staging`:

```cpp
decode_staging.stage(uid, migrated_token, n, EMPTY_TOKEN, 0);
```

The `EMPTY_TOKEN` and `0` for the speculative pair indicate that there is no speculative token -- the first decode step is always non-speculative. The Writer will pick up this staged token in its next decode batch (see Ch3 for `DecodeStaging` consumption).

## Multi-Turn Session: Cumulative KV Cache Growth

Over multiple turns, the KV cache grows monotonically:

$$\text{ctx\_used}(T) = \sum_{t=1}^{T} \left( \text{prompt\_len}_t + \text{generated\_len}_t \right)$$

For the slot-7 worked example across three turns:

| Turn | Prompt Tokens | Generated Tokens | Cumulative Position | Remaining Budget ($\text{max\_seq\_len} = 2048$) |
|------|--------------|-------------------|---------------------|------|
| 1 | 128 | 64 | 192 | 1856 |
| 2 | 32 | 48 | 272 | 1776 |
| 3 | 24 | 56 | 352 | 1696 |

If the budget reaches zero, context exhaustion occurs. See [Context Exhaustion](context_exhaustion.md).

## Cost Model

The performance advantage of `CONTINUE` over re-submitting the full conversation is significant:

$$\text{Prefill savings} = \frac{\text{reused\_positions}}{\text{total\_positions}} \times \text{prefill\_time}$$

For a typical multi-turn conversation where each user message is much shorter than the accumulated context, this ratio approaches 1.0. A 10-turn conversation with 4096 total context tokens and a 128-token new message achieves:

$$\text{Savings} = \frac{3968}{4096} \approx 96.9\%$$

This is why `CONTINUE` is the preferred path for all multi-turn serving, and why the KV cache is not cleared on `COMPLETE`.

## Edge Cases

**CONTINUE on a non-COMPLETE slot.** The IS should wait for `is_complete = true` before sending CONTINUE. The scheduler does not explicitly reject a premature CONTINUE, but the behavior is undefined.

**CONTINUE with zero tokens.** Not a valid operation. The IS must always provide at least one new token.

**CONTINUE after context exhaustion.** If the previous turn ended with `ctx_exhausted = true`, the IS should not send CONTINUE -- there is no room for new tokens. Send `CANCEL` and start a fresh session instead. See [Context Exhaustion](context_exhaustion.md).

---

*[< Session State Machine](session_state_machine.md) | [Deferred Cancellation >](deferred_cancellation.md)*
