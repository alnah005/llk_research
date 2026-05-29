# 2.1 Configuration Schema Reference

This section provides a complete reference for every struct, field, default, and validation rule in the Pi0.5 pipeline configuration system. The configuration is loaded from a JSON file via `parse_pipeline_config()` and can be selectively overridden by CLI arguments. All structs are defined in `examples/pi05_pipeline_runner.cpp`.

## 2.1.1 Top-Level Struct: `PipelineConfig`

`PipelineConfig` is the root container returned by `parse_pipeline_config()`. It aggregates all sub-configurations and a boolean flag indicating whether socket mode is active.

```cpp
// examples/pi05_pipeline_runner.cpp:248-257
struct PipelineConfig {
    std::vector<PhaseConfig> phases;
    uint32_t input_tokens = 0;
    uint32_t output_tokens = 0;
    uint32_t num_users = 0;
    uint32_t max_turns = 0;
    SocketConfigParams socket;       // populated when "socket" block present
    PixelPayloadConfig pixel;        // populated when "pixel_payload" block present
    bool use_sockets = false;        // true when "socket" block present
};
```

### Field Reference

| JSON Key | C++ Field | Type | Default | Required | Description |
|----------|-----------|------|---------|----------|-------------|
| `"phases"` | `phases` | `vector<PhaseConfig>` | -- | **Yes** | Array of pipeline phase definitions (at least one required) |
| `"input_tokens"` | `input_tokens` | `uint32_t` | `0` (=> CLI default 128) | No | Synthetic prompt length in tokens |
| `"output_tokens"` | `output_tokens` | `uint32_t` | `0` (=> CLI default 64) | No | Max decode tokens per turn |
| `"num_users"` | `num_users` | `uint32_t` | `0` (=> CLI default 4) | No | Concurrent user sessions |
| `"max_turns"` | `max_turns` | `uint32_t` | `0` (=> CLI default 3) | No | Multi-turn rounds per user |
| `"socket"` | `socket` + `use_sockets` | `SocketConfigParams` | -- | No | Socket configuration block; presence sets `use_sockets = true` |
| `"pixel_payload"` | `pixel` | `PixelPayloadConfig` | defaults below | No | Pixel payload dimensions for bulk H2D transfer |

**Note on zero-as-sentinel:** Fields like `input_tokens`, `output_tokens`, `num_users`, and `max_turns` use `0` as a sentinel meaning "not specified in JSON." A value of `0` in the config file is treated identically to the field being absent. The resolution logic in `main()` (lines 778-798) checks `> 0` and applies a three-tier precedence: CLI argument > JSON value > hardcoded default.

## 2.1.2 `PhaseConfig` (Cross-Reference)

Each element of the `phases` array maps to a `PhaseConfig` struct, which was introduced in [Chapter 1](../ch1_architecture_overview_and_operating_modes/index.md) as part of the three-phase pipeline model. The struct definition, derived methods (`num_stages()`, `latency_us()`, `effective_stages()`, `effective_latency_us()`), and their formulas are documented there. This section covers only the JSON-to-C++ field mapping and parsing-specific validation rules.

### Phase JSON Fields

| JSON Key | C++ Field | Type | Default | Constraint |
|----------|-----------|------|---------|------------|
| `"name"` | `name` | `string` | -- | Must be non-empty |
| `"stages"` | `stage_durations_us` | `array of uint32_t` | -- | Must be non-empty; every element must be $\geq 1$ |
| `"loop_count"` | `loop_count` | `uint32_t` | `1` | Must be $\geq 1$ |

### Phase Validation (Lines 387-403)

The parser enforces four invariants per phase immediately after parsing each phase object:

1. **Name required:** `phase.name.empty()` triggers `"Phase missing \"name\" field"`.
2. **Stages required:** `phase.stage_durations_us.empty()` triggers `"Phase \"<name>\" has no stages"`.
3. **Minimum stage duration:** Every element must be $\geq 1\mu s$, checked individually with the stage index in the error message: `"Phase \"<name>\" stage <i> duration must be >= 1us"`.
4. **Loop count minimum:** `loop_count < 1` triggers `"Phase \"<name>\" loop_count must be >= 1"`.

Unknown fields within a phase object are silently skipped via `skip_json_value()` for forward compatibility.

## 2.1.3 `SocketConfigParams` Struct

When the JSON contains a `"socket"` key, the parser sets `use_sockets = true` and populates the `SocketConfigParams` struct:

```cpp
// examples/pi05_pipeline_runner.cpp:217-228
struct SocketConfigParams {
    std::string h2d_token_socket_id;
    std::string d2h_token_socket_id;
    std::string h2d_bulk_socket_id;
    std::string d2h_bulk_socket_id;
    std::string h2d_mode = "DEVICE_PULL";  // "HOST_PUSH" or "DEVICE_PULL"
    uint32_t connect_timeout_ms = 30000;
    double pcie_bandwidth_gbps = 16.0;
    uint32_t fifo_size = 524288;  // 512 KB
    std::string mode = "full";   // "full" (SocketPipeline + bulk) or "bulk_only" (PipelineSimulator + bulk)
    uint32_t num_users_for_launch = 8;  // set at runtime from --num-users
};
```

### Socket JSON Fields

| JSON Key | C++ Field | Type | Default | Constraints |
|----------|-----------|------|---------|-------------|
| `"h2d_token_socket_id"` | `h2d_token_socket_id` | `string` | `""` | Required for `mode = "full"` |
| `"d2h_token_socket_id"` | `d2h_token_socket_id` | `string` | `""` | Required for `mode = "full"` |
| `"h2d_bulk_socket_id"` | `h2d_bulk_socket_id` | `string` | `""` | Required for all socket modes |
| `"d2h_bulk_socket_id"` | `d2h_bulk_socket_id` | `string` | `""` | Required for all socket modes |
| `"h2d_mode"` | `h2d_mode` | `string` | `"DEVICE_PULL"` | `"HOST_PUSH"` or `"DEVICE_PULL"` |
| `"connect_timeout_ms"` | `connect_timeout_ms` | `uint32_t` | `30000` | Milliseconds to wait for socket descriptors |
| `"pcie_bandwidth_gbps"` | `pcie_bandwidth_gbps` | `double` | `16.0` | **Informational only** (see warning below) |
| `"fifo_size"` | `fifo_size` | `uint32_t` | `524288` | Socket FIFO buffer size in bytes (512 KB default) |
| `"mode"` | `mode` | `string` | `"full"` | `"full"` or `"bulk_only"` |

> **WARNING: `pcie_bandwidth_gbps` is informational only.** This field is stored in the config struct and printed in benchmark output, but it is **never used in any bandwidth calculation or transfer logic**. The actual measured bandwidth is computed post-hoc from transfer timestamps: `h2d_bw_gbps = payload_bytes / (h2d_mean_us * 1000.0)` (line 1172). The `parse_float_val()` function reads it (line 435), but `pcie_bandwidth_gbps` itself is purely documentary -- recording the theoretical PCIe link speed of the test system.

The `num_users_for_launch` field is **not settable from JSON** -- it is populated at runtime from the resolved `--num-users` value (line 782):

```cpp
// examples/pi05_pipeline_runner.cpp:782
pipeline_config.socket.num_users_for_launch = num_users;
```

### Socket Mode Semantics

The `mode` field controls which pipeline backend handles the token path:

- **`"full"`** (default): Uses `SocketPipeline` for the token path (real device round-trips via H2D/D2H token sockets) plus bulk H2D/D2H channels for pixel data. Requires all four socket IDs. Requires `PI05_HAS_SOCKETS` at compile time.
- **`"bulk_only"`**: Uses `PipelineSimulator` for the token path (CPU-side timing simulation) plus real bulk H2D/D2H channels. Only requires the two bulk socket IDs. This is the hybrid mode for disconnected N150 boards where the token-path socket infrastructure is unavailable.

In `bulk_only` mode, the token socket IDs (`h2d_token_socket_id`, `d2h_token_socket_id`) are not needed, as shown in `pi05_config_hybrid.json` which omits them.

## 2.1.4 `PixelPayloadConfig` Struct

The `pixel_payload` block defines the geometry of the vision input tensor and auxiliary action history:

```cpp
// examples/pi05_pipeline_runner.cpp:230-239
struct PixelPayloadConfig {
    uint32_t width = 224;
    uint32_t height = 224;
    uint32_t channels = 3;
    uint32_t frames = 3;
    uint32_t action_history_bytes = 4096;

    uint32_t pixel_bytes() const { return width * height * channels * frames; }
    uint32_t total_bytes() const { return pixel_bytes() + action_history_bytes; }
};
```

### Pixel Payload JSON Fields

| JSON Key | C++ Field | Type | Default | Description |
|----------|-----------|------|---------|-------------|
| `"width"` | `width` | `uint32_t` | `224` | Image width in pixels |
| `"height"` | `height` | `uint32_t` | `224` | Image height in pixels |
| `"channels"` | `channels` | `uint32_t` | `3` | Color channels (RGB) |
| `"frames"` | `frames` | `uint32_t` | `3` | Temporal frames in observation window |
| `"action_history_bytes"` | `action_history_bytes` | `uint32_t` | `4096` | Past action chunk appended to pixel data |

### Derived Methods

$$\texttt{pixel\_bytes()} = \texttt{width} \times \texttt{height} \times \texttt{channels} \times \texttt{frames}$$

With default values:

$$\texttt{pixel\_bytes()} = 224 \times 224 \times 3 \times 3 = 451{,}584 \text{ bytes} \approx 441 \text{ KB}$$

$$\texttt{total\_bytes()} = \texttt{pixel\_bytes()} + \texttt{action\_history\_bytes} = 451{,}584 + 4{,}096 = 455{,}680 \text{ bytes} \approx 445 \text{ KB}$$

These dimensions model a Pi0.5 robotics inference input: three consecutive $224 \times 224$ RGB frames from a camera feed, plus a 4 KB action history buffer from previous turns. The bulk D2H transfer is a fixed 3,200 bytes (`ACTION_OUTPUT_BYTES = 50 x 32 bfloat16`, line 203).

## 2.1.5 CLI Argument Override Precedence

The pipeline runner resolves each configurable parameter using a strict three-tier precedence. CLI arguments always win, then JSON values, then hardcoded defaults. This logic is implemented in `main()` at lines 778-798:

```cpp
// examples/pi05_pipeline_runner.cpp:778-798
const uint32_t num_users = num_users_explicit ? num_users_cli
    : (pipeline_config.num_users > 0 ? pipeline_config.num_users : 4);
// ...
const uint32_t prompt_len = prompt_len_explicit ? prompt_len_cli
    : (pipeline_config.input_tokens > 0 ? pipeline_config.input_tokens : 128);
// ...
const uint32_t max_decode = max_decode_explicit ? max_decode_cli
    : (pipeline_config.output_tokens > 0 ? pipeline_config.output_tokens : 64);
// ...
const uint32_t max_turns = max_turns_explicit ? max_turns_cli
    : (pipeline_config.max_turns > 0 ? pipeline_config.max_turns : 3);
```

The precedence chain is:

$$\texttt{resolved\_value} = \begin{cases} \texttt{CLI argument} & \text{if \texttt{--flag} provided} \\ \texttt{JSON value} & \text{if JSON field} > 0 \\ \texttt{hardcoded default} & \text{otherwise} \end{cases}$$

### Overridable Parameters

| CLI Flag | JSON Key | Hardcoded Default | Resolution |
|----------|----------|-------------------|------------|
| `--num-users N` | `"num_users"` | 4 | CLI wins if present; else JSON if `> 0`; else `4` |
| `--prompt-len N` | `"input_tokens"` | 128 | CLI wins if present; else JSON if `> 0`; else `128` |
| `--max-decode N` | `"output_tokens"` | 64 | CLI wins if present; else JSON if `> 0`; else `64` |
| `--max-turns N` | `"max_turns"` | 3 | CLI wins if present; else JSON if `> 0`; else `3` |
| `--accept-rate F` | -- | 1.0 | CLI only -- no JSON equivalent |
| `--seed N` | -- | 42 | CLI only -- no JSON equivalent |
| `--timeout-s N` | -- | 30 | CLI only -- no JSON equivalent |
| `--stagger-us N` | -- | auto | CLI only; auto = `total_latency_us / num_users` |

### Non-Overridable Config

The following are consumed directly from JSON with no CLI override mechanism:

- `phases` array (pipeline structure)
- `socket` block (all socket parameters)
- `pixel_payload` block (vision input geometry)

The `--config FILE` argument itself is the only required CLI flag.

The stagger delay has a special auto-computation default (lines 831-832):

```cpp
// examples/pi05_pipeline_runner.cpp:831-832
const uint32_t stagger_us = stagger_explicit ? stagger_us_cli
    : static_cast<uint32_t>(total_latency_us / num_users);
```

This spaces initial user submissions evenly across the full pipeline latency, ensuring that users fill the pipeline uniformly rather than all competing for the first stage simultaneously.

## 2.1.6 Consolidated Validation Rules

The parser and `main()` enforce several validation constraints. Violations throw `std::runtime_error` with a descriptive message.

### Parse-Time Validation (in `parse_pipeline_config`)

| Rule | Location | Error Message |
|------|----------|---------------|
| At least one phase must exist | Line 476-478 | `"Config must contain at least one phase"` |
| Every phase must have a `"name"` field | Line 387-389 | `"Phase missing \"name\" field"` |
| Every phase must have at least one stage | Line 390-392 | `"Phase \"<name>\" has no stages"` |
| Every stage duration must be $\geq 1\mu s$ | Lines 393-399 | `"Phase \"<name>\" stage <i> duration must be >= 1us"` |
| `loop_count` must be $\geq 1$ | Lines 400-403 | `"Phase \"<name>\" loop_count must be >= 1"` |

### Runtime Validation (in `main`)

| Rule | Location | Error Message |
|------|----------|---------------|
| `num_users` must be in $[1, 1024]$ | Lines 781-783 | `"num_users must be in [1, 1024]"` |
| `input_tokens` must be $\geq 1$ | Lines 786-788 | `"input_tokens must be >= 1"` |
| `output_tokens` must be $\geq 1$ | Lines 791-793 | `"output_tokens must be >= 1"` |
| `max_turns` must be $\geq 1$ | Lines 796-798 | `"max_turns must be >= 1"` |
| Total pipeline stages must be $\geq 1$ | Lines 808-809 | `"Total pipeline stages must be >= 1"` |
| Effective stage duration must be $\geq 1\mu s$ | Lines 811-813 | `"Effective stage duration must be >= 1us"` |
| `--accept-rate` must be in $[0.0, 1.0]$ | Lines 771-773 | `"--accept-rate must be in [0.0, 1.0]"` |
| `--prompt-len` must be $\geq 1$ (if explicit) | Lines 765-767 | `"--prompt-len must be >= 1"` |
| `--max-turns` must be $\geq 1$ (if explicit) | Lines 768-770 | `"--max-turns must be >= 1"` |

### Socket-Specific Validation: FIFO Size Check

When `use_sockets` is `true`, an additional validation ensures the maximum padded H2D payload fits within the FIFO buffer (lines 848-856):

```cpp
// examples/pi05_pipeline_runner.cpp:848-856
uint32_t max_payload = sizeof(BulkH2DHeader) + pipeline_config.pixel.total_bytes();
uint32_t max_padded = ((max_payload + BULK_PAGE_SIZE - 1) / BULK_PAGE_SIZE) * BULK_PAGE_SIZE;
if (max_padded > pipeline_config.socket.fifo_size) {
    throw std::runtime_error("Padded H2D payload (" + std::to_string(max_padded) +
        " bytes) exceeds FIFO size (" + std::to_string(pipeline_config.socket.fifo_size) + ")");
}
```

In formula form:

$$\texttt{max\_padded} = \left\lceil \frac{\texttt{sizeof(BulkH2DHeader)} + \texttt{pixel.total\_bytes()}}{\texttt{BULK\_PAGE\_SIZE}} \right\rceil \times \texttt{BULK\_PAGE\_SIZE}$$

With default values:

$$\texttt{max\_payload} = 8 + 455{,}680 = 455{,}688 \text{ bytes}$$

$$\texttt{max\_padded} = \lceil 455{,}688 / 4{,}096 \rceil \times 4{,}096 = 112 \times 4{,}096 = 458{,}752 \text{ bytes}$$

$$458{,}752 \leq 524{,}288 \quad \checkmark$$

The `BULK_PAGE_SIZE` constant (line 201) is 4,096 bytes, matching the system page size used by the tt-metal socket layer.

---

**Next:** [`02_custom_json_parser.md`](./02_custom_json_parser.md)
