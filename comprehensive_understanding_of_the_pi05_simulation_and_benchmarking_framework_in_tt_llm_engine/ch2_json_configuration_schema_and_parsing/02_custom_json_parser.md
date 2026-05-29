# 2.2 Custom JSON Parser

The pipeline runner includes a hand-rolled recursive-descent JSON parser implemented entirely within the `parse_pipeline_config()` function. This section documents its design rationale, structure, forward-compatibility mechanism, comma-handling strategies, and known limitations.

## 2.2.1 Why a Custom Parser?

The `parse_pipeline_config()` function (lines 261-480 of `examples/pi05_pipeline_runner.cpp`) is a self-contained ~120-line JSON parser with zero external dependencies. The design is motivated by several constraints:

1. **No external JSON dependency.** The pipeline runner is a standalone example binary. Adding a dependency on nlohmann/json, RapidJSON, or any other library would complicate the build and increase the binary's dependency footprint.

2. **Minimal scope.** The parser only needs to handle the specific JSON dialect used by the config files: top-level objects with string keys, nested objects (socket, pixel_payload, phase objects), arrays of integers (stage durations) and arrays of objects (phases), unsigned integers, floating-point numbers, and strings. Full JSON compliance (Unicode escapes, arbitrary nesting, etc.) is unnecessary.

3. **Forward compatibility via `skip_json_value`.** Unknown keys at any nesting level are silently skipped, allowing older binaries to read configs with new fields added in later revisions.

4. **Single-function encapsulation.** The entire parser lives in one function using C++ lambdas and `std::function`, keeping the implementation local and easily auditable. It does **not** produce a generic DOM. Instead, it directly populates the `PipelineConfig` struct during a single pass through the input.

## 2.2.2 Parser Structure

The parser operates on a `std::string content` buffer loaded from the config file, with a single `size_t pos` cursor that advances through the input. All parsing primitives are defined as lambdas that capture `content` and `pos` by reference.

### Primitive Functions

#### `skip_ws()`
```cpp
// examples/pi05_pipeline_runner.cpp:272-274
auto skip_ws = [&]() {
    while (pos < content.size() && std::isspace(content[pos])) pos++;
};
```
Advances `pos` past any whitespace characters (spaces, tabs, newlines). Called at the beginning of every other primitive.

#### `expect(char c)`
```cpp
// examples/pi05_pipeline_runner.cpp:275-283
auto expect = [&](char c) {
    skip_ws();
    if (pos >= content.size() || content[pos] != c) {
        throw std::runtime_error(
            "Config parse error at position " + std::to_string(pos) +
            ": expected '" + c + "', got '" +
            (pos < content.size() ? std::string(1, content[pos]) : "EOF") + "'");
    }
    pos++;
};
```
Skips whitespace, then asserts that the current character matches `c` and consumes it. On mismatch, throws with a position-based error message showing the expected and actual characters.

#### `peek()`
```cpp
// examples/pi05_pipeline_runner.cpp:284-287
auto peek = [&]() -> char {
    skip_ws();
    return pos < content.size() ? content[pos] : '\0';
};
```
Skips whitespace and returns the current character without consuming it. Returns `'\0'` at end-of-input. Used extensively for lookahead to determine which branch of the parser to take (e.g., checking for `}`, `]`, or `,`).

#### `parse_string()`
```cpp
// examples/pi05_pipeline_runner.cpp:288-296
auto parse_string = [&]() -> std::string {
    skip_ws();
    expect('"');
    std::string result;
    while (pos < content.size() && content[pos] != '"') {
        result += content[pos++];
    }
    expect('"');
    return result;
};
```
Parses a JSON string by consuming characters between double quotes. The opening quote is consumed by `expect('"')`, then characters are appended one-by-one to the result string until the next unescaped `"`, which is consumed by the second `expect('"')`. **Important limitation:** This function does not handle escape sequences -- see Section 2.2.6.

#### `parse_number()`
```cpp
// examples/pi05_pipeline_runner.cpp:297-306
auto parse_number = [&]() -> uint32_t {
    skip_ws();
    size_t start = pos;
    while (pos < content.size() && std::isdigit(content[pos])) pos++;
    if (pos == start) {
        throw std::runtime_error(
            "Config parse error at position " + std::to_string(pos) + ": expected a number");
    }
    return static_cast<uint32_t>(std::stoul(content.substr(start, pos - start)));
};
```
Parses a sequence of digit characters and converts to `uint32_t` via `std::stoul`. Only consumes ASCII digits `[0-9]`. **Important limitation:** Does not handle negative integers or floating-point values -- see Section 2.2.6.

#### `parse_float_val()`
```cpp
// examples/pi05_pipeline_runner.cpp:307-319
auto parse_float_val = [&]() -> double {
    skip_ws();
    size_t start = pos;
    if (pos < content.size() && content[pos] == '-') pos++;
    while (pos < content.size() && (std::isdigit(content[pos]) || content[pos] == '.' ||
           content[pos] == 'e' || content[pos] == 'E' || content[pos] == '+' || content[pos] == '-'))
        pos++;
    if (pos == start) {
        throw std::runtime_error(
            "Config parse error at position " + std::to_string(pos) + ": expected a float");
    }
    return std::stod(content.substr(start, pos - start));
};
```
Parses a floating-point number including optional leading negative sign, decimal point, and scientific notation (`e`/`E` with optional `+`/`-`). Unlike `parse_number()`, this function supports negative values and fractional parts. Used only for the `pcie_bandwidth_gbps` field.

**Note the asymmetry:** `parse_number()` cannot handle negatives, but `parse_float_val()` can. This is intentional -- all integer fields in the config are unsigned (`uint32_t`), while `pcie_bandwidth_gbps` is a `double`.

### Primitive Call Graph

```
parse_pipeline_config()
  |
  +-- skip_ws()          Advance past whitespace
  +-- expect(char)       Consume a specific character or throw
  +-- peek()             Look at next non-whitespace character
  +-- parse_string()     Consume "..." and return contents
  +-- parse_number()     Consume digits and return uint32_t
  +-- parse_float_val()  Consume a floating-point literal (with sign, exponent)
  +-- skip_json_value()  Recursively skip any JSON value (forward compat)
```

### Key Design Decisions

1. **`expect()` includes `skip_ws()`**: Every structural character match automatically skips leading whitespace, so callers never need to call `skip_ws()` before `expect()`. Note that `parse_string()` calls `skip_ws()` explicitly before `expect('"')` -- the `skip_ws()` inside `expect` makes the explicit call redundant, but this is how the source code reads.

2. **`peek()` returns `'\0'` on EOF**: This lets callers use simple `peek() != '}'` loops without separate bounds checking.

3. **`parse_number()` returns `uint32_t`**: The config format uses only non-negative integers for stage durations, counts, sizes, and timeouts. The parser uses `std::stoul()` for conversion.

4. **`parse_float_val()` supports signed floats**: Unlike `parse_number()`, this function handles leading `-`, decimal points, and scientific notation. It is used only for `pcie_bandwidth_gbps`.

## 2.2.3 Forward Compatibility: `skip_json_value`

The most architecturally significant component of the parser is the `skip_json_value` function (lines 324-354):

```cpp
// examples/pi05_pipeline_runner.cpp:324-354
std::function<void()> skip_json_value;
skip_json_value = [&]() {
    skip_ws();
    if (pos >= content.size()) throw std::runtime_error("Unexpected end of config");
    char c = content[pos];
    if (c == '"') { parse_string(); }
    else if (c == '{') {
        pos++;
        while (peek() != '}') {
            if (peek() == ',') pos++;
            parse_string(); expect(':'); skip_json_value();
        }
        expect('}');
    } else if (c == '[') {
        pos++;
        bool first = true;
        while (peek() != ']') {
            if (!first) { if (peek() == ',') pos++; }
            first = false;
            skip_json_value();
        }
        expect(']');
    } else if (c == 't' || c == 'f' || c == 'n') {
        while (pos < content.size() && std::isalpha(content[pos])) pos++;
    } else {
        if (c == '-') pos++;
        while (pos < content.size() && (std::isdigit(content[pos]) || content[pos] == '.' ||
               content[pos] == 'e' || content[pos] == 'E' || content[pos] == '+' || content[pos] == '-'))
            pos++;
    }
};
```

This function must be declared as `std::function<void()>` rather than `auto` because it is **recursive** -- objects and arrays can contain nested objects and arrays. The lambda captures itself by reference.

### Value Type Dispatch

`skip_json_value` dispatches on the first non-whitespace character:

| First Character | Type Detected | Skip Strategy |
|-----------------|---------------|---------------|
| `"` | String | Delegates to `parse_string()` (consumes and discards) |
| `{` | Object | Recursively skips key-value pairs until `}` |
| `[` | Array | Recursively skips elements until `]` |
| `t`, `f`, `n` | Boolean/null | Consumes alphabetic chars (`true`, `false`, `null`) |
| Digit or `-` | Number | Consumes digit/float chars (same pattern as `parse_float_val`) |

### Where `skip_json_value` Is Invoked

The function is called in four places, each guarding an unknown-key path:

1. **Unknown field in a phase object** (line 382): Allows phases to carry extra metadata without breaking the parser.

```cpp
// Line 383: Unknown field inside a phase object
} else {
    // Unknown field in phase -- skip for forward compatibility
    skip_json_value();
}
```

2. **Unknown field in the socket block** (line 442): Allows new socket parameters in future config revisions.

```cpp
// Line 441: Unknown field inside the socket block
} else {
    // Unknown field in socket -- skip for forward compatibility
    skip_json_value();
}
```

3. **Unknown field in pixel_payload** (line 464): Future vision config fields are silently ignored.

```cpp
// Line 463: Unknown field inside the pixel_payload block
} else {
    // Unknown field in pixel_payload -- skip for forward compatibility
    skip_json_value();
}
```

4. **Unknown top-level key** (line 470): Any root-level key other than `phases`, `input_tokens`, `output_tokens`, `num_users`, `max_turns`, `socket`, or `pixel_payload` is skipped.

```cpp
// Line 469: Unknown top-level key
} else {
    // Unknown top-level key -- skip for forward compatibility
    skip_json_value();
}
```

This design means a future config revision could add any new fields -- strings, nested objects, arrays of arrays -- without breaking older binaries. The parser will simply skip over values it does not recognize.

## 2.2.4 Parser Control Flow Diagram

The top-level parsing flow through `parse_pipeline_config()` follows this structure:

```
parse_pipeline_config(path)
+-- Read file into string
+-- expect('{')                          // top-level object
+-- while (peek() != '}')               // iterate top-level keys
|   +-- parse_string() -> key
|   +-- expect(':')
|   +-- if key == "phases"
|   |   +-- expect('[')
|   |   +-- while (peek() != ']')       // iterate phase objects
|   |       +-- expect('{')
|   |       +-- while (peek() != '}')   // iterate phase fields
|   |       |   +-- parse_string() -> field
|   |       |   +-- expect(':')
|   |       |   +-- "name"    -> parse_string()
|   |       |   +-- "stages"  -> expect('['), parse_number()*, expect(']')
|   |       |   +-- "loop_count" -> parse_number()
|   |       |   +-- else      -> skip_json_value()
|   |       +-- expect('}')
|   |       +-- validate phase (name, stages, loop_count)
|   +-- if key == "input_tokens" -> parse_number()
|   +-- if key == "output_tokens" -> parse_number()
|   +-- if key == "num_users" -> parse_number()
|   +-- if key == "max_turns" -> parse_number()
|   +-- if key == "socket"
|   |   +-- set use_sockets = true
|   |   +-- expect('{')
|   |   +-- while (peek() != '}')       // iterate socket fields
|   |       +-- string fields -> parse_string()
|   |       +-- uint32 fields -> parse_number()
|   |       +-- float fields  -> parse_float_val()
|   |       +-- else          -> skip_json_value()
|   +-- if key == "pixel_payload"
|   |   +-- expect('{')
|   |   +-- while (peek() != '}')       // iterate pixel fields
|   |       +-- uint32 fields -> parse_number()
|   |       +-- else          -> skip_json_value()
|   +-- else -> skip_json_value()        // unknown top-level key
|   +-- consume optional comma
+-- expect('}')
+-- validate: phases non-empty
```

The parser makes a single left-to-right pass through the file with no backtracking. Each key-value pair is processed exactly once, and unknown keys are skipped in $O(n)$ where $n$ is the size of the skipped value. The total complexity is $O(N)$ where $N$ is the file size.

## 2.2.5 Comma-Handling Strategies

The parser uses three different comma-handling strategies, which is worth understanding because it affects what inputs are accepted or rejected:

**Strategy 1 -- Strict enforcement (phases array and stages array):** Commas between elements are consumed via `expect(',')` when a prior element has already been parsed. This means a missing comma produces an error, and a trailing comma after the last element (e.g., `[{...}, {...},]`) also produces an error because `expect(',')` would be followed by `]`, which the next `expect('{')` would reject.

```cpp
// examples/pi05_pipeline_runner.cpp:362-363
if (!config.phases.empty()) expect(',');
```

**Strategy 2 -- Tolerant peek-and-consume (object fields):** Within object bodies (phase fields, socket fields, pixel_payload fields), commas are consumed via a peek-and-consume pattern. This is more tolerant -- it allows (but does not require) commas between key-value pairs:

```cpp
// examples/pi05_pipeline_runner.cpp:367
if (peek() == ',') { pos++; }
```

**Strategy 3 -- Tolerant trailing (top-level):** After each top-level key-value pair, a comma is optionally consumed:

```cpp
// examples/pi05_pipeline_runner.cpp:472
if (peek() == ',') pos++;
```

**Summary of implications:** Trailing commas at the top level and within objects are tolerated, but trailing commas in the phases array and stages array are **not** tolerated. Missing commas between object fields are tolerated (the parser simply looks for the next `"` to start a new key), but missing commas in arrays produce parse errors.

## 2.2.6 Known Limitations and Edge Cases

The parser is intentionally minimal and has several known limitations:

### 1. No Escape Sequence Handling in `parse_string`

The `parse_string()` function reads characters verbatim between quotes. If a string contains `\"`, `\\`, `\n`, `\t`, or Unicode escapes (`\uXXXX`), they will be included literally in the result rather than being interpreted. For example:

```json
{"name": "phase \"one\""}
```

would terminate parsing at the first `\"` and produce a parse error, because the backslash is consumed as a literal character and the following `"` is treated as the closing quote.

**Impact:** Low. All config string values are simple identifiers (`"vision"`, `"pi05_h2d_bulk"`, `"DEVICE_PULL"`) that never contain special characters.

### 2. No Negative Integer Support in `parse_number`

The `parse_number()` function only accepts `[0-9]+`. A value like `"input_tokens": -1` would cause a parse error at the `-` character.

**Impact:** None for current usage. All integer config fields are `uint32_t` and semantically non-negative. Note that `parse_float_val()` **does** handle negative values.

### 3. No Trailing Comma Tolerance in Arrays

As discussed in Section 2.2.5, arrays use `expect(',')` for element separation. A trailing comma before `]` will produce:

```
Config parse error at position N: expected a number
```

(for the stages array) or a phase parsing error (for the phases array).

### 4. No Duplicate Key Detection

If the same key appears twice at the same level (e.g., two `"phases"` arrays), the second value silently overwrites the first. No warning or error is produced. For example:

```json
{"num_users": 8, "num_users": 16}
```

would result in `num_users = 16` with no warning.

### 5. No Line/Column Error Reporting

Error messages report the byte position in the file (`"Config parse error at position 142"`), not line and column numbers. For multi-line JSON files, this requires manual counting or a text editor with byte offset display to locate the error. The byte position is relative to the start of the file.

### 6. No Comment Support

JSON does not support comments natively, and neither does this parser. A `//` or `/* */` comment in the config file will cause a parse error. If comments are needed, a workaround is to use `"_comment": "..."` keys, which would be silently skipped by `skip_json_value()`.

## 2.2.7 Error Reporting Summary

All parser errors are thrown as `std::runtime_error`. The complete catalog of error message patterns:

| Error | Source | Message Pattern |
|-------|--------|-----------------|
| File not found | Line 264 | `"Cannot open config file: <path>"` |
| Unexpected character | `expect()`, line 278 | `"Config parse error at position N: expected 'X', got 'Y'"` |
| Expected number | `parse_number()`, line 305 | `"Config parse error at position N: expected a number"` |
| Expected float | `parse_float_val()`, line 319 | `"Config parse error at position N: expected a float"` |
| Unexpected EOF | `skip_json_value()`, line 327 | `"Unexpected end of config"` |
| Missing phase name | Line 388 | `"Phase missing \"name\" field"` |
| Empty stages | Line 391 | `"Phase \"<name>\" has no stages"` |
| Stage duration < 1 | Line 395 | `"Phase \"<name>\" stage N duration must be >= 1us"` |
| Loop count < 1 | Line 401 | `"Phase \"<name>\" loop_count must be >= 1"` |
| No phases at all | Line 477 | `"Config must contain at least one phase"` |

Structural errors (missing required fields, invalid stage durations) are reported with more specific messages that include the phase name and field context. These validation errors are thrown after the phase object is fully parsed, during the post-parse validation block (lines 387-403).

---

**Next:** [`03_reference_configs_walkthrough.md`](./03_reference_configs_walkthrough.md)
