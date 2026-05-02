# 7.1 Host-Device Communication

This section documents the complete host-side communication stack used by the LLK test infrastructure to talk to a Tenstorrent device. Every operation -- from detecting which chip is installed, to writing a single word into L1, to sending power-management ARC messages -- flows through the `tt-exalens` library. Understanding this layer is essential because pure-LLK testing bypasses TT-Metal entirely: the Python test harness is the only runtime, and `tt-exalens` is its sole hardware interface.

---

## 7.1.1 The tt-exalens Library: Role and API Surface

The `tt-exalens` library (imported as `ttexalens`) provides a low-level debug and control interface to Tenstorrent silicon. In the LLK test infrastructure, it serves three roles:

1. **Memory access** -- reading and writing arbitrary addresses in L1 and configuration register space.
2. **ELF loading** -- parsing RISC-V ELF files and writing their segments to the correct device memory addresses.
3. **Debug control** -- reading soft-reset registers, injecting instructions, and inspecting RISC core state (program counter, callstack, ebreak detection).

The test infrastructure imports these functions in `device.py`:

```python
from ttexalens.context import Context
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.debug_tensix import TensixDebug
from ttexalens.hardware.risc_debug import CallstackEntry
from ttexalens.tt_exalens_lib import (
    ParsedElfFile,
    TTException,
    arc_msg,
    callstack,
    check_context,
    convert_coordinate,
    parse_elf,
    read_from_device,
    read_word_from_device,
    write_to_device,
    write_words_to_device,
)
```

Note that `load_elf` is imported separately in `test_config.py` from the same library. Together, these functions constitute the complete API surface between the host and the Tensix core.

### Core Memory Functions

| Function | Signature | Purpose |
|----------|-----------|---------|
| `write_to_device` | `(location, addr, data_bytes)` | Write a raw byte buffer to L1 at the specified address |
| `write_words_to_device` | `(location, addr, [word_list])` | Write a list of 32-bit words to L1 |
| `read_from_device` | `(location, addr, num_bytes)` | Read `num_bytes` from L1, returning a `bytes` object |
| `read_word_from_device` | `(location, addr, device_id)` | Read a single 32-bit word from L1 |

The `location` parameter is a NOC coordinate string like `"0,0"` identifying which Tensix tile to target. For single-core LLK testing, this is always `"0,0"` -- the first worker core on the grid.

### ELF Loading Functions

| Function | Signature | Purpose |
|----------|-----------|---------|
| `load_elf` | `(elf_file, location, risc_name, ...)` | Parse and load an ELF file to the device |
| `parse_elf` | `(elf_path, context)` | Parse an ELF without loading; returns `ParsedElfFile` with symbol table |

The `load_elf` function is the primary mechanism for getting compiled kernels onto the device. The `risc_name` parameter (`"brisc"`, `"trisc0"`, `"trisc1"`, `"trisc2"`) tells the loader which RISC core's memory region to target. On Wormhole, `load_elf` can return the `_start` symbol address via `return_start_address=True` -- this is critical for the start-address cache mechanism discussed in [Section 7.2](./02_step_by_step_execution.md).

### Debug and Control Functions

| Function | Purpose |
|----------|---------|
| `check_context()` | Returns the active tt-exalens context (connection to device) |
| `convert_coordinate(location, device_id, context)` | Converts string coordinates to `OnChipCoordinate` |
| `arc_msg(device_id, msg_code, wait_for_done, args, timeout)` | Send a message to the ARC processor |
| `callstack(location, elfs, risc_name, device_id)` | Extract the callstack from a halted RISC core |

---

## 7.1.2 Device Initialization and Context Setup

Before any test can run, the tt-exalens library must establish a connection to the hardware. This happens in `conftest.py` at pytest session start:

```python
# conftest.py -- pytest_configure()
if TestConfig.MODE != TestMode.PRODUCE:
    if test_target.run_simulator:
        # Simulator path -- see below
    else:
        tt_exalens_init.init_ttexalens(use_4B_mode=False)
```

The `init_ttexalens()` call performs PCI BAR mapping, enumerates devices, and creates the global `Context` object that all subsequent API calls reference. The `use_4B_mode=False` parameter selects the standard 8-byte addressing mode for the device's TLB configuration.

For simulator-based testing, the path differs:

```python
if test_target.run_simulator:
    _exalens_server = ExalensServer(
        simulator_path=simulator_path,
        port=test_target.simulator_port,
    )
    # Later, in pytest_runtest_setup:
    _exalens_server.start()
    tt_exalens_init.init_ttexalens_remote(
        port=test_target.simulator_port, use_4B_mode=False
    )
```

In both cases, the result is a global context accessible via `check_context()` that provides the device handle, coordinate system, and register stores. The context is the root object from which all device interactions flow:

```
check_context() --> Context
    .devices[0] --> Device
        .get_block(coordinate) --> NocBlock
            .get_register_store() --> RegisterStore
                .read_register("RISCV_DEBUG_REG_SOFT_RESET_0")
                .write_register(...)
```

The `Mailboxes` module-level variable in `device.py` starts as an `_UninitializedMailboxes` placeholder. Until `TestConfig.setup_build()` runs, any attempt to use mailbox addresses raises a clear error:

```python
class _UninitializedMailboxes:
    def __getattr__(self, name):
        raise RuntimeError(
            "Mailboxes have not been initialized. "
            "Ensure TestConfig.setup_build() has been called ..."
        )
```

---

## 7.1.3 Architecture Detection and Configuration

The LLK infrastructure must know which chip it is running on because Blackhole, Wormhole, and Quasar have different register layouts, RISC core numbering, boot modes, and compilation flags. Architecture detection is implemented in `chip_architecture.py`:

```python
def get_chip_architecture():
    global _cached_chip_architecture
    if _cached_chip_architecture is not None:
        return _cached_chip_architecture

    chip_architecture = os.getenv("CHIP_ARCH")
    if not chip_architecture:
        context = check_context()
        chip_architecture = str(context.devices[0]._arch)
        if chip_architecture == "wormhole_b0":
            chip_architecture = "wormhole"
        os.environ["CHIP_ARCH"] = chip_architecture

    _cached_chip_architecture = ChipArchitecture.from_string(chip_architecture)
    return _cached_chip_architecture
```

The detection has two paths:

1. **Environment variable** -- if `CHIP_ARCH` is set (to `"blackhole"`, `"wormhole"`, or `"quasar"`), use that directly. This is common in CI where the hardware is known.
2. **Hardware query** -- otherwise, query the tt-exalens context's device object for the architecture string and normalize `"wormhole_b0"` to `"wormhole"`.

The result is cached globally. `TestConfig.setup_arch()` then configures all downstream settings via a match/case block:

```python
# test_config.py -- setup_arch()
match TestConfig.CHIP_ARCH:
    case ChipArchitecture.BLACKHOLE:
        TestConfig.ARCH_NON_COMPUTE = "-mcpu=tt-bh"
        TestConfig.ARCH_COMPUTE = "-mcpu=tt-bh-tensix"
        TestConfig.ARCH_DEFINE = "-DARCH_BLACKHOLE"
        TestConfig.ARCH_LLK_ROOT = "tt_llk_blackhole"
        TestConfig.DATA_FORMAT_ENUM = BLACKHOLE_DATA_FORMAT_ENUM_VALUES
    case ChipArchitecture.WORMHOLE:
        TestConfig.ARCH_NON_COMPUTE = "-mcpu=tt-wh"
        TestConfig.ARCH_COMPUTE = "-mcpu=tt-wh-tensix"
        TestConfig.ARCH_DEFINE = "-DARCH_WORMHOLE"
        TestConfig.ARCH_LLK_ROOT = "tt_llk_wormhole_b0"
        TestConfig.DATA_FORMAT_ENUM = WORMHOLE_DATA_FORMAT_ENUM_VALUES
    case ChipArchitecture.QUASAR:
        TestConfig.ARCH_NON_COMPUTE = "-mcpu=tt-bh"
        TestConfig.ARCH_COMPUTE = "-mcpu=tt-bh-tensix"
        TestConfig.ARCH_DEFINE = "-DARCH_QUASAR"
        TestConfig.ARCH_LLK_ROOT = "tt_llk_quasar"
        TestConfig.DATA_FORMAT_ENUM = QUASAR_DATA_FORMAT_ENUM_VALUES
        TestConfig.KERNEL_COMPONENTS = ["unpack", "math", "pack", "sfpu"]
```

At the hardware level, `"-mcpu=tt-bh"` selects the Blackhole-specific RISC-V instruction set extensions for non-compute cores (BRISC), while `"-mcpu=tt-bh-tensix"` enables the Tensix compute extensions (SFPU instructions, Tensix intrinsics) for the TRISC kernel cores.

The architecture dictates every subsequent decision:

| Architecture | Default Boot Mode | RISC Cores | Non-Compute CPU Flag | Compute CPU Flag | Kernel Components |
|-------------|-------------------|------------|---------------------|-----------------|-------------------|
| Wormhole | BRISC | BRISC + TRISC0-2 | `-mcpu=tt-wh` | `-mcpu=tt-wh-tensix` | unpack, math, pack |
| Blackhole | BRISC | BRISC + TRISC0-2 | `-mcpu=tt-bh` | `-mcpu=tt-bh-tensix` | unpack, math, pack |
| Quasar | TRISC | TRISC0-3 (no BRISC) | `-mcpu=tt-bh` | `-mcpu=tt-bh-tensix` | unpack, math, pack, sfpu |

---

## 7.1.4 ARC Messages: Power State Management

The ARC processor on Tenstorrent chips manages power states, thermal throttling, and clock gating. Before any test execution can begin, the host must transition the chip to its active power state. This is done at pytest session boundaries:

```python
# conftest.py
def pytest_sessionstart(session):
    if hasattr(session.config, "workerinput"):
        return
    test_target = TestTargetConfig()
    if not test_target.run_simulator and not TestConfig.MODE == TestMode.PRODUCE:
        _send_arc_message("GO_BUSY", test_target.device_id)

def pytest_sessionfinish(session):
    if hasattr(session.config, "workerinput"):
        return
    test_target = TestTargetConfig()
    if not test_target.run_simulator and not TestConfig.MODE == TestMode.PRODUCE:
        _send_arc_message("GO_IDLE", test_target.device_id)
```

The `_send_arc_message()` helper in `device.py` constructs the ARC message:

```python
def _send_arc_message(message_type: str, device_id: int):
    ARC_COMMON_PREFIX = 0xAA00
    message_codes = {"GO_BUSY": 0x52, "GO_IDLE": 0x54}

    arc_msg(
        device_id=device_id,
        msg_code=ARC_COMMON_PREFIX | message_codes[message_type],
        wait_for_done=True,
        args=[0, 0],
        timeout=datetime.timedelta(seconds=10),
    )
```

| Message | Code | Full Value | Effect |
|---------|------|-----------|--------|
| `GO_BUSY` | `0x52` | `0xAA52` | Tells ARC to hold clocks active, disable power gating |
| `GO_IDLE` | `0x54` | `0xAA54` | Tells ARC it can re-enable clock gating and power savings |

The `wait_for_done=True` parameter makes this a synchronous call -- the host blocks until the ARC acknowledges the state transition, with a 10-second timeout. Without sending `GO_BUSY`, ARC might clock-gate the Tensix cores during kernel execution, causing hangs or corrupted results. Simulator runs skip ARC messaging because the simulator does not model the ARC power management subsystem.

---

## 7.1.5 Card Reset via tt-smi

The `HardwareController` class provides a last-resort recovery mechanism. If the chip enters a bad state (hung kernels, corrupted configuration registers), a full card reset is available:

```python
class HardwareController:
    def __init__(self):
        self.chip_architecture = get_chip_architecture()

    def reset_card(self):
        test_target = TestTargetConfig()
        if test_target.run_simulator:
            logger.info("Running under simulator, unable to reset")
            return
        if self.chip_architecture == ChipArchitecture.BLACKHOLE:
            run_shell_command("tt-smi -r")
        elif self.chip_architecture == ChipArchitecture.WORMHOLE:
            run_shell_command("tt-smi -r")
```

This shells out to `tt-smi -r`, Tenstorrent's System Management Interface tool, which performs a PCI-level device reset. A card reset asserts the PCIe reset signal, reinitializes all on-chip memory to undefined state, reboots the ARC processor, and re-enumerates the device on the PCIe bus. After a card reset, `init_ttexalens()` must be called again to re-establish the host-device connection. This is not part of normal test execution -- it is a recovery mechanism for manual intervention.

---

## 7.1.6 Register Access: Soft-Reset Control

The most critical register interaction in the LLK infrastructure is soft-reset control. Each RISC core on a Tensix tile has a soft-reset bit in the `RISCV_DEBUG_REG_SOFT_RESET_0` register. Setting the bit holds the core in reset; clearing it allows the core to execute from its `_start` address.

The `set_tensix_soft_reset()` function in `device.py` manipulates these bits:

```python
def set_tensix_soft_reset(value, cores: list[RiscCore] = ALL_CORES, location="0,0", device_id=0):
    soft_reset = get_register_store(location, device_id).read_register(
        "RISCV_DEBUG_REG_SOFT_RESET_0"
    )
    if value:
        soft_reset |= get_soft_reset_mask(cores)
    else:
        soft_reset &= ~get_soft_reset_mask(cores)

    get_register_store(location, device_id).write_register(
        "RISCV_DEBUG_REG_SOFT_RESET_0", soft_reset
    )
```

The `RiscCore` enum uses an overridden `.value` property that returns the architecture-dependent hardware bit position:

```python
class RiscCore(IntEnum):
    # Internal identifiers (0-4), but .value returns the hardware bit position
    BRISC = 0
    TRISC0 = 1
    TRISC1 = 2
    TRISC2 = 3
    TRISC3 = 4

    @property
    def value(self):
        arch = get_chip_architecture()
        is_quasar = arch == ChipArchitecture.QUASAR
        mapping = {
            RiscCore.BRISC:  -1 if is_quasar else 11,
            RiscCore.TRISC0: 11 if is_quasar else 12,
            RiscCore.TRISC1: 12 if is_quasar else 13,
            RiscCore.TRISC2: 13 if is_quasar else 14,
            RiscCore.TRISC3: 14 if is_quasar else -1,
        }
        return mapping[self]
```

The soft-reset mask is computed as `sum(1 << core.value for core in cores)`. For all four cores on Blackhole (BRISC=11, TRISC0=12, TRISC1=13, TRISC2=14), this produces `0x7800` (bits 11-14). The `TRISC_SOFT_RESET_MASK` constant in `boot.h` is `0x7000` (bits 12-14), which covers only the three TRISC cores -- allowing BRISC to control their reset state independently.

The hardware bit positions across architectures:

| Core | Wormhole/Blackhole Bit | Quasar Bit |
|------|------------------------|------------|
| BRISC | 11 | N/A (`-1`) |
| TRISC0 | 12 | 11 |
| TRISC1 | 13 | 12 |
| TRISC2 | 14 | 13 |
| TRISC3 | N/A (`-1`) | 14 |

The `get_register_store()` helper navigates the tt-exalens device hierarchy to reach the debug register file:

```python
def get_register_store(location="0,0", device_id=0, neo_id=0):
    context = check_context()
    device = context.devices[device_id]
    chip_coordinate = OnChipCoordinate.create(location, device=device)
    noc_block = device.get_block(chip_coordinate)
    register_store = noc_block.get_register_store()
    return register_store
```

> **Cross-reference:** [Section 7.2](./02_step_by_step_execution.md) traces the complete sequence of soft-reset manipulations during kernel launch, showing exactly when each core transitions from reset to running. [Chapter 3](../ch3_kernel_architecture/02_boot_sequence_and_firmware.md) covers the boot sequence from the firmware perspective.

---

## 7.1.7 The Host-Device Data Path

The fundamental data movement operations are `write_to_device` / `write_words_to_device` (host to device) and `read_from_device` / `read_word_from_device` (device to host). These operate on physical L1 addresses at a given core NOC coordinate.

The complete data path for a pure-LLK matmul test looks like this:

```
Host-Device Data Path for Pure-LLK Matmul
==========================================

 HOST (Python)                           DEVICE (Tensix L1 at core 0,0)
 =============                           =================================

 1. generate_stimuli()
    |  random torch tensors
    v
 2. tilize_block()
    |  row-major --> face layout
    v
 3. pack_fp16() / pack_bfp8_b()
    |  torch --> hardware bytes
    v
 4. write_to_device("0,0", addr, bytes)---->  L1[0x21000]: Input A tiles
    write_to_device("0,0", addr, bytes)---->  L1[A + sizeA]: Input B tiles
                                              L1[A + sizeA + sizeB]: (reserved for output)

 5. struct.pack(runtime_format, ...)
    |  config struct serialization
    v
 6. write_to_device("0,0", 0x20000, .)---->  L1[0x20000]: RuntimeParams struct

 7. load_elf("brisc.elf", risc="brisc")---->  BRISC instruction memory
    load_elf("unpack.elf", risc="trisc0")->  TRISC0 instruction memory
    load_elf("math.elf",   risc="trisc1")->  TRISC1 instruction memory
    load_elf("pack.elf",   risc="trisc2")->  TRISC2 instruction memory

 8. set_tensix_soft_reset(0) ------------->  Cores begin execution
                                              TRISC0: unpack A,B from L1 to SrcA/SrcB
                                              TRISC1: matmul in FPU, result in Dest
                                              TRISC2: pack from Dest to L1

 9. poll mailboxes until 0xFF  <-----------  Each TRISC writes 0xFF on completion

10. read_from_device("0,0", res_addr, n) <-  L1[res_addr]: packed output tiles
    |
    v
11. unpack_res_tiles()
    |  hardware bytes --> torch tensor
    v
12. compare with golden_tensor
```

Every byte written to L1, every ELF loaded, and every register access follows a single stack: Python test script -> tt-exalens Python API -> tt-exalens C++ backend (libttexalens) -> UMD (User Mode Driver) -> Tenstorrent kernel driver (PCIe) -> P150 hardware. There is no DMA, no network fabric setup, and no multi-chip coordination -- just direct PCIe-mapped memory access to a single Tensix core's L1.

---

## 7.1.8 Coordinate System and Multi-Core Addressing

Although LLK testing uses only a single Tensix tile at `"0,0"`, the infrastructure supports addressing any tile on the NOC grid. For parallel test execution (`pytest-xdist`), the `workers_tensix_coordinates` fixture in `conftest.py` assigns different tiles to different workers:

```python
@pytest.fixture()
def workers_tensix_coordinates(worker_id):
    if worker_id == "master":
        return "0,0"
    row, col = divmod(int(worker_id[2:]), 8)
    return f"{row},{col}"
```

This maps xdist worker IDs (like `gw0`, `gw1`, ...) to distinct NOC coordinates, enabling concurrent test execution on separate Tensix tiles without interference. Each test runs on exactly one Tensix core -- there is no multi-core orchestration, no NOC routing of tile data between cores, and no circular buffer management.

---

## 7.1.9 The TensixDebug Interface

For the alternative EXALENS boot mode (used in specialized debugging scenarios), the infrastructure accesses the Tensix tile through a richer debug interface:

```python
def exalens_device_setup(chip_arch, location="0,0", device_id=0):
    context = check_context()
    device = context.devices[device_id]
    chip_coordinate = OnChipCoordinate.create(location, device=device)
    debug_tensix = TensixDebug(chip_coordinate)
    ops = debug_tensix.device.instructions

    if chip_arch == ChipArchitecture.BLACKHOLE:
        get_register_store(location, device_id).write_register(
            "RISCV_DEBUG_REG_DEST_CG_CTRL", 0
        )
        debug_tensix.inject_instruction(ops.TT_OP_ZEROACC(3, 0, 0, 1, 0), 0)
    else:
        debug_tensix.inject_instruction(ops.TT_OP_ZEROACC(3, 0, 0), 0)

    debug_tensix.inject_instruction(ops.TT_OP_SFPENCC(3, 0, 0, 10), 0)
    debug_tensix.inject_instruction(ops.TT_OP_NOP(), 0)
    debug_tensix.inject_instruction(ops.TT_OP_SFPCONFIG(0, 11, 1), 0)
    debug_tensix.inject_instruction(ops.TT_OP_SEMINIT(1, 0, 2), 0)
    debug_tensix.inject_instruction(ops.TT_OP_SEMINIT(1, 0, 7), 0)
    debug_tensix.inject_instruction(ops.TT_OP_SEMINIT(1, 0, 4), 0)
```

The `TensixDebug.inject_instruction()` method injects Tensix ISA instructions directly into the tile's execution pipeline from the host -- without any RISC core running. This performs the same hardware initialization that `device_setup()` in `boot.h` does from BRISC firmware: clearing the accumulator, enabling the condition code stack, setting SFPU constant registers, and initializing semaphores. This mode is not used for typical matmul execution but demonstrates that the host has full instruction-level control over the Tensix tile through tt-exalens.

> **Cross-reference:** [Chapter 5, Section 3](../ch5_compile_time_config/03_data_format_inference_and_pipeline_configuration.md) documents the Tensix ISA instructions used in `device_setup()` and their role in matmul execution. [Chapter 3, Section 1](../ch3_kernel_architecture/01_single_source_three_kernels.md) covers the BRISC/TRISC architecture and why BRISC normally handles this initialization.

---

**Previous:** [Chapter 6 -- Linker Scripts and Memory Regions](../ch6_compilation/03_linker_scripts_and_memory_regions.md) | **Next:** [Step-by-Step Execution](./02_step_by_step_execution.md)
