# Writing Tests

This section covers the practical details of writing both functional and structural parts of an LLK test. It walks through the Python test function, the `TestConfig` and `StimuliConfig` objects, golden generation, validation, and the C++ kernel file structure.

## Python Test File

### Naming and Location

Place functional test files in `tests/python_tests/` with the naming pattern `test_<name>.py`. Performance tests use the pattern `perf_<name>.py` (covered in [performance_testing.md](./performance_testing.md)).

### Imports

A typical test file imports from several helper modules:

```python
from helpers.golden_generators import SomeGoldenGenerator, get_golden_generator
from helpers.llk_params import format_dict, DestAccumulation
from helpers.test_variant_parameters import TILE_COUNT, NUM_FACES, NUM_BLOCKS, NUM_TILES_IN_BLOCK
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.param_config import parametrize, input_output_formats
from helpers.utils import passed_test
```

### The @parametrize Decorator

The `@parametrize` decorator (from `tests/python_tests/helpers/param_config.py`) wraps `@pytest.mark.parametrize` and generates variant IDs. Parameters can be lists of values or lambda functions that compute valid values based on earlier parameters:

```python
@parametrize(
    formats=input_output_formats(
        [DataFormat.Float16, DataFormat.Float16_b, DataFormat.Bfp8_b]
    ),
    dest_acc=lambda formats: get_valid_dest_accumulation_modes(formats),
    tile_count=16,
    math_fidelity=lambda formats: get_valid_math_fidelities(formats),
)
def test_my_kernel(formats, dest_acc, tile_count, math_fidelity, workers_tensix_coordinates):
    ...
```

The `workers_tensix_coordinates` argument must always be present -- it is a pytest fixture that provides coordinates for the Tensix cores used for parallel test execution.

### Stimuli Generation

The `generate_stimuli()` function creates input tensors and returns them along with tile counts:

```python
src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
    stimuli_format_A=formats.input_format,
    input_dimensions_A=[32, 32],
    stimuli_format_B=formats.input_format,
    input_dimensions_B=[32, 32],
)
```

### Golden Generators

Golden generators compute the expected output using PyTorch. They are defined in `tests/python_tests/helpers/golden_generators.py`. Use `get_golden_generator()` to obtain a callable:

```python
generate_golden = get_golden_generator(DataCopyGolden)
golden_tensor = generate_golden(src_A, formats.output_format, num_faces, input_dimensions)
```

Different operations have different golden generator classes (e.g., `DataCopyGolden`, `TilizeGolden`, and various others for binary, reduce, and matmul operations).

## TestConfig

The `TestConfig` class (defined in `tests/python_tests/helpers/test_config.py`) is the central configuration object. Its constructor signature:

```python
TestConfig(
    test_name: str,                              # Path to C++ file relative to tests/
    formats: InputOutputFormat = None,           # Input/output data formats
    templates: list[TemplateParameter] = [],     # Compile-time constexpr parameters
    runtimes: list[RuntimeParameter] = [],       # Runtime struct fields
    variant_stimuli: StimuliConfig = None,       # Stimuli to load into L1
    boot_mode: BootMode = BootMode.DEFAULT,      # Kernel boot mode
    profiler_build: ProfilerBuild = ProfilerBuild.No,
    L1_to_L1_iterations: int = 1,
    unpack_to_dest: bool = False,                # Unpack directly to Dest register
    disable_format_inference: bool = False,       # Skip format inference
    dest_acc: DestAccumulation = DestAccumulation.No,  # Dest accumulation mode
    l1_acc: L1Accumulation = L1Accumulation.No,  # L1 accumulation (Quasar only)
    skip_build_header: bool = False,             # Skip build.h generation (fused tests)
    compile_time_formats: bool = False,          # Treat formats as compile-time args
)
```

Key arguments:

| Parameter | Description |
|-----------|-------------|
| `test_name` | Path to the `*.cpp` kernel file relative to `tests/`. This is the only mandatory argument. |
| `formats` | An `InputOutputFormat` dataclass (from `tests/python_tests/helpers/format_config.py`) containing input and output data formats. Drives format inference for unpack source/dest and pack source/dest register formats. |
| `templates` | List of `TemplateParameter` subclass instances. Each generates a `constexpr` definition in `build.h`. |
| `runtimes` | List of `RuntimeParameter` subclass instances. Each becomes a field in `struct RuntimeParams`. |
| `variant_stimuli` | A `StimuliConfig` object wrapping the input tensors and their format/tile metadata. |
| `unpack_to_dest` | When `True`, stimuli is unpacked directly to the Dest register instead of srcA/srcB. Sets the `unpack_to_dest` constexpr in `build.h`. |
| `dest_acc` | Controls the `is_fp32_dest_acc_en` constexpr in `build.h`. For some format combinations, dest accumulation is enabled implicitly by format inference. |
| `boot_mode` | How the kernel starts. Default depends on architecture: `BootMode.BRISC` for WH/BH, `BootMode.TRISC` for Quasar. |

### Template and Runtime Parameter Classes

Both parameter types are defined in `tests/python_tests/helpers/test_variant_parameters.py`. They inherit from abstract base classes:

```python
@dataclass
class TemplateParameter(ABC):
    @abstractmethod
    def convert_to_cpp(self) -> str:
        pass

@dataclass
class RuntimeParameter(ABC):
    @abstractmethod
    def convert_to_cpp(self) -> str:
        pass

    @abstractmethod
    def convert_to_struct_fields(self) -> tuple[str, str]:
        pass
```

Common parameter classes include `TILE_COUNT`, `NUM_FACES`, `NUM_BLOCKS`, `NUM_TILES_IN_BLOCK`, `DEST_INDEX`, `MATH_FIDELITY`, `MATH_OP`, and `TILIZE`. You instantiate them with a value:

```python
templates=[MATH_FIDELITY(MathFidelity.HiFi4), MATH_OP(mathop=MathOperation.Elwadd)]
runtimes=[TILE_COUNT(tile_cnt_A), NUM_FACES(4), NUM_BLOCKS(num_blocks), NUM_TILES_IN_BLOCK(num_tiles_in_block)]
```

To write a custom parameter, create a new `@dataclass` inheriting from `RuntimeParameter` or `TemplateParameter` and implement the required methods. For runtime parameters, `convert_to_struct_fields()` must return a tuple where the first element is the C++ struct field declaration (e.g., `"std::uint32_t MY_PARAM"`) and the second is a Python `struct` format character (e.g., `"I"` for unsigned 32-bit integer). See the [Python struct format reference](https://docs.python.org/3/library/struct.html#format-characters) for the full list.

## StimuliConfig

The `StimuliConfig` class (defined in `tests/python_tests/helpers/stimuli_config.py`) handles:

1. Computing L1 addresses for stimuli buffers (the base address is `StimuliConfig.STIMULI_L1_ADDRESS`).
2. Serializing (packing) tensor data into byte streams matching hardware data formats.
3. Writing byte streams to L1 before kernel execution.
4. Reading results from L1 after execution and deserializing them for validation.

Constructor signature:

```python
StimuliConfig(
    buffer_A,                                   # Input tensor A (from generate_stimuli)
    stimuli_A_format: DataFormat,               # Format of input A
    buffer_B,                                   # Input tensor B
    stimuli_B_format: DataFormat,               # Format of input B
    stimuli_res_format: DataFormat,             # Format of the result
    tile_count_A: int = 1,
    tile_count_B: int = None,
    tile_count_res: int = 1,
    buffer_C=None,                              # Optional third input
    stimuli_C_format: DataFormat = None,
    tile_count_C: int = None,
    num_faces: int = 4,
    face_r_dim: int = 16,
    tile_dimensions: list[int] = [32, 32],
    sfpu: bool = False,
    write_full_tiles: bool = False,
    use_dense_tile_dimensions: bool = False,
    operand_res_tile_size: int = None,
)
```

The `buffer_A`/`buffer_B` arguments accept the tensors returned by `generate_stimuli()`. For performance tests where actual stimuli data is not needed, these can be `None`.

## Validation with passed_test()

The `passed_test()` function (from `tests/python_tests/helpers/utils.py`) compares the kernel output against the golden tensor using `torch.isclose()` with format-specific tolerances:

| Format | atol | rtol |
|--------|------|------|
| Float16 | 0.05 | 0.05 |
| Float16_b | 0.05 | 0.05 |
| Float32 | 0.05 | 0.05 |
| Int32 | 0 | 0 |
| UInt32 | 0 | 0 |
| Int16 | 0 | 0 |
| UInt16 | 0 | 0 |
| Int8 | 0 | 0 |
| UInt8 | 0 | 0 |
| Bfp8_b | 0.1 | 0.2 |
| Bfp4_b | 0.25 | 0.3 |
| MxFp8R | 0.2 | 0.3 |
| MxFp8P | 0.2 | 0.3 |
| Fp8_e4m3 | 0.2 | 0.2 |

Integer formats use exact matching (zero tolerance). Floating-point formats use progressively looser tolerances as precision decreases. The tolerance table is defined in `tests/python_tests/helpers/utils.py`.

A typical validation sequence:

```python
res_from_L1 = configuration.run(workers_tensix_coordinates).result

assert len(res_from_L1) == len(golden_tensor)

res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

assert passed_test(golden_tensor, res_tensor, formats.output_format)
```

## C++ Kernel File Structure

### Mandatory Includes and Globals

Every kernel file must include `ckernel.h` and `llk_defs.h` at the top level (outside the `#ifdef` guards) and declare the three global variables used by the LLK runtime:

```cpp
#include <cstdint>
#include "ckernel.h"
#include "llk_defs.h"

// Globals -- required by the LLK API runtime
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;
```

### Three run_kernel Functions

Each `#ifdef` section must define `void run_kernel(RUNTIME_PARAMETERS params)`. The `RUNTIME_PARAMETERS` macro expands to `const struct RuntimeParams&`, giving the kernel access to all runtime parameters serialized by Python.

Within each section, include `params.h` (which pulls in `build.h`) and whatever LLK API headers are needed for that thread's work:

- **Unpack section** (`#ifdef LLK_TRISC_UNPACK`): Typically includes `llk_unpack_*.h` and `llk_unpack_common.h`. Responsible for configuring hardware unpack and moving data from L1 into source registers.
- **Math section** (`#ifdef LLK_TRISC_MATH`): Typically includes `llk_math_common.h` and the operation-specific math header (e.g., `llk_math_eltwise_binary.h`). Performs computation on data in source/dest registers.
- **Pack section** (`#ifdef LLK_TRISC_PACK`): Typically includes `llk_pack.h` and `llk_pack_common.h`. Moves results from dest registers back to L1 memory.

### Accessing Runtime Parameters

Inside `run_kernel`, access runtime values through the `params` struct. For format configuration when formats are runtime parameters:

```cpp
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
```

For custom runtime parameters:

```cpp
const std::uint32_t tile_count = params.TILE_CNT;
const std::uint8_t face_r_dim = static_cast<std::uint8_t>(params.TEST_FACE_R_DIM);
```

The field names in `params` correspond exactly to the struct field names generated by each `RuntimeParameter` subclass's `convert_to_struct_fields()` method.

### Accessing Template Parameters

Template parameters are `constexpr` variables defined in `build.h` and are available directly as global constants:

```cpp
// These are constexpr variables from build.h, used as template arguments:
_llk_unpack_hw_configure_<is_fp32_dest_acc_en>(...);
_llk_math_eltwise_binary_<MATH_OP, BroadcastType::NONE, MATH_FIDELITY, is_fp32_dest_acc_en>(...);
```

## Complete Example

Here is a simplified but complete functional test:

**Python** (`tests/python_tests/test_magnificent.py`):

```python
from helpers.golden_generators import DataCopyGolden, get_golden_generator
from helpers.llk_params import format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import TILE_COUNT, NUM_FACES
from helpers.utils import passed_test
import torch

@parametrize(
    formats=input_output_formats([DataFormat.Float16, DataFormat.Float16_b]),
    num_faces=[4],
    input_dimensions=[[32, 32]],
)
def test_magnificent(formats, num_faces, input_dimensions, workers_tensix_coordinates):
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
    )

    generate_golden = get_golden_generator(DataCopyGolden)
    golden_tensor = generate_golden(src_A, formats.output_format, num_faces, input_dimensions)

    configuration = TestConfig(
        "sources/magnificent_test.cpp",
        formats,
        templates=[],
        runtimes=[TILE_COUNT(tile_cnt_A), NUM_FACES(num_faces)],
        variant_stimuli=StimuliConfig(
            src_A, formats.input_format,
            src_B, formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
            num_faces=num_faces,
        ),
    )

    res_from_L1 = configuration.run(workers_tensix_coordinates).result
    assert len(res_from_L1) == len(golden_tensor)
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    assert passed_test(golden_tensor, res_tensor, formats.output_format)
```

---

**Next:** [`running_and_debugging.md`](./running_and_debugging.md)
