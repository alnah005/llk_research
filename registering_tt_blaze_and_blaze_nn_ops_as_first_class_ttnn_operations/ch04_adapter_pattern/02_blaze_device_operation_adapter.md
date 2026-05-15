# 4.2 BlazeDeviceOperationAdapter -- The Complete C++ Design

This section presents the full design of `BlazeDeviceOperationAdapter<OpTag>`, the C++ class template that bridges Blaze's pre-built `MeshProgramDescriptor` to TTNN's `DeviceOperationConcept`. Every type alias, every required method, and every optional extension is shown with its implementation rationale traced back to the translation challenges from [Section 3.2](../ch03_compilation_model_gap/02_translation_challenges.md). The adapter reuses `GenericMeshProgramFactory` through a thin factory wrapper (`BlazeAdapterMeshProgramFactory`) -- zero new factory logic is written. Per-op identity comes from the `OpTag` template parameter, a minimal struct that carries a name string and optional SFINAE-detected hooks for progressive extensibility.

---

## 4.2.1 The OpTag Struct Pattern

Each Blaze op is identified by a tag struct that provides compile-time identity:

```cpp
// PROPOSED: Minimal OpTag -- sufficient for Tier 1 registration
// Source: (new) ttnn/cpp/ttnn/operations/blaze/blaze_op_tags.hpp

namespace ttnn::operations::blaze {

struct MatmulTag {
    static constexpr const char* name = "blaze::matmul";
};

struct RMSNormTag {
    static constexpr const char* name = "blaze::rmsnorm";
};

struct SDPATag {
    static constexpr const char* name = "blaze::sdpa";
};

struct GatedMLPTag {
    static constexpr const char* name = "blaze::gated_mlp";
};

// ... one tag per registered Blaze op

}  // namespace ttnn::operations::blaze
```

The tag has no data members, no virtual functions, and no runtime cost. It exists solely to differentiate template instantiations at the type level.

| Constraint | Rationale |
|------------|-----------|
| Must be a distinct C++ type | `register_operation<>` enforces type uniqueness via `operation_key_t`. Two ops sharing the same `OpTag` produce a compile error. |
| Must have `static constexpr const char* name` | Used for registration strings, profiler zone names, graph trace node names, error messages, and cache hash seeding. |
| Must be stateless (no data members) | The tag is used only as a template parameter; instances are never stored. |
| Should follow `blaze::{op_name}` naming | The `blaze::` prefix avoids collision with native TTNN op names (e.g., `"blaze::matmul"` vs `"matmul"`). |

### Extended OpTag (for Tier 2/3 Ops)

For ops that need custom validation or hash hooks, the tag can carry additional static members detected by the adapter via SFINAE:

```cpp
// PROPOSED: Extended OpTag for high-value ops (Tier 3)
struct MatmulTagExtended {
    static constexpr const char* name = "blaze::matmul";

    // Optional: custom validation hook (detected via SFINAE)
    static void validate(
        const tt::tt_metal::experimental::MeshProgramDescriptor& desc,
        const std::vector<Tensor>& io_tensors) {
        TT_FATAL(io_tensors.size() >= 3,
            "Matmul requires at least 2 inputs + 1 output, got {}",
            io_tensors.size());
    }

    // Optional: custom hash hook (detected via SFINAE)
    static std::optional<tt::stl::hash::hash_t> custom_hash(
        const tt::tt_metal::experimental::MeshProgramDescriptor& desc,
        const std::vector<Tensor>& io_tensors) {
        if (!desc.mesh_programs.empty()) {
            const auto& pd = desc.mesh_programs.front().second;
            if (pd.custom_program_hash) {
                return *pd.custom_program_hash;
            }
        }
        return std::nullopt;  // fall back to descriptor walk
    }
};
```

SFINAE detection traits enable compile-time hook discovery:

```cpp
// PROPOSED: SFINAE detection for OpTag extensions
template <typename Tag, typename = void>
struct HasValidateHook : std::false_type {};

template <typename Tag>
struct HasValidateHook<Tag, std::void_t<decltype(Tag::validate(
    std::declval<const tt::tt_metal::experimental::MeshProgramDescriptor&>(),
    std::declval<const std::vector<Tensor>&>()))>> : std::true_type {};

template <typename Tag, typename = void>
struct HasCustomHashHook : std::false_type {};

template <typename Tag>
struct HasCustomHashHook<Tag, std::void_t<decltype(Tag::custom_hash(
    std::declval<const tt::tt_metal::experimental::MeshProgramDescriptor&>(),
    std::declval<const std::vector<Tensor>&>()))>> : std::true_type {};
```

The tag is a **progressive extension point**: start with a minimal 2-line struct, add hooks only when profiling or debugging demands it. No template specialization in separate files is needed.

---

## 4.2.2 Shared External Types

The adapter's `operation_attributes_t` and `tensor_args_t` are defined as **shared types outside the template**, ensuring all adapter instantiations share a single type definition. This avoids creating 112 distinct but structurally identical types and enables the factory wrapper to operate uniformly.

```cpp
// PROPOSED: ttnn/cpp/ttnn/operations/blaze/blaze_device_operation_adapter.hpp

#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <variant>
#include <optional>
#include <tuple>
#include <unordered_set>

#include "ttnn/api/ttnn/operation_concepts.hpp"
#include "ttnn/api/ttnn/device_operation.hpp"
#include "ttnn/cpp/ttnn/operations/generic/device/generic_op_program_factory.hpp"
#include "ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.hpp"
#include "tt_metal/api/tt-metalium/experimental/mesh_program_descriptor.hpp"

namespace ttnn::operations::blaze {

// ============================================================
// BlazeAdapterAttributes -- shared operation_attributes_t
//
// Wraps MeshProgramDescriptor with op identity metadata.
// Defined OUTSIDE the template so that all adapter
// instantiations share a single type, enabling factory
// compatibility and avoiding 112 distinct attribute types.
// ============================================================
struct BlazeAdapterAttributes {
    tt::tt_metal::experimental::MeshProgramDescriptor mesh_program_descriptor;
    std::string op_name;  // e.g., "blaze::matmul" -- from OpTag::name

    // Optional: user-supplied program hash for cache optimization (Tier 2).
    // When present, compute_program_hash() uses this instead of walking
    // the full descriptor. Passed as an explicit invoke() parameter
    // rather than mutating the MeshProgramDescriptor.
    std::optional<tt::stl::hash::hash_t> custom_program_hash;

    // Reflection support for Inspector and profiler attribute serialization
    static constexpr auto attribute_names =
        std::forward_as_tuple("op_name", "num_mesh_programs");
    auto attribute_values() const {
        return std::make_tuple(
            std::string_view(op_name),
            mesh_program_descriptor.mesh_programs.size());
    }
};

// ============================================================
// BlazeAdapterTensorArgs -- shared tensor_args_t
//
// Mirrors GenericOpDeviceOperation::tensor_args_t structure.
// The output tensor is always the last element of io_tensors,
// pre-allocated by the Blaze compiler.
// ============================================================
struct BlazeAdapterTensorArgs {
    const std::vector<Tensor>& io_tensors;
    const Tensor& output_tensor;  // alias for io_tensors.back()
};
```

> **Reference-lifetime guarantee.** `BlazeAdapterTensorArgs` stores `io_tensors` and `output_tensor` by reference, not by value. This is safe because the TTNN dispatch pipeline guarantees that `invoke()` parameters outlive the entire `launch<>` call -- the caller's storage is not destroyed until after `enqueue_mesh_workload` completes. `GenericOpDeviceOperation::tensor_args_t` relies on the same reference-lifetime guarantee, so this is a documented framework constraint rather than a design flaw.

---

## 4.2.3 The Complete Adapter Template

```cpp
// ============================================================
// BlazeDeviceOperationAdapter<OpTag>
//
// A DeviceOperationConcept-conforming wrapper that gives each
// Blaze op a unique C++ type identity while reusing
// GenericMeshProgramFactory for program construction.
// ============================================================

template <typename OpTag>
struct BlazeDeviceOperationAdapter {

    // ========================================================================
    // Type Aliases 1-2: shared external types
    // ========================================================================
    using operation_attributes_t = BlazeAdapterAttributes;
    using tensor_args_t          = BlazeAdapterTensorArgs;

    // ========================================================================
    // Type Aliases 3-4: return types
    // ========================================================================
    using spec_return_value_t    = TensorSpec;
    using tensor_return_value_t  = Tensor;

    // ========================================================================
    // Factory Wrapper: BlazeAdapterMeshProgramFactory
    //
    // Bridges the type mismatch between the adapter's
    // operation_attributes_t (BlazeAdapterAttributes, which CONTAINS
    // a MeshProgramDescriptor) and GenericMeshProgramFactory, which
    // expects the operation_attributes_t to BE a MeshProgramDescriptor.
    //
    // The wrapper extracts mesh_program_descriptor from attrs and
    // delegates to GenericMeshProgramFactory. This adds no per-op code
    // and introduces no behavioral changes -- it is purely a type-level
    // adapter for the factory interface.
    // ========================================================================
    struct BlazeAdapterMeshProgramFactory {
        using mesh_shared_variables_t =
            ttnn::operations::generic::program::GenericMeshProgramFactory
                ::mesh_shared_variables_t;
        using cached_mesh_workload_t =
            ttnn::device_operation::CachedMeshWorkload<mesh_shared_variables_t>;

        static cached_mesh_workload_t create_mesh_workload(
                const operation_attributes_t& attrs,
                const ttnn::MeshCoordinateRangeSet& tensor_coords,
                const tensor_args_t& tensor_args,
                tensor_return_value_t& tensor_return_value) {
            // Build GenericOpDeviceOperation-compatible arguments
            typename ttnn::operations::generic::GenericOpDeviceOperation
                ::tensor_args_t generic_tensor_args{
                .io_tensors = tensor_args.io_tensors,
                .output_tensor = tensor_args.output_tensor,
            };

            // Extract MeshProgramDescriptor and delegate
            return ttnn::operations::generic::program
                ::GenericMeshProgramFactory::create_mesh_workload(
                    attrs.mesh_program_descriptor,
                    tensor_coords,
                    generic_tensor_args,
                    tensor_return_value);
        }

        static void override_runtime_arguments(
                cached_mesh_workload_t& cached_workload,
                const operation_attributes_t& attrs,
                const tensor_args_t& tensor_args,
                tensor_return_value_t& tensor_return_value) {
            typename ttnn::operations::generic::GenericOpDeviceOperation
                ::tensor_args_t generic_tensor_args{
                .io_tensors = tensor_args.io_tensors,
                .output_tensor = tensor_args.output_tensor,
            };

            ttnn::operations::generic::program
                ::GenericMeshProgramFactory::override_runtime_arguments(
                    cached_workload,
                    attrs.mesh_program_descriptor,
                    generic_tensor_args,
                    tensor_return_value);
        }
    };

    // ========================================================================
    // Type Alias 5: program_factory_t
    //
    // Uses the adapter's factory wrapper as the sole variant.
    // Single-alternative variant means no select_program_factory()
    // is needed -- the framework auto-selects the sole alternative.
    // ========================================================================
    using program_factory_t = std::variant<BlazeAdapterMeshProgramFactory>;

    // ========================================================================
    // Required Method 1: validate_on_program_cache_miss
    //
    // Validates descriptor structure (no duplicate mesh coordinate
    // ranges). Optionally calls OpTag::validate if the tag provides
    // a validation hook (detected at compile time via SFINAE).
    // ========================================================================
    static void validate_on_program_cache_miss(
            const operation_attributes_t& attrs,
            const tensor_args_t& tensor_args) {
        // Base validation: verify no duplicate mesh coordinate ranges
        const auto& mesh_programs = attrs.mesh_program_descriptor.mesh_programs;
        std::unordered_set<tt::tt_metal::distributed::MeshCoordinateRange> seen;
        seen.reserve(mesh_programs.size());
        for (const auto& [range, pd] : mesh_programs) {
            auto [it, inserted] = seen.insert(range);
            TT_FATAL(inserted,
                "BlazeDeviceOperationAdapter<{}>::validate: "
                "duplicate MeshCoordinateRange in descriptor",
                OpTag::name);
        }

        // Validate tensor requirements
        TT_FATAL(!tensor_args.io_tensors.empty(),
            "{}: io_tensors must not be empty", OpTag::name);

        // Extended validation: OpTag hook (detected at compile time)
        if constexpr (HasValidateHook<OpTag>::value) {
            OpTag::validate(
                attrs.mesh_program_descriptor,
                tensor_args.io_tensors);
        }
    }

    // ========================================================================
    // Required Method 2: compute_output_specs
    //
    // Returns the spec of the pre-allocated output tensor. This
    // preserves Blaze's pre-allocation model (Challenge 3 from
    // Section 3.2) while enabling graph tracing in NO_DISPATCH mode
    // to introspect output shape without execution.
    // ========================================================================
    static spec_return_value_t compute_output_specs(
            const operation_attributes_t& attrs,
            const tensor_args_t& tensor_args) {
        return tensor_args.output_tensor.tensor_spec();
    }

    // ========================================================================
    // Required Method 3: create_output_tensors
    //
    // Returns the pre-allocated output tensor unchanged. Blaze
    // allocates outputs before compilation because CB descriptors
    // reference tensor Buffer* pointers (Challenge 3).
    // ========================================================================
    static tensor_return_value_t create_output_tensors(
            const operation_attributes_t& attrs,
            const tensor_args_t& tensor_args) {
        return tensor_args.output_tensor;
    }

    // ========================================================================
    // Optional Method: compute_program_hash
    //
    // Three-tier hash cascade with explicit precedence:
    //
    //   +---------------------------------------------------------+
    //   | HASH TIER PRECEDENCE                                    |
    //   +---------------------------------------------------------+
    //   | Priority 1: OpTag::custom_hash() hook (Tier 3 ops)      |
    //   |   Checked via SFINAE. When the tag provides a           |
    //   |   custom_hash() static method, it takes highest         |
    //   |   priority. This is used for per-op semantic hashing    |
    //   |   defined in C++ (e.g., hash M,K,N for matmul).        |
    //   |                                                         |
    //   | Priority 2: custom_program_hash field (Tier 2 ops)      |
    //   |   An explicit hash passed from Python via invoke().     |
    //   |   This is the recommended Tier 2 path: Blaze computes   |
    //   |   a semantic hash in Python and passes it through.      |
    //   |                                                         |
    //   | Priority 3: Full descriptor walk (Tier 1 default)       |
    //   |   Hashes the MeshProgramDescriptor field-by-field.      |
    //   |   Cost: O(n) in descriptor complexity, ~2-5 us.         |
    //   +---------------------------------------------------------+
    //
    // All tiers are prefixed with type_hash<BlazeDeviceOperationAdapter<OpTag>>
    // to guarantee per-op cache namespace isolation.
    // ========================================================================
    static tt::stl::hash::hash_t compute_program_hash(
            const operation_attributes_t& attrs,
            const tensor_args_t& tensor_args) {

        constexpr auto type_seed =
            tt::stl::hash::type_hash<BlazeDeviceOperationAdapter<OpTag>>;

        // Priority 1: OpTag custom hash hook (Tier 3)
        if constexpr (HasCustomHashHook<OpTag>::value) {
            auto custom = OpTag::custom_hash(
                attrs.mesh_program_descriptor,
                tensor_args.io_tensors);
            if (custom.has_value()) {
                return tt::stl::hash::hash_objects_with_default_seed(
                    type_seed, *custom);
            }
        }

        // Priority 2: Explicit custom_program_hash from Python (Tier 2)
        //
        // Hash Source Precedence Note:
        //   - BlazeAdapterAttributes.custom_program_hash (set via explicit invoke
        //     parameter) takes Tier 2 priority -- it is checked here.
        //   - ProgramDescriptor.custom_program_hash (per-descriptor field) is
        //     checked in the Tier 1 fallback's per-program hash loop, inside
        //     compute_program_descriptor_hash().
        //   - If both are set, the attrs-level hash wins (Tier 2 > Tier 1),
        //     because this branch returns before the descriptor walk executes.
        //   - The two paths serve different use cases: the attrs-level hash is
        //     for ops that compute a single overall semantic hash in Python,
        //     while the descriptor-level hash provides per-device hash control
        //     within the Tier 1 descriptor walk.
        //
        if (attrs.custom_program_hash.has_value()) {
            return tt::stl::hash::hash_objects_with_default_seed(
                type_seed, attrs.custom_program_hash.value());
        }

        // Priority 3: Full descriptor walk (Tier 1 default)
        // Iterate mesh_programs and hash each ProgramDescriptor
        auto hash = type_seed;
        for (const auto& [mesh_coord_range, program_descriptor] :
                 attrs.mesh_program_descriptor.mesh_programs) {
            tt::stl::hash::hash_combine(hash, mesh_coord_range);
            // If ProgramDescriptor.custom_program_hash is set, use it;
            // otherwise hash the descriptor field-by-field.
            tt::stl::hash::hash_combine(hash,
                ttnn::operations::generic::compute_program_descriptor_hash(
                    program_descriptor));
        }
        return hash;
    }

    // ========================================================================
    // Required: invoke()
    //
    // Converts user-facing arguments to (attrs, tensor_args).
    // Called by registered_operation_t::operator().
    //
    // Accepts optional custom_program_hash parameter, passed
    // explicitly from Python rather than embedded in the descriptor.
    // This is architecturally cleaner than mutating the
    // MeshProgramDescriptor before the call.
    // ========================================================================
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
            const std::vector<Tensor>& io_tensors,
            const tt::tt_metal::experimental::MeshProgramDescriptor&
                mesh_program_descriptor,
            std::optional<tt::stl::hash::hash_t> custom_program_hash
                = std::nullopt) {
        TT_FATAL(
            io_tensors.size() >= 2,
            "{}: io_tensors must contain at least 1 input + 1 output, got {}",
            OpTag::name, io_tensors.size());

        return {
            operation_attributes_t{
                .mesh_program_descriptor = mesh_program_descriptor,
                .op_name = std::string(OpTag::name),
                .custom_program_hash = custom_program_hash},
            tensor_args_t{
                .io_tensors = io_tensors,
                .output_tensor = io_tensors.back()}
        };
    }
};

}  // namespace ttnn::operations::blaze
```

---

## 4.2.4 Concept Satisfaction Proof

The following table maps every `DeviceOperationConcept` requirement (from [Section 1.1](../ch01_ttnn_registration_contract/01_device_operation_concept.md)) to the adapter's implementation:

### Required Concept Requirements

| Concept Requirement | Adapter Implementation | Status |
|---|---|:-:|
| `operation_attributes_t` type alias | `BlazeAdapterAttributes` (shared external struct) | Satisfied |
| `tensor_args_t` type alias | `BlazeAdapterTensorArgs` (shared external struct) | Satisfied |
| `spec_return_value_t` type alias | `TensorSpec` | Satisfied |
| `tensor_return_value_t` type alias | `Tensor` | Satisfied |
| `program_factory_t` type alias (valid variant) | `std::variant<BlazeAdapterMeshProgramFactory>` | Satisfied |
| `validate_on_program_cache_miss(attrs, tensor_args)` | Duplicate-range check + optional `OpTag::validate` hook | Satisfied |
| `compute_output_specs(attrs, tensor_args) -> TensorSpec` | Returns `tensor_args.output_tensor.tensor_spec()` | Satisfied |
| `create_output_tensors(attrs, tensor_args) -> Tensor` | Returns pre-allocated `tensor_args.output_tensor` | Satisfied |
| `AllFactoriesValid<program_factory_t>` | `BlazeAdapterMeshProgramFactory` satisfies `MeshWorkloadFactoryConcept` | Satisfied |

### Optional Concept Extensions

| Optional Concept | Satisfied? | Rationale |
|---|:-:|---|
| `DeviceOperationWithCustomProgramCacheConcept` | **Yes** | `compute_program_hash()` with type-isolated three-tier cascade |
| `HasValidateOnProgramCacheHit` | **No** (deliberate) | Blaze's validation runs in Python before the descriptor reaches C++. Lightweight cache-hit validation adds cost without catching new bugs in the generic case. Can be added per-op via OpTag hooks if needed. |
| `HasSelectProgramFactory` | **No** (not needed) | Single-alternative variant; framework auto-selects the sole factory. |
| `HasSkipLaunch` | **No** (deliberate) | Blaze programs always execute. Short-circuiting would require semantic knowledge the adapter does not have. |

### Compile-Time Verification

The following `static_assert` can be placed alongside the template to verify concept conformance:

```cpp
// PROPOSED: Concept verification (in a test or at namespace scope)
namespace ttnn::operations::blaze {
struct TestTag { static constexpr const char* name = "blaze::test"; };
static_assert(
    ttnn::device_operation::DeviceOperationConcept<
        BlazeDeviceOperationAdapter<TestTag>>,
    "BlazeDeviceOperationAdapter must satisfy DeviceOperationConcept");
static_assert(
    ttnn::device_operation::DeviceOperationWithCustomProgramCacheConcept<
        BlazeDeviceOperationAdapter<TestTag>>,
    "BlazeDeviceOperationAdapter must satisfy custom program cache concept");
}  // namespace ttnn::operations::blaze
```

---

## 4.2.5 BlazeAdapterMeshProgramFactory -- Factory Bridge

`BlazeAdapterMeshProgramFactory` is defined inline in the adapter template (Section 4.2.3, lines 214--260). It extracts `attrs.mesh_program_descriptor` and delegates to `GenericMeshProgramFactory`, bridging the type mismatch without modifying existing factory interfaces.

### Why Not Modify GenericMeshProgramFactory?

Three alternatives were considered:

| Alternative | Why Not |
|-------------|---------|
| Modify `GenericMeshProgramFactory` to accept `BlazeAdapterAttributes` | Changes the factory's public interface; risks breaking `GenericOpDeviceOperation` |
| Make `BlazeAdapterAttributes` implicitly convertible to `MeshProgramDescriptor` | Implicit conversions are error-prone; violates TTNN coding conventions |
| Templatize `GenericMeshProgramFactory` on attribute type | Increases factory complexity for all users; template bloat |

The wrapper approach is preferred because it is explicit, localized (inside the adapter), and does not modify any existing code.

---

## 4.2.6 Registration Syntax

### Minimal Registration

```cpp
// PROPOSED: Per-op registration
// Source: (new) ttnn/cpp/ttnn/operations/blaze/blaze_op_registrations.hpp

#include "blaze_device_operation_adapter.hpp"
#include "blaze_op_tags.hpp"  // Tags defined in Section 4.2.1
#include "ttnn/api/ttnn/decorators.hpp"

namespace ttnn::blaze {

constexpr auto matmul    = register_operation<"ttnn::blaze::matmul",
    operations::blaze::BlazeDeviceOperationAdapter<operations::blaze::MatmulTag>>();
constexpr auto copy      = register_operation<"ttnn::blaze::copy",
    operations::blaze::BlazeDeviceOperationAdapter<operations::blaze::CopyTag>>();
constexpr auto reduce    = register_operation<"ttnn::blaze::reduce",
    operations::blaze::BlazeDeviceOperationAdapter<operations::blaze::ReduceTag>>();
constexpr auto rmsnorm   = register_operation<"ttnn::blaze::rmsnorm",
    operations::blaze::BlazeDeviceOperationAdapter<operations::blaze::RMSNormTag>>();
constexpr auto sdpa      = register_operation<"ttnn::blaze::sdpa",
    operations::blaze::BlazeDeviceOperationAdapter<operations::blaze::SDPATag>>();
constexpr auto gated_mlp = register_operation<"ttnn::blaze::gated_mlp",
    operations::blaze::BlazeDeviceOperationAdapter<operations::blaze::GatedMLPTag>>();

}  // namespace ttnn::blaze
```

### X-Macro Convenience

For 112+ ops, an X-macro pattern generates both tags and registrations from a single list:

```cpp
// PROPOSED: ttnn/cpp/ttnn/operations/blaze/blaze_op_list.hpp
// Single source of truth -- add new ops here only
#define BLAZE_OP_LIST(X)    \
    X(matmul, Matmul)       \
    X(copy, Copy)           \
    X(reduce, Reduce)       \
    X(mcast, Mcast)         \
    X(gather, Gather)       \
    X(transpose, Transpose) \
    X(topk, TopK)           \
    X(rmsnorm, RMSNorm)     \
    X(sdpa, SDPA)           \
    X(gated_mlp, GatedMLP)  \
    X(rotary_embedding, RotaryEmbedding)  \
    X(fused_attention, FusedAttention)
    // ... add more here ...

// Generate tag structs
#define DEFINE_BLAZE_TAG(op_name, tag_name) \
    struct tag_name##Tag { \
        static constexpr const char* name = "blaze::" #op_name; \
    };

namespace ttnn::operations::blaze {
BLAZE_OP_LIST(DEFINE_BLAZE_TAG)
}  // namespace ttnn::operations::blaze

// Generate registrations
#define REGISTER_BLAZE_OP_FROM_LIST(op_name, tag_name)            \
    constexpr auto op_name = ::ttnn::register_operation<          \
        "ttnn::blaze::" #op_name,                                 \
        ::ttnn::operations::blaze::BlazeDeviceOperationAdapter<   \
            ::ttnn::operations::blaze::tag_name##Tag>>();

namespace ttnn::blaze {
BLAZE_OP_LIST(REGISTER_BLAZE_OP_FROM_LIST)
}  // namespace ttnn::blaze
```

Note that the string literal concatenation `"ttnn::blaze::" #op_name` in the `REGISTER_BLAZE_OP_FROM_LIST` macro is valid as a `FixedString` non-type template parameter (NTTP). C++ preprocessor string literal concatenation (e.g., `"ttnn::blaze::" "matmul"` producing `"ttnn::blaze::matmul"`) occurs at translation phase 6, before template argument deduction, so the result is a single string literal that the compiler can use directly to construct a `FixedString` NTTP.

Each X-macro invocation generates both the tag struct and the `constexpr` registration. Adding a new op is a single line in `BLAZE_OP_LIST`. The nanobind binding macro (Section 4.3) reuses the same list.

---

## 4.2.7 End-to-End Data Flow Diagram

The following diagram traces a Blaze op invocation from Python through the adapter into TTNN's dispatch pipeline, showing MISS and HIT paths:

```
Python: Blaze Compilation Pipeline
=======================================
  BlazeOp.emit() / FusedOp.compose()
       |
       v
  FusedProgram (stateful accumulator)
       |
  CBEngine + SemEngine + CTArgEngine       <-- Challenges 1, 2, 7
       |
       v
  kernel_codegen.py                         <-- Challenge 5
       |
       v
  ProgramDescriptor assembly
       |
       v
  MeshProgramDescriptor (complete)
       |
       |  _run_program(io_tensors, descriptor, op_name="matmul")
       v
  ttnn.blaze.matmul(io_tensors, descriptor)        <-- NEW dispatch path
       |
=======|===========================================
       |  C++ / nanobind boundary
       v
Step 1: registered_operation_t<"ttnn::blaze::matmul",
            BlazeDeviceOperationAdapter<MatmulTag>>::operator()
       |
Step 2: BlazeDeviceOperationAdapter<MatmulTag>::invoke(io_tensors, descriptor)
       |   -> returns (BlazeAdapterAttributes{descriptor, "blaze::matmul"},
       |               BlazeAdapterTensorArgs{io_tensors, io_tensors.back()})
       v
Step 3: device_operation::launch<BlazeDeviceOperationAdapter<MatmulTag>>(...)
       |
Step 4: GraphTracker::track_function_start("ttnn::blaze::matmul", ...)
       |                                    ^^^^^^^^^^^^^^^^^^^^
       |                                    NAMED (was "ttnn::generic_op")
       |
Step 5: create_output_tensors(attrs, args)
       |   -> returns args.output_tensor (pre-allocated)    <-- Challenge 3
       |
Step 6: compute_program_hash(attrs, args)
       |   -> type_hash<BlazeDeviceOperationAdapter<MatmulTag>>   <-- UNIQUE
       |      + descriptor hash (or custom_program_hash)
       |
Step 7: program_cache.contains(hash)?
       |
       +-------- MISS (first invocation) ---------+
       |                                           |
Step 8a: validate_on_program_cache_miss(...)       |
       |   -> verify no duplicate ranges           |
       |   -> OpTag::validate() [if present]       |
       |                                           |
Step 9a: BlazeAdapterMeshProgramFactory            |
       |   ::create_mesh_workload(...)             |
       |   -> extract attrs.mesh_program_descriptor|
       |   -> delegate to GenericMeshProgramFactory |
       |      -> for each (range, pd):             |
       |            Program{pd}   [ProgramBuilder] |
       |   -> return CachedMeshWorkload            |
       |                                           |
Step 10a: program_cache.insert(hash, ...)          |
       |                                           |
       +-------- HIT (subsequent) ----------------+
       |                                           |
Step 8b: BlazeAdapterMeshProgramFactory            |
       |   ::override_runtime_arguments(...)       |
       |   -> patch Buffer* addrs + RT args        |
       |                                           |
       +-------------------------------------------+
       |
Step 11: enqueue_mesh_workload(...)
       |   -> TracyOpMeshWorkload("ttnn::blaze::matmul")   <-- NAMED zone
       |
Step 12: GraphTracker::track_function_end(output_tensor)
       |
       v
  Return output tensor to Python
```

**What changed vs. the `generic_op` path ([Chapter 2](../ch02_generic_op_dispatch/index.md)):**

| Pipeline Stage | `generic_op` (Before) | Adapter (After) |
|---|---|---|
| Operation name in profiler | `"ttnn::generic_op"` | `"ttnn::blaze::matmul"` |
| Operation name in graph trace | `"GenericOpDeviceOperation"` | `"BlazeDeviceOperationAdapter<MatmulTag>"` |
| Operation name in error messages | `"generic_op: ..."` | `"blaze::matmul: ..."` |
| Program cache namespace | `type_hash<GenericOpDeviceOperation>` (shared) | `type_hash<Adapter<MatmulTag>>` (per-op) |
| Program factory | `GenericMeshProgramFactory` (direct) | `BlazeAdapterMeshProgramFactory` (wrapper -> same factory) |
| Program construction | `ProgramBuilder` from descriptor | `ProgramBuilder` from descriptor (identical) |
| Hardware execution | Identical | Identical |
| Output tensor management | Pre-allocated passthrough | Pre-allocated passthrough (identical) |

The adapter changes only the **identity layer** of the dispatch pipeline. The **execution layer** is bit-identical.

---

## 4.2.8 Challenge Resolution Table

| # | Challenge ([Section 3.2](../ch03_compilation_model_gap/02_translation_challenges.md)) | Adapter Resolution | Component |
|---|---|---|---|
| 1 | **Language boundary**: TTNN expects C++ factories; Blaze logic is in Python | Adapter accepts pre-built `MeshProgramDescriptor` from Python via `operation_attributes_t`. Factory never calls back into Python. | `BlazeAdapterAttributes`, `invoke()` |
| 2 | **Stateful vs stateless**: `FusedProgram` accumulates state; TTNN factories are stateless | `operation_attributes_t` wraps the already-built descriptor -- the immutable snapshot of accumulated state. | `BlazeAdapterAttributes.mesh_program_descriptor` |
| 3 | **Output tensor management**: Blaze pre-allocates; TTNN expects `create_output_tensors()` to allocate | `create_output_tensors()` returns the pre-allocated tensor unchanged. `compute_output_specs()` reads its spec. | `create_output_tensors()`, `compute_output_specs()` |
| 4 | **Multi-phase programs**: FusedOps produce single multi-phase kernels | Registers at FusedOp granularity. The entire multi-phase `ProgramDescriptor` is one factory output. TTNN sees one operation. | One `OpTag` per FusedOp; single factory variant |
| 5 | **Dynamic kernel sources**: `kernel_codegen.py` generates C++ at compile time | Factory passes kernel paths through to `ProgramBuilder` via the descriptor. `compute_program_hash()` captures kernel identity via the structural hash. | `BlazeAdapterMeshProgramFactory` delegates |
| 6 | **Per-core specialization**: `PerCoreCompileTimeDescriptor` varies CT args per core | `ProgramBuilder` iterates all `KernelDescriptor` entries and creates one kernel per unique CT arg configuration. Inherited behavior, unchanged. | `BlazeAdapterMeshProgramFactory` delegates |
| 7 | **Engine pipeline**: CBEngine/SemEngine/CTArgEngine have no C++ equivalent | Engines continue to run in Python. Their output is embedded in the `MeshProgramDescriptor` consumed by the adapter. | `BlazeAdapterAttributes.mesh_program_descriptor` |
| 8 | **100+ ops scaling** | Template parameterized by `OpTag`. Per-op cost: one tag struct (~2 lines) + one registration (~3 lines). Total for 112 ops: ~600 lines. | `BlazeDeviceOperationAdapter<OpTag>` + X-macro |

Every challenge is resolved by accepting Blaze's pre-built output rather than trying to reproduce Blaze's internal logic in C++. The adapter is a *post-compilation wrapper*, not a reimplementation.

---

## 4.2.9 What the Adapter Does NOT Do

To maintain clarity about the adapter's scope, the following capabilities are explicitly *not* part of the adapter design:

| Capability | Why Not | Where It Lives Instead |
|------------|---------|----------------------|
| Program construction from semantic attributes | Would require embedding Blaze's `emit()`/`compose()` in C++ | Blaze's Python compilation pipeline |
| True output shape inference from input shapes | Would require embedding Blaze's shape logic in C++ | `compute_output_specs()` returns pre-allocated tensor's spec |
| CB/semaphore/CT-arg optimization | Would require porting the engine pipeline to C++ | `CBEngine`, `SemEngine`, `CTArgEngine` in Python |
| Kernel source generation | Would require porting `kernel_codegen.py` to C++ | Blaze's JIT compilation pipeline |
| Multiple factory variants (e.g., broadcast vs non-broadcast) | Single factory handles all cases via `ProgramDescriptor` dispatch | Blaze selects the right program structure during compilation |
| TTNN-managed output allocation | Would invalidate `Buffer*` addresses in the `ProgramDescriptor` | Future architectural change (see [Chapter 8](../ch08_build_migration_alternatives/index.md)) |
| Performance model integration | Requires typed semantic attributes | Per-op wrappers (Tier 3) for high-value ops |

The adapter's power comes from its restraint: by accepting pre-built descriptors and injecting only identity metadata, it avoids the ~53,000 lines of C++ that a full-port approach would require.

---

## 4.2.10 Build Impact

Each `BlazeDeviceOperationAdapter<OpTag>` instantiation triggers template expansion of `device_operation::launch<>` and `MeshDeviceOperationAdapter`. For 112 ops:

| Build Artifact | Per-Instantiation | 112 Instantiations |
|----------------|------------------:|-------------------:|
| `launch<>` expansion | ~2--5 KB object code | ~220--560 KB |
| `MeshDeviceOperationAdapter` wrapper | ~1--3 KB | ~110--340 KB |
| `BlazeAdapterMeshProgramFactory` visitor | ~0.5--1 KB | ~56--112 KB |
| **Total incremental object code** | | **~400 KB -- 1 MB** |

Mitigation strategies:

1. **Explicit template instantiation** in a single `.cpp` file prevents redundant instantiation across translation units.
2. **Unity build** combines all registrations into one compilation unit.
3. **LTO (link-time optimization)** merges identical method bodies across instantiations.

The incremental cost (~400 KB -- 1 MB, ~1 second compile time) is modest relative to TTNN's existing binary size.

---

## Key Takeaways

- **`BlazeDeviceOperationAdapter<OpTag>` is a single C++ class template** that satisfies all `DeviceOperationConcept` requirements through delegation, not reimplementation. Every method either delegates to `GenericOpDeviceOperation` logic or returns a passthrough value. The adapter is a metadata wrapper, not a program builder.

- **`OpTag` is a progressive extension point.** Start with a 2-line struct carrying only a name. Add `validate()` or `custom_hash()` hooks only when profiling or debugging demands it. The adapter detects these at compile time via SFINAE.

- **`BlazeAdapterMeshProgramFactory` bridges the type mismatch** between the adapter's `BlazeAdapterAttributes` and `GenericMeshProgramFactory`'s expected `MeshProgramDescriptor`. It extracts the descriptor and delegates, adding zero new factory logic.

- **The three-tier `compute_program_hash` cascade** (OpTag hook -> `custom_program_hash` field -> descriptor walk) supports all three tiers of the hybrid strategy. The `type_hash` prefix guarantees per-op cache isolation regardless of which tier is active.

- **Every translation challenge from [Chapter 3](../ch03_compilation_model_gap/index.md) is addressed** by a specific adapter design decision. No challenge requires per-op C++ code for the generic case.

- **The adapter does not change what runs on hardware.** The same `ProgramDescriptor` reaches the same `ProgramBuilder`, producing the same `Program` objects. The only change is how the program reaches `enqueue_mesh_workload()` -- through a named, typed dispatch path instead of the anonymous `generic_op` path.

---

**Next:** [`03_python_dispatch_and_registration.md`](./03_python_dispatch_and_registration.md)
