# Agent B Review: Chapter 1 — Pass 1

1. **File:** `02_registration_dispatch_and_bindings.md`, lines 424-443 (Section 1.2.6, End-to-End Flow Summary). **Error:** The flow diagram for `ttnn.add(a, b)` shows the framework decomposing `BinaryOperation<ADD>::invoke(a, b)` into `(operation_attributes_t, tensor_args_t)` and then the framework calling `device_operation::launch<BinaryDeviceOperation>(attrs, tensor_args)`. This is Branch 1 behavior (the device-operation path in `registered_operation_t::invoke`). However, Section 1.2.2 (line 102) explicitly states that `BinaryOperation<ADD>` is a **composite** operation, which takes Branch 2 — its `invoke` is called directly and the composite itself is responsible for calling `device_operation::launch<>` internally. The diagram conflates these two paths: it shows the framework performing the attribute decomposition and launch dispatch, when in reality the composite's `invoke` method does this internally. A reader implementing a composite operation would incorrectly believe their `invoke` only needs to return `(operation_attributes_t, tensor_args_t)` and the framework handles the rest, when they actually need to call `device_operation::launch<>` themselves. **Fix:** Either (a) change the example to use a pure device operation (not a composite wrapper) so the flow diagram accurately reflects Branch 1, or (b) re-draw the diagram to show that `BinaryOperation<ADD>::invoke` internally constructs attributes/tensor_args and calls `device_operation::launch<BinaryDeviceOperation>`, making clear this is the composite doing the work, not the framework.

---

# Agent B Review: Chapter 1 — Pass 2

**Pass 1 fix verification:** The end-to-end flow diagram in Section 1.2.6 (lines 424-450) now correctly labels `BinaryOperation<ADD>` as a composite (Branch 2), shows its `invoke` being called directly by the framework, and nests the attribute construction and `device_operation::launch<BinaryDeviceOperation>` call inside the composite's invoke with the comment "Inside the composite's invoke -- NOT the framework." This accurately represents the two-branch dispatch and resolves the Pass 1 issue.

**No feedback — chapter approved.**

---

# Agent B Review: Chapter 1 — Pass 3

**No feedback — chapter approved.**
