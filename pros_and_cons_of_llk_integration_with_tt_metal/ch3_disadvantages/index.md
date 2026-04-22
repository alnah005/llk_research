# Chapter 3: Disadvantages and Pain Points

This chapter documents concrete problems, friction, and risks observed in the current LLK integration with Metal. Each pain point is grounded in specific code evidence and assessed for severity.

## Pain Point Summary

| Pain Point | Severity | File |
|---|---|---|
| Include path explosion across three sources of truth | **High** | [build_system_complexity.md](./build_system_complexity.md) |
| SFPI toolchain management complexity | **Medium** | [build_system_complexity.md](./build_system_complexity.md) |
| Submodule update friction | **Medium** | [coupling_and_synchronization.md](./coupling_and_synchronization.md) |
| Implicit `_llk_*` API contracts | **High** | [coupling_and_synchronization.md](./coupling_and_synchronization.md) |
| `#ifdef TRISC_*` conditional compilation fragility | **High** | [coupling_and_synchronization.md](./coupling_and_synchronization.md) |
| Global mutable state in LLK tests | **Low** | [coupling_and_synchronization.md](./coupling_and_synchronization.md) |
| Deep include chains for kernel authors | **Medium** | [developer_ergonomics.md](./developer_ergonomics.md) |
| Macro-heavy kernel authoring via build-injected defines | **Medium** | [developer_ergonomics.md](./developer_ergonomics.md) |
| Legacy vs simplified kernel syntax coexistence | **Medium** | [developer_ergonomics.md](./developer_ergonomics.md) |
| Error messages from deep template instantiation | **Low** | [developer_ergonomics.md](./developer_ergonomics.md) |

## Severity Criteria

- **High**: Actively causes bugs, blocks refactoring, or creates maintenance burden across multiple teams.
- **Medium**: Creates friction and slows development but has established workarounds.
- **Low**: Annoyance or technical debt that does not regularly block work.

---

**Next:** [`build_system_complexity.md`](./build_system_complexity.md)
