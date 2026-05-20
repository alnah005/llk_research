# CPU Affinity

The decode scheduler pins each of its three threads to a distinct CPU core. This is not optional polish -- on a multi-socket server running dozens of other processes, uncontrolled thread migration causes cache-line bouncing between L2 caches, latency spikes in the pipeline injection loop, and unpredictable scheduling jitter. The affinity system provides three modes: automatic assignment, explicit per-thread pinning, and opt-out.

## SchedulerParams CPU Fields

From `decode_types.hpp`:

```cpp
struct SchedulerParams {
    // ...
    static constexpr int AUTO_CPU = -1;
    static constexpr int NOPIN_CPU = -2;
    int writer_cpu = AUTO_CPU;
    int reader_cpu = AUTO_CPU;
    int api_cpu    = AUTO_CPU;
};
```

| Value | Meaning |
|---|---|
| `AUTO_CPU` (-1) | Automatically select a core from the process's available set |
| `NOPIN_CPU` (-2) | Do not pin this thread; let the OS scheduler decide |
| >= 0 | Pin to this specific core ID |

The default is `AUTO_CPU` for all three threads, which means the scheduler will auto-select distinct cores without any user intervention.

## resolve_cpu_assignments()

This static function maps `AUTO_CPU` values to concrete core IDs. It runs once during `start()`, before threads are spawned.

### Step 1: Check Whether Resolution Is Needed

If none of the three CPU values is `AUTO_CPU`, the function returns immediately. Explicit values (including `NOPIN_CPU`) pass through unchanged.

### Step 2: Discover Available Cores

```cpp
static std::vector<int> get_available_cpus() {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) != 0) return {};
    std::vector<int> cpus;
    for (int i = 0; i < CPU_SETSIZE; i++) {
        if (CPU_ISSET(i, &mask)) cpus.push_back(i);
    }
    return cpus;
}
```

`sched_getaffinity(0, ...)` queries the calling process's CPU affinity mask. This respects any external restrictions (cgroups, taskset, Docker `--cpuset-cpus`). The result is a sorted list of available core IDs.

### Step 3: Remove Explicitly Claimed Cores

If one thread has an explicit core assignment (e.g., `writer_cpu = 4`), that core is removed from the available set before AUTO_CPU resolution. This prevents two threads from being pinned to the same core.

```cpp
auto remove_used = [&](int cpu) {
    if (cpu >= 0) {
        available.erase(
            std::remove(available.begin(), available.end(), cpu),
            available.end());
    }
};
remove_used(writer);
remove_used(reader);
remove_used(api);
```

### Step 4: Stride-Based Distribution

With `N` available cores, the function spaces the three picks by `stride = max(N/3, 1)`:

```cpp
size_t n = available.size();
size_t stride = std::max<size_t>(n / 3, 1);
size_t idx = 0;
auto pick = [&]() -> int {
    if (available.empty()) return NOPIN_CPU;
    int cpu = available[idx % n];
    idx += stride;
    return cpu;
};

if (writer == AUTO_CPU) writer = pick();
if (reader == AUTO_CPU) reader = pick();
if (api    == AUTO_CPU) api    = pick();
```

The stride computation uses modular indexing (`available[idx % n]`), so it wraps around defensively. The resolution order is Writer first, then Reader, then API.

### Example: 24-Core System

With 24 available cores (0..23) and all three threads on AUTO:
- `stride = 24 / 3 = 8`
- Writer gets core 0
- Reader gets core 8
- API gets core 16

This places threads 8 cores apart, which on typical dual-socket or multi-CCX topologies means they are likely on different NUMA nodes or cache domains.

### Example: Mixed Explicit and AUTO

If `writer_cpu = 4` (explicit), `reader_cpu = AUTO_CPU`, `api_cpu = AUTO_CPU`:
1. Available set: {0,1,2,3,4,5,...,23}
2. Remove core 4: {0,1,2,3,5,6,...,23} (23 remaining)
3. stride = 23/3 = 7
4. Reader picks available[0] = 0
5. API picks available[7] = 8

### Fallback

If the available set is empty (which can happen if `sched_getaffinity` fails), AUTO_CPU resolves to `NOPIN_CPU` -- the thread runs unpinned.

## pin_to_cpu()

After all three threads are spawned, `start()` pins each to its resolved core:

```cpp
static void pin_to_cpu(std::thread& t, int cpu, const char* name) {
    if (cpu < 0) return;  // NOPIN_CPU or unresolved -> skip
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    int rc = pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::cerr << "WARNING: failed to pin " << name
                  << " to CPU " << cpu << " (errno=" << rc << ")" << std::endl;
    }
}
```

Key details:
- Uses `pthread_setaffinity_np` with the thread's native handle, because the C++ standard library does not expose affinity control.
- The affinity mask contains exactly one bit, restricting the thread to a single core.
- Failure is non-fatal -- a warning is printed, and the thread runs unpinned. This allows graceful degradation in containers with restricted CPU sets.
- Pinning happens after thread creation (`std::thread` constructor). This means a thread may execute a few iterations on an unpinned core before migration. This is acceptable because the pin happens once at startup, and the thread's first useful work occurs after the `running` flag is observed, which synchronizes with `start()`.

## Production Tuning Guidance

### Dedicated Core Isolation

For lowest-latency deployments, isolate the scheduler's three cores from the OS scheduler: boot with `isolcpus=X,Y,Z`, set `writer_cpu`, `reader_cpu`, `api_cpu` explicitly in `SchedulerParams` to match, and ensure no other threads are pinned to these cores. This prevents OS timer ticks and IRQ processing from preempting the scheduler threads.

### NUMA Awareness

On multi-socket systems, pin all three threads to the same NUMA node as the Tenstorrent accelerator's PCIe device. Cross-NUMA memory access adds latency to every atomic operation. The AUTO_CPU stride heuristic is not NUMA-aware -- use explicit core assignments on production systems.

### Avoid Hyperthreading Siblings

The stride-based auto-assignment may place two threads on hyperthreading siblings. Both the Writer and Reader spin continuously, so sharing a physical core halves effective throughput. Use `lscpu --extended` or `/sys/devices/system/cpu/cpu*/topology/core_id` to identify siblings and assign distinct physical cores. The API thread is less latency-sensitive and can share a physical core with a non-scheduler thread if cores are scarce.

### Container Deployments

Ensure at least 3 CPUs are available to the container. AUTO_CPU respects the container's cgroup affinity mask automatically. With fewer than 3 CPUs, threads share cores (functional but suboptimal) -- consider `NOPIN_CPU` in this case.

### Disabling Pinning Entirely

Set all three CPU fields to `NOPIN_CPU` for development, testing, or environments where the OS scheduler is trusted:

```cpp
SchedulerParams params;
params.writer_cpu = SchedulerParams::NOPIN_CPU;
params.reader_cpu = SchedulerParams::NOPIN_CPU;
params.api_cpu    = SchedulerParams::NOPIN_CPU;
```

### Monitoring

Verify pinning after startup by checking `/proc/<pid>/task/<tid>/status` for the `Cpus_allowed_list` mask. Each scheduler thread should show a single allowed CPU matching its configured assignment.

---

**Previous:** [Threading Model](./threading_model.md) | **Next:** [FreeIdPool](./free_id_pool.md)
