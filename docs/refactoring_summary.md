# Nebula GPU/CUDA Refactoring Summary

**Date**: 2026-02-20
**Status**: Complete (all phases implemented, uncommitted)

---

## Overview

Refactoring of the Nebula Hesai CUDA-accelerated decoder implementation across 3 repositories. The code delivers a 3.86x speedup but grew incrementally over 8 commits and needed cleanup for elegance, readability, and reviewability.

**Key improvements:**
- 404-line function with 12+ nesting levels → flat allocation with helper lambdas
- 236-line monolithic flush function → 4 focused helpers + orchestrator
- Unnecessary GPU-to-CPU config copy per scan → eliminated (host reference)
- Misleading "ring buffer" naming → corrected to "batch buffer"
- Always-on profiling instrumentation → gated behind `NEBULA_CUDA_PROFILING` flag
- Memory leak in destructor → fixed

---

## Phase 1: Refactor `initialize_cuda()` + Fix Memory Leak

**File:** `nebula_hesai_decoders/.../decoders/hesai_decoder.hpp`

- Added `GpuScanBuffer::cleanup()` method that frees all device and pinned-host memory with null checks and nullptr reset.
- Flattened **per-packet buffer allocation** (6 nested if/else blocks → flat `alloc_per_packet_ok` lambda with sequential calls).
- Flattened **scan buffer allocation** (~190 lines of 12-level nested if/else → ~40 lines with `alloc_scan_ok` lambda). Same semantics: on failure, all prior allocations are freed and `use_scan_batching_` is set to false.
- **Fixed destructor memory leak**: Added `gpu_scan_buffer_.cleanup()` call. The destructor was missing cleanup of 5 device + 6 pinned-host scan buffer allocations.

---

## Phase 2: Refactor `flush_gpu_scan_buffer()`

**File:** `hesai_decoder.hpp`

- Extracted `build_batch_config(n_entries)` → returns `CudaDecoderConfig` populated with sensor config, angle data, and multi-frame handling (~60 lines).
- Extracted `transfer_scan_to_device(n_entries, total_data_size, config)` → 5 `cudaMemcpyAsync` calls for bulk H2D transfer.
- Extracted `process_gpu_results(n_entries, valid_point_count, sparse_buffer_size)` → D2H sparse buffer copy + CPU-side compaction into NebulaPoint pointcloud.
- `flush_gpu_scan_buffer()` reduced from **236 lines to ~50 lines** calling the helpers in sequence with profiling event recording between steps.

---

## Phase 3: Eliminate Unnecessary D2H Config Copy

**Files:** `hesai_decoder.hpp`, `hesai_cuda_kernels.cu`, `hesai_cuda_decoder.hpp`

**Problem:** Both `launch_decode_hesai_scan_batch` and `launch_decode_hesai_packet` accepted a device pointer to the config, then immediately copied it back to host via `cudaMemcpyAsync` + `cudaStreamSynchronize` to read grid-sizing parameters. The caller already had these values on the host.

**Fix:**
- Changed both launcher signatures from `const CudaDecoderConfig* d_config` (device pointer) to `const CudaDecoderConfig& config` (host reference).
- Removed `cudaMemcpyAsync` + `cudaStreamSynchronize` from both launchers — eliminates a synchronous D2H copy + stream sync per scan.
- Removed the now-unused `d_config_` device member variable, its `cudaMalloc`, and destructor cleanup.
- Updated all call sites and extern declarations across 3 files.

---

## Phase 4: Naming Corrections

**Files:** `hesai_decoder.hpp`, `hesai_cuda_kernels.cu`, `hesai_cuda_decoder.hpp`

- Renamed `d_distances_ring` → `d_distances_batch` and `d_reflectivities_ring` → `d_reflectivities_batch` across all source files and documentation.
- Updated struct comment: "Ring buffers" → "Batch buffers (filled linearly, reset after flush)".

These buffers are linear staging areas filled from index 0 and reset to 0 after each flush — not ring buffers.

---

## Phase 5: Conditional Profiling Instrumentation

**Files:** `hesai_decoder.hpp`, `CMakeLists.txt`

- Added `NEBULA_CUDA_PROFILING` CMake option (default OFF), nested inside `NEBULA_CUDA_ENABLED` guard.
- Wrapped all timing member declarations (`cudaEvent_t`, `GpuTimingStats`) in `#ifdef NEBULA_CUDA_PROFILING`.
- Wrapped all event recording calls, elapsed time computation, log output, and PROFILING JSON stderr output in `#ifdef NEBULA_CUDA_PROFILING`.
- Wrapped timing event initialization in `initialize_cuda()` and cleanup in the destructor.
- **Production builds now have zero profiling overhead.** Profiling is opt-in via `-DNEBULA_CUDA_PROFILING=ON`.

### Build commands

```bash
# Production build (no profiling overhead)
colcon build --symlink-install --packages-up-to nebula_hesai \
    --cmake-args -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=89

# Benchmark build (with profiling instrumentation)
colcon build --symlink-install --packages-up-to nebula_hesai \
    --cmake-args -DCMAKE_BUILD_TYPE=Release \
    -DNEBULA_CUDA_PROFILING=ON \
    -DCMAKE_CUDA_ARCHITECTURES=89
```

---

## Phase 6: cuda_blackboard Cleanup

**Files:** `cuda_blackboard_subscriber.hpp`, `cuda_blackboard_subscriber.cpp`

- Extracted duplicate timer initialization from both constructor overloads into private `init_negotiation_retry()` method.
- Replaced magic numbers with named constants:
  - `NEGOTIATION_RETRY_INTERVAL_MS = 500`
  - `NEGOTIATION_MAX_RETRIES = 30`
- Replaced string concatenation in retry logging loop with `std::ostringstream`.

---

## Phase 8: Update Benchmark Documentation

**Files:** `BENCHMARK_PLAN.md`, `BENCHMARK_REPORT.md`, `docs/cuda_implementation_summary.md`

- Updated build commands to distinguish production (no profiling) vs benchmark (`-DNEBULA_CUDA_PROFILING=ON`) builds.
- Fixed all "ring buffer" references → "batch buffer" in documentation.
- Added **CUDA Kernel Branching Analysis** section documenting why branches in the GPU kernel are intentional:
  - **Uniform branches** (`config.is_multi_frame`, `n_returns == 2`): zero divergence cost.
  - **Early-exit branches** (`raw_distance == 0`, range/FOV checks): standard GPU optimization.
  - **Data-dependent branches** (overlap, dual-return filtering): affect small fraction of threads, unavoidable for correctness.

---

## Files Modified

### nebula (`src/sensor_component/external/nebula`)

| File | Changes |
|------|---------|
| `src/nebula_hesai/nebula_hesai_decoders/include/nebula_hesai_decoders/decoders/hesai_decoder.hpp` | Phases 1-5: flatten allocations, fix leak, extract helpers, eliminate D2H config copy, rename _ring→_batch, conditional profiling |
| `src/nebula_hesai/nebula_hesai_decoders/src/cuda/hesai_cuda_kernels.cu` | Phases 3-4: config by reference, rename _ring→_batch |
| `src/nebula_hesai/nebula_hesai_decoders/include/nebula_hesai_decoders/cuda/hesai_cuda_decoder.hpp` | Phases 3-4: updated extern declarations |
| `src/nebula_hesai/nebula_hesai_decoders/CMakeLists.txt` | Phase 5: NEBULA_CUDA_PROFILING option |
| `BENCHMARK_REPORT.md` | Phase 8: updated build commands |
| `docs/cuda_implementation_summary.md` | Phase 8: batch buffer naming, build commands, branching analysis |

### cuda_blackboard (`src/universe/external/cuda_blackboard`)

| File | Changes |
|------|---------|
| `include/cuda_blackboard/cuda_blackboard_subscriber.hpp` | Phase 6: constants, init_negotiation_retry() declaration |
| `src/cuda_blackboard_subscriber.cpp` | Phase 6: extract duplicate init, ostringstream, use constants |

### autoware root

| File | Changes |
|------|---------|
| `BENCHMARK_PLAN.md` | Phase 8: updated build commands |

---

## Design Constraints Preserved

- **`BUILD_CUDA` CMake option**: When OFF, no CUDA code is compiled. Pure CPU build.
- **`NEBULA_CUDA_ENABLED` compile definition**: All CUDA code remains inside `#ifdef` guards.
- **`cuda_enabled_` runtime flag**: Falls back to CPU path if CUDA initialization fails.
- **`gpu_pipeline_mode` runtime config**: Controls zero-copy vs host-copy behavior.
- **`NEBULA_CUDA_PROFILING`**: Nested inside `NEBULA_CUDA_ENABLED` (profiling only makes sense when CUDA is enabled).
