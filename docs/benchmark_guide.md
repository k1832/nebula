# Benchmark Guide: CPU vs GPU Decode Latency

**Sensor**: Pandar128E4X (OT128), ~72,000 points/scan

---

## Overview

This guide measures the latency of decoding a single LiDAR scan: the **CPU baseline** vs the **CUDA GPU implementation**.

| Measurement | nebula branch | cuda_blackboard | Decode Flow |
|---|---|---|---|
| **CPU Baseline** | `benchmark/cpu-baseline` | (autoware.repos default) | `unpack()` → `convert_returns()` per packet, accumulated across scan |
| **GPU** | `feat/cuda-refactor` | `feat/cuda-refactor` | `unpack()` → accumulate packets → `flush_gpu_scan_buffer()` (H2D → kernel → D2H) |

**Fair comparison**: CPU accumulated `unpack()` time across all packets in a scan vs GPU `flush_gpu_scan_buffer()` total time — both represent the full work to produce one complete scan.

### Branches

| Branch | Repo | Description |
|---|---|---|
| `benchmark/cpu-baseline` | nebula | Mainline code + CPU profiling instrumentation only |
| `feat/cuda-refactor` | nebula | CUDA GPU decoder with refactored code |
| `feat/cuda-refactor` | cuda_blackboard | Cleanup for GPU zero-copy pipeline |

### Test Data

Default rosbag used by the benchmark runner:

```
~/autoware/src/sensor_component/external/nebula/src/nebula_hesai/nebula_hesai/test_resources/decoder_ground_truth/ot128/1730271167765338806
```

---

## Measuring the CPU Baseline

### 1. Reset repos and switch to CPU benchmark branch

```bash
cd ~/autoware

# Sync all repos to autoware.repos versions
vcs import src < repositories/autoware.repos

# Install/update system dependencies (ansible roles)
./setup-dev-env.sh

# Switch nebula to the CPU benchmark branch
cd ~/autoware/src/sensor_component/external/nebula
git checkout benchmark/cpu-baseline
```

The `benchmark/cpu-baseline` branch is based on the mainline release tag (`v0.3.2.2`) and adds only CPU profiling instrumentation: accumulated decode time per scan + `PROFILING` JSON output to stderr.

### 2. Build

```bash
cd ~/autoware
colcon build --symlink-install --packages-up-to nebula_hesai \
    --cmake-args -DCMAKE_BUILD_TYPE=Release
```

### 3. Run

```bash
source ~/autoware/install/setup.bash
cd ~/autoware/src/sensor_component/external/nebula

# Quick manual check — verify PROFILING lines appear
ros2 launch nebula_hesai nebula_launch.py \
    sensor_model:=Pandar128E4X launch_hw:=false 2>&1 | grep PROFILING &
ros2 bag play -l <rosbag_path>
# Expected: PROFILING {"d_cpu_unpack_ms": 3.5, "n_points": 72000}
```

Using the benchmark runner:

```bash
./scripts/benchmark_runner.bash --cpu -n 3 -t 20
```

### 4. Analyze

```bash
python3 scripts/analyze_benchmark.py \
    --log-dir ./benchmark_output_cpu \
    --output cpu_results.json
```

---

## Measuring the GPU

### 1. Switch to GPU branches

```bash
cd ~/autoware/src/sensor_component/external/nebula
git checkout feat/cuda-refactor

cd ~/autoware/src/universe/external/cuda_blackboard
git checkout feat/cuda-refactor
```

### 2. Build with CUDA and profiling

```bash
cd ~/autoware
colcon build --symlink-install --packages-up-to nebula_hesai \
    --cmake-args -DCMAKE_BUILD_TYPE=Release \
    -DNEBULA_CUDA_PROFILING=ON \
    -DCMAKE_CUDA_ARCHITECTURES=89
```

> Adjust `CMAKE_CUDA_ARCHITECTURES` for your GPU: `75` (Turing), `86` (Ampere), `87` (Jetson Orin), `89` (Ada).
> The CMakeLists.txt defaults to `"75;86;87;89"` if not specified, but targeting your exact architecture produces faster builds.

### 3. Run

```bash
source ~/autoware/install/setup.bash
cd ~/autoware/src/sensor_component/external/nebula

./scripts/benchmark_runner.bash --gpu -n 3 -t 20
```

Expected output:

```
PROFILING {"d_gpu_h2d_ms": 0.5, "d_gpu_kernel_ms": 0.25, "d_gpu_d2h_ms": 0.17, "d_gpu_total_ms": 0.92, "n_points": 72000}
```

### 4. Analyze

```bash
python3 scripts/analyze_benchmark.py \
    --log-dir ./benchmark_output_gpu \
    --output gpu_results.json
```

---

## Comparing Results

```bash
python3 scripts/analyze_benchmark.py --compare \
    --cpu-results cpu_results.json \
    --gpu-results gpu_results.json
```

### Metrics

| Metric | Formula | Description |
|---|---|---|
| **CPU Total** | `d_cpu_unpack_ms` (accumulated) | Total CPU decode time per scan |
| **GPU Total** | `d_gpu_h2d_ms + d_gpu_kernel_ms + d_gpu_d2h_ms` | Full GPU pipeline |
| **Effective GPU (Zero-Copy)** | `GPU Total - d_gpu_d2h_ms` | Without D2H transfer |
| **Speedup** | `CPU Total / GPU Total` | Performance improvement |
| **Zero-Copy Speedup** | `CPU Total / Effective GPU` | Best case (cuda_blackboard pipeline) |

---

## Switching Branches

`vcs import` resets **all** repos to the exact tags/commits in `autoware.repos` (e.g. nebula → `v0.3.2.2`, cuda_blackboard → `0.3.0`), regardless of what branch they are currently on. No need to manually checkout main in each repo first.

### → CPU baseline

```bash
cd ~/autoware
vcs import src < repositories/autoware.repos
./setup-dev-env.sh

cd ~/autoware/src/sensor_component/external/nebula
git checkout benchmark/cpu-baseline
```

### → GPU

```bash
cd ~/autoware/src/sensor_component/external/nebula
git checkout feat/cuda-refactor

cd ~/autoware/src/universe/external/cuda_blackboard
git checkout feat/cuda-refactor
```

### → main (autoware.repos default)

```bash
cd ~/autoware
vcs import src < repositories/autoware.repos
./setup-dev-env.sh
```

> **Rebuild required** after every branch switch — different branches have different code and compile flags.

---

## Benchmark Runner Options

| Flag | Default | Description |
|---|---|---|
| `--cpu` / `--gpu` | `--cpu` | Select mode |
| `-n, --n-iterations` | 10 | Number of iterations |
| `-t, --runtime` | 30 | Seconds per iteration |
| `-m, --sensor-model` | Pandar128E4X | Sensor model |
| `-o, --output-dir` | `./benchmark_output` | Output directory |
| `-b, --rosbag-path` | auto-detected | Path to rosbag |
| `-f, --maxfreq` | 2500000 | CPU frequency lock (Hz) |
| `-c, --taskset-cores` | all | CPU core affinity |

---

## Previous Results (2026-02-03)

| Implementation | Mean | P50 | P95 | P99 | Speedup |
|---|---|---|---|---|---|
| CPU | 3.686 ms | 3.616 ms | 4.285 ms | 6.048 ms | 1.0x |
| GPU | 0.955 ms | 0.910 ms | 1.566 ms | 1.786 ms | **3.86x** |
| GPU Zero-Copy | 0.784 ms | — | — | — | **4.70x** |

GPU time breakdown: H2D 55.3%, Kernel 26.8%, D2H 17.9%.

---

## Notes

- **CPU frequency locking** is important for consistent results. The benchmark runner handles this automatically.
- GPU profiling is gated behind `NEBULA_CUDA_PROFILING` (CMake option, default OFF). Production builds have zero profiling overhead.
- The benchmark scripts (`benchmark_runner.bash`, `analyze_benchmark.py`) are committed on both `benchmark/cpu-baseline` and `feat/cuda-refactor` branches under `scripts/`.
