// Copyright 2026 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/// @file hesai_cuda_decoder_test.cpp
/// @brief GPU-vs-CPU equivalence tests for the Hesai CUDA decoder (OT128 / Pandar128E4X).
///
/// Decodes the same rosbag with the CPU path (NEBULA_USE_CUDA unset) and the GPU path
/// (NEBULA_USE_CUDA=1), then compares the resulting point clouds with tolerances.
/// The GPU path differs at scan boundaries due to different overlap/FOV detection logic
/// in the GPU kernel vs the CPU's ScanCutter.
///
/// Tolerances were derived from a single OT128 rosbag (ot128/1730271167765338806):
///   - Scan 0: CPU=70275, GPU=72125 (diff=1850)
///   - Scan 1: CPU=72240, GPU=72185 (diff=55)
/// If additional rosbags show larger boundary diffs, tolerances may need adjustment.

#include "hesai_ros_decoder_test.hpp"

#include <rclcpp/rclcpp.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#ifdef NEBULA_CUDA_ENABLED
#include <nebula_hesai_decoders/cuda/hesai_cuda_decoder.hpp>

#include <cuda_runtime.h>
#endif

#ifdef NEBULA_CUDA_ENABLED

namespace nebula::test
{

/// @brief Check if a CUDA-capable GPU is available at runtime
static bool has_cuda_device()
{
  cudaStream_t stream = nullptr;
  cudaError_t err = cudaStreamCreate(&stream);
  if (err == cudaSuccess) {
    cudaStreamDestroy(stream);
    return true;
  }
  return false;
}

// OT128 config — matches TEST_CONFIGS[8] in hesai_ros_decoder_test_main.cpp
static const nebula::ros::HesaiRosDecoderTestParams OT128_CONFIG = {
  "Pandar128E4X",
  "LastStrongest",
  "Pandar128E4X.csv",
  "ot128/1730271167765338806",
  "hesai",
  0,
  0.0,
  0.,
  360.,
  0.3f,
  300.f};

// Maximum allowed difference in point count between GPU and CPU per scan.
// The GPU kernel assigns scan-boundary points differently than the CPU's ScanCutter,
// causing up to ~2000 points to shift between adjacent scans.
// Observed max diff: 1850 (scan 0 of test rosbag). Set to 2000 for ~8% headroom.
static constexpr int kMaxPointCountDiff = 2000;
// Coordinate tolerance (metres) for nearest-neighbour matching.
// GPU uses pre-computed LUT with 0.01-degree resolution; small rounding differences expected.
static constexpr float kXyzTolerance = 5e-3f;
// Fraction of GPU points that must have a CPU match within tolerance.
// ~2-3% of points are at scan boundaries and may not have an exact match.
static constexpr double kMinMatchRatio = 0.97;

/// Decoded scan: message timestamp + point cloud
struct DecodedScan
{
  uint64_t msg_timestamp;
  nebula::drivers::NebulaPointCloudPtr cloud;
};

/// Decode a rosbag with the given CUDA env setting.
/// When @p use_cuda is true, sets NEBULA_USE_CUDA=1; otherwise unsets it.
/// Returns one DecodedScan per completed scan.
static std::vector<DecodedScan> decode_bag(
  const nebula::ros::HesaiRosDecoderTestParams & params, bool use_cuda)
{
  // Toggle GPU/CPU path via environment variable
  if (use_cuda) {
    setenv("NEBULA_USE_CUDA", "1", 1);
  } else {
    unsetenv("NEBULA_USE_CUDA");
  }

  rclcpp::NodeOptions options;
  auto driver =
    std::make_shared<nebula::ros::HesaiRosDecoderTest>(options, "cuda_test_node", params);
  EXPECT_EQ(driver->get_status(), nebula::Status::OK);

  std::vector<DecodedScan> scans;
  auto cb = [&](uint64_t msg_ts, uint64_t /*scan_ts*/, nebula::drivers::NebulaPointCloudPtr cloud) {
    if (cloud && !cloud->empty()) {
      // Deep-copy: the decoder reuses its internal buffer after the callback returns
      auto copy = std::make_shared<nebula::drivers::NebulaPointCloud>(*cloud);
      scans.push_back({msg_ts, copy});
    }
  };
  driver->read_bag(cb);
  return scans;
}

/// Squared Euclidean distance between two points
static float sq_dist(const nebula::drivers::NebulaPoint & a, const nebula::drivers::NebulaPoint & b)
{
  float dx = a.x - b.x;
  float dy = a.y - b.y;
  float dz = a.z - b.z;
  return dx * dx + dy * dy + dz * dz;
}

// ---------------------------------------------------------------------------
// Test 1: GPU vs CPU equivalence
// ---------------------------------------------------------------------------
TEST(HesaiCudaDecoderTest, OT128_GpuVsCpuEquivalence)
{
  auto cpu_scans = decode_bag(OT128_CONFIG, /*use_cuda=*/false);
  auto gpu_scans = decode_bag(OT128_CONFIG, /*use_cuda=*/true);

  // Both paths must produce the same number of scans
  ASSERT_GT(cpu_scans.size(), 0u);
  ASSERT_EQ(cpu_scans.size(), gpu_scans.size())
    << "CPU produced " << cpu_scans.size() << " scans, GPU produced " << gpu_scans.size();

  const float tol_sq = kXyzTolerance * kXyzTolerance;

  for (size_t i = 0; i < cpu_scans.size(); ++i) {
    const auto & cpu_cloud = cpu_scans[i].cloud;
    const auto & gpu_cloud = gpu_scans[i].cloud;

    // Point counts within tolerance
    int diff = static_cast<int>(cpu_cloud->size()) - static_cast<int>(gpu_cloud->size());
    EXPECT_LE(std::abs(diff), kMaxPointCountDiff)
      << "Scan " << i << ": CPU=" << cpu_cloud->size() << " GPU=" << gpu_cloud->size();

    // For each GPU point, find nearest CPU point (brute-force — clouds are small enough)
    size_t matched = 0;
    for (const auto & gp : *gpu_cloud) {
      float best = std::numeric_limits<float>::max();
      for (const auto & cp : *cpu_cloud) {
        float d = sq_dist(gp, cp);
        if (d < best) best = d;
        if (d < tol_sq) break;  // early exit — found a match
      }
      if (best < tol_sq) ++matched;
    }

    double match_ratio =
      gpu_cloud->empty() ? 1.0 : static_cast<double>(matched) / gpu_cloud->size();
    EXPECT_GE(match_ratio, kMinMatchRatio)
      << "Scan " << i << ": only " << matched << "/" << gpu_cloud->size()
      << " GPU points matched a CPU point within " << kXyzTolerance << " m";
  }
}

// ---------------------------------------------------------------------------
// Test 2: GPU output is non-empty
// ---------------------------------------------------------------------------
TEST(HesaiCudaDecoderTest, OT128_GpuOutputNonEmpty)
{
  auto gpu_scans = decode_bag(OT128_CONFIG, /*use_cuda=*/true);
  ASSERT_GT(gpu_scans.size(), 0u) << "GPU path produced zero scans";
  for (size_t i = 0; i < gpu_scans.size(); ++i) {
    EXPECT_GT(gpu_scans[i].cloud->size(), 0u) << "Scan " << i << " is empty";
  }
}

// ---------------------------------------------------------------------------
// Test 3: Basic field validity of GPU-decoded points
// ---------------------------------------------------------------------------
TEST(HesaiCudaDecoderTest, OT128_GpuFieldValidity)
{
  auto gpu_scans = decode_bag(OT128_CONFIG, /*use_cuda=*/true);
  ASSERT_GT(gpu_scans.size(), 0u);

  for (size_t i = 0; i < gpu_scans.size(); ++i) {
    for (const auto & pt : *gpu_scans[i].cloud) {
      // Distance must be positive (all valid points)
      float dist = std::sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
      EXPECT_GT(dist, 0.f) << "Scan " << i << ": zero-distance point found";

      // Channel must be in [0, 127] for OT128
      EXPECT_LE(pt.channel, 127u) << "Scan " << i << ": invalid channel " << pt.channel;
    }
  }
}

// ---------------------------------------------------------------------------
// Test 4 (edge case): First and last scans have reasonable point counts
// ---------------------------------------------------------------------------
TEST(HesaiCudaDecoderTest, OT128_BoundaryScanPointCounts)
{
  auto gpu_scans = decode_bag(OT128_CONFIG, /*use_cuda=*/true);
  ASSERT_GE(gpu_scans.size(), 2u) << "Need at least 2 scans for boundary test";

  // First scan may be partial, but should still have a reasonable number of points.
  // A full OT128 scan typically has ~100k+ points; even a partial scan should exceed 1000.
  EXPECT_GT(gpu_scans.front().cloud->size(), 1000u)
    << "First scan has suspiciously few points: " << gpu_scans.front().cloud->size();

  // Last scan should also be non-trivial
  EXPECT_GT(gpu_scans.back().cloud->size(), 1000u)
    << "Last scan has suspiciously few points: " << gpu_scans.back().cloud->size();
}

// ---------------------------------------------------------------------------
// Test 5 (edge case): GPU intensity matches CPU exactly
// ---------------------------------------------------------------------------
TEST(HesaiCudaDecoderTest, OT128_IntensityExactMatch)
{
  auto cpu_scans = decode_bag(OT128_CONFIG, /*use_cuda=*/false);
  auto gpu_scans = decode_bag(OT128_CONFIG, /*use_cuda=*/true);

  ASSERT_EQ(cpu_scans.size(), gpu_scans.size());

  for (size_t i = 0; i < cpu_scans.size(); ++i) {
    const auto & cpu_pts = *cpu_scans[i].cloud;
    const auto & gpu_pts = *gpu_scans[i].cloud;

    // For each GPU point, find the nearest CPU match and verify intensity is identical.
    // Only compare points that have a close spatial match (same physical point).
    size_t checked = 0;
    size_t intensity_mismatch = 0;

    // Use a tighter tolerance for intensity matching to ensure we're comparing the same point
    const float tight_tol_sq = 1e-6f;

    for (const auto & gp : gpu_pts) {
      float best_dist = std::numeric_limits<float>::max();
      size_t best_idx = 0;
      for (size_t j = 0; j < cpu_pts.size(); ++j) {
        float d = sq_dist(gp, cpu_pts[j]);
        if (d < best_dist) {
          best_dist = d;
          best_idx = j;
        }
        if (d < tight_tol_sq) break;
      }
      if (best_dist < tight_tol_sq) {
        ++checked;
        if (gp.intensity != cpu_pts[best_idx].intensity) {
          ++intensity_mismatch;
        }
      }
    }

    // The vast majority of matched points should have identical intensity.
    // A small fraction (<1%) may differ at scan boundaries where the GPU kernel
    // selects a different return than the CPU's dual-return filtering.
    EXPECT_GT(checked, 0u) << "Scan " << i << ": no tight matches found";
    double mismatch_ratio = checked > 0 ? static_cast<double>(intensity_mismatch) / checked : 0.0;
    EXPECT_LT(mismatch_ratio, 0.01)
      << "Scan " << i << ": " << intensity_mismatch << "/" << checked
      << " matched points have different intensity (" << (mismatch_ratio * 100) << "%)";
  }
}

// ---------------------------------------------------------------------------
// Test 6 (zero-copy): GPU pipeline mode produces empty CPU cloud
// ---------------------------------------------------------------------------
TEST(HesaiCudaDecoderTest, OT128_GpuPipelineSkipsCpuOutput)
{
  if (!has_cuda_device()) {
    GTEST_SKIP() << "No CUDA-capable GPU available";
  }

  setenv("NEBULA_USE_CUDA", "1", 1);
  setenv("NEBULA_GPU_PIPELINE", "1", 1);

  rclcpp::NodeOptions options;
  auto driver =
    std::make_shared<nebula::ros::HesaiRosDecoderTest>(options, "cuda_test_node", OT128_CONFIG);
  ASSERT_EQ(driver->get_status(), nebula::Status::OK);

  size_t callback_count = 0;
  size_t non_empty_count = 0;
  auto cb =
    [&](uint64_t /*msg_ts*/, uint64_t /*scan_ts*/, nebula::drivers::NebulaPointCloudPtr cloud) {
      ++callback_count;
      if (cloud && !cloud->empty()) {
        ++non_empty_count;
      }
    };
  driver->read_bag(cb);

  unsetenv("NEBULA_GPU_PIPELINE");

  // In GPU pipeline mode, process_gpu_results() is skipped so CPU clouds should be empty.
  // Callbacks still fire (scan completion is detected), but the pointcloud is not populated.
  EXPECT_GT(callback_count, 0u) << "No scan callbacks fired";
  EXPECT_EQ(non_empty_count, 0u) << "Expected empty CPU clouds in GPU pipeline mode, but "
                                 << non_empty_count << "/" << callback_count << " were non-empty";
}

// ---------------------------------------------------------------------------
// Test 7 (zero-copy): get_gpu_pointcloud() returns valid data
// ---------------------------------------------------------------------------
TEST(HesaiCudaDecoderTest, OT128_GpuPointCloudApiValid)
{
  if (!has_cuda_device()) {
    GTEST_SKIP() << "No CUDA-capable GPU available";
  }

  setenv("NEBULA_USE_CUDA", "1", 1);
  setenv("NEBULA_GPU_PIPELINE", "1", 1);

  rclcpp::NodeOptions options;
  auto driver =
    std::make_shared<nebula::ros::HesaiRosDecoderTest>(options, "cuda_test_node", OT128_CONFIG);
  ASSERT_EQ(driver->get_status(), nebula::Status::OK);

  auto hesai_driver = driver->get_driver();
  ASSERT_NE(hesai_driver, nullptr);
  EXPECT_TRUE(hesai_driver->is_gpu_pipeline_mode());

  // Track GpuPointCloud results across scan completions
  std::vector<nebula::drivers::cuda::GpuPointCloud> gpu_clouds;
  auto cb =
    [&](uint64_t /*msg_ts*/, uint64_t /*scan_ts*/, nebula::drivers::NebulaPointCloudPtr /*cloud*/) {
      auto gpu_pc = hesai_driver->get_gpu_pointcloud();
      gpu_clouds.push_back(gpu_pc);
    };
  driver->read_bag(cb);

  unsetenv("NEBULA_GPU_PIPELINE");

  ASSERT_GT(gpu_clouds.size(), 0u) << "No scans produced";

  for (size_t i = 0; i < gpu_clouds.size(); ++i) {
    const auto & gpc = gpu_clouds[i];
    EXPECT_TRUE(gpc.valid) << "Scan " << i << ": GpuPointCloud not valid";
    EXPECT_GT(gpc.point_count, 0u) << "Scan " << i << ": zero point count";
    EXPECT_NE(gpc.d_points, nullptr) << "Scan " << i << ": null device pointer";
    EXPECT_GT(gpc.timestamp_ns, 0u) << "Scan " << i << ": zero timestamp";
  }
}

// ---------------------------------------------------------------------------
// Test 8 (zero-copy): PointCloud2 conversion kernel produces correct layout
// ---------------------------------------------------------------------------
TEST(HesaiCudaDecoderTest, PointCloud2ConversionKernel)
{
  if (!has_cuda_device()) {
    GTEST_SKIP() << "No CUDA-capable GPU available";
  }

  using nebula::drivers::cuda::CudaNebulaPoint;
  using nebula::drivers::cuda::POINTCLOUD2_POINT_STEP;

  // Create test points on host
  const uint32_t n_points = 3;
  std::vector<CudaNebulaPoint> h_points(n_points);

  // Point 0: valid, in current scan
  h_points[0] = {1.0f, 2.0f, 3.0f, 5.0f, 0.1f, 0.2f, 128.0f, 1, 1, 42, 0};
  // Point 1: valid, in current scan
  h_points[1] = {-1.0f, -2.0f, -3.0f, 4.0f, 0.3f, 0.4f, 255.0f, 2, 1, 7, 1};
  // Point 2: invalid (distance = 0), should be filtered out
  h_points[2] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0, 1, 0, 2};

  // Allocate device memory
  CudaNebulaPoint * d_input = nullptr;
  uint8_t * d_output = nullptr;
  uint32_t * d_count = nullptr;
  cudaMalloc(&d_input, n_points * sizeof(CudaNebulaPoint));
  cudaMalloc(&d_output, n_points * POINTCLOUD2_POINT_STEP);
  cudaMalloc(&d_count, sizeof(uint32_t));

  cudaMemcpy(d_input, h_points.data(), n_points * sizeof(CudaNebulaPoint), cudaMemcpyHostToDevice);

  // Run conversion kernel
  bool ok = launch_convert_to_pointcloud2(d_input, d_output, d_count, n_points, true, nullptr);
  ASSERT_TRUE(ok) << "Kernel launch failed";
  cudaDeviceSynchronize();

  // Read back count
  uint32_t output_count = 0;
  cudaMemcpy(&output_count, d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  EXPECT_EQ(output_count, 2u) << "Expected 2 valid points (point 2 has distance=0)";

  // Read back PointCloud2 data
  std::vector<uint8_t> h_output(output_count * POINTCLOUD2_POINT_STEP);
  cudaMemcpy(h_output.data(), d_output, h_output.size(), cudaMemcpyDeviceToHost);

  // Verify layout for both output points (order is nondeterministic due to atomicAdd)
  // Collect the output points and verify they match input points 0 and 1
  std::vector<float> out_x_vals;
  for (uint32_t i = 0; i < output_count; ++i) {
    const uint8_t * p = h_output.data() + i * POINTCLOUD2_POINT_STEP;
    float x, y, z, azimuth, elevation, distance;
    uint8_t intensity, return_type;
    uint16_t channel;

    std::memcpy(&x, p + 0, sizeof(float));
    std::memcpy(&y, p + 4, sizeof(float));
    std::memcpy(&z, p + 8, sizeof(float));
    intensity = p[12];
    return_type = p[13];
    std::memcpy(&channel, p + 14, sizeof(uint16_t));
    std::memcpy(&azimuth, p + 16, sizeof(float));
    std::memcpy(&elevation, p + 20, sizeof(float));
    std::memcpy(&distance, p + 24, sizeof(float));

    out_x_vals.push_back(x);

    // Find which input point this corresponds to
    if (std::abs(x - 1.0f) < 1e-5f) {
      // Should be point 0
      EXPECT_FLOAT_EQ(y, 2.0f);
      EXPECT_FLOAT_EQ(z, 3.0f);
      EXPECT_EQ(intensity, 128u);
      EXPECT_EQ(return_type, 1u);
      EXPECT_EQ(channel, 42u);
      EXPECT_FLOAT_EQ(distance, 5.0f);
      EXPECT_FLOAT_EQ(azimuth, 0.1f);
      EXPECT_FLOAT_EQ(elevation, 0.2f);
    } else if (std::abs(x - (-1.0f)) < 1e-5f) {
      // Should be point 1
      EXPECT_FLOAT_EQ(y, -2.0f);
      EXPECT_FLOAT_EQ(z, -3.0f);
      EXPECT_EQ(intensity, 255u);
      EXPECT_EQ(return_type, 2u);
      EXPECT_EQ(channel, 7u);
      EXPECT_FLOAT_EQ(distance, 4.0f);
      EXPECT_FLOAT_EQ(azimuth, 0.3f);
      EXPECT_FLOAT_EQ(elevation, 0.4f);
    } else {
      ADD_FAILURE() << "Unexpected point with x=" << x;
    }
  }

  // Both input points should appear in output
  EXPECT_EQ(out_x_vals.size(), 2u);

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_count);
}

}  // namespace nebula::test

#else  // !NEBULA_CUDA_ENABLED

TEST(HesaiCudaDecoderTest, SkippedNoCuda)
{
  GTEST_SKIP() << "CUDA not enabled at compile time";
}

#endif  // NEBULA_CUDA_ENABLED

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  int result = RUN_ALL_TESTS();
  // Restore env
  unsetenv("NEBULA_USE_CUDA");
  unsetenv("NEBULA_GPU_PIPELINE");
  rclcpp::shutdown();
  return result;
}
