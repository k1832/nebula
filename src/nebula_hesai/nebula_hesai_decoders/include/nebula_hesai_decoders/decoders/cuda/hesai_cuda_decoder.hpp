// Copyright 2025 TIER IV, Inc.
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

#pragma once

#ifndef NEBULA_CUDA_ENABLED
#error "This header requires CUDA support. Define NEBULA_CUDA_ENABLED to include this file."
#endif

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace nebula::drivers::cuda
{

/// @brief GPU-side point structure matching NebulaPoint layout
struct alignas(32) CudaNebulaPoint
{
  float x;
  float y;
  float z;
  float distance;
  float azimuth;
  float elevation;
  uint8_t intensity;
  uint8_t return_type;
  uint16_t channel;
  uint32_t time_stamp;
};

/// @brief Pre-computed angle correction data for GPU
struct CudaAngleCorrectionData
{
  float azimuth_rad;
  float elevation_rad;
  float sin_azimuth;
  float cos_azimuth;
  float sin_elevation;
  float cos_elevation;
};

/// @brief Configuration for CUDA decoder
struct CudaDecoderConfig
{
  float min_range;
  float max_range;
  float sensor_min_range;
  float sensor_max_range;
  float dual_return_distance_threshold;
  float fov_min_rad;
  float fov_max_rad;
  float scan_emit_angle_rad;
  uint32_t n_channels;
  uint32_t n_blocks;
  uint32_t max_returns;
  float dis_unit;
};

/// @brief CUDA decoder for Hesai LiDAR point cloud processing
class HesaiCudaDecoder
{
public:
  HesaiCudaDecoder();
  ~HesaiCudaDecoder();

  /// @brief Initialize CUDA resources
  /// @param max_points Maximum number of points per scan
  /// @param n_channels Number of laser channels
  /// @return true if initialization successful
  bool initialize(size_t max_points, uint32_t n_channels);

  /// @brief Upload angle correction lookup table to GPU
  /// @param angle_data Pre-computed angle corrections for all azimuth/channel combinations
  /// @param n_azimuths Number of azimuth divisions
  /// @param n_channels Number of laser channels
  void upload_angle_corrections(
    const std::vector<CudaAngleCorrectionData> & angle_data, uint32_t n_azimuths,
    uint32_t n_channels);

  /// @brief Decode packet data on GPU
  /// @param packet_data Raw packet data
  /// @param packet_size Size of packet data
  /// @param config Decoder configuration
  /// @param output_points Output points (device memory)
  /// @param output_count Output point count (device memory)
  /// @param stream CUDA stream for async execution
  void decode_packet(
    const uint8_t * packet_data, size_t packet_size, const CudaDecoderConfig & config,
    CudaNebulaPoint * output_points, uint32_t * output_count, cudaStream_t stream);

  /// @brief Copy results from GPU to host
  /// @param d_points Device points
  /// @param d_count Device point count
  /// @param h_points Host points buffer
  /// @param max_points Maximum points to copy
  /// @param stream CUDA stream
  /// @return Number of points copied
  size_t copy_results_to_host(
    const CudaNebulaPoint * d_points, const uint32_t * d_count, CudaNebulaPoint * h_points,
    size_t max_points, cudaStream_t stream);

  /// @brief Get device pointer for output points
  CudaNebulaPoint * get_device_points() { return d_output_points_; }

  /// @brief Get device pointer for output count
  uint32_t * get_device_count() { return d_output_count_; }

  /// @brief Synchronize CUDA stream
  void synchronize(cudaStream_t stream);

private:
  // Device memory
  CudaNebulaPoint * d_output_points_ = nullptr;
  uint32_t * d_output_count_ = nullptr;
  uint8_t * d_packet_buffer_ = nullptr;
  CudaAngleCorrectionData * d_angle_corrections_ = nullptr;

  // Pinned host memory for async transfers
  uint8_t * h_pinned_packet_buffer_ = nullptr;

  size_t max_points_ = 0;
  size_t packet_buffer_size_ = 0;
  uint32_t n_azimuths_ = 0;
  uint32_t n_channels_ = 0;
  bool initialized_ = false;
};

// CUDA kernel launch functions
void launch_decode_hesai_packet_kernel(
  const uint8_t * d_packet, size_t packet_size, const CudaDecoderConfig * d_config,
  const CudaAngleCorrectionData * d_angle_corrections, uint32_t n_azimuths, uint32_t n_channels,
  CudaNebulaPoint * d_output_points, uint32_t * d_output_count, cudaStream_t stream);

}  // namespace nebula::drivers::cuda
