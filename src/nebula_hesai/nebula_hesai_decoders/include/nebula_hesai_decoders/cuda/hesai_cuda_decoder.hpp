// Copyright 2024 TIER IV, Inc.
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

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace nebula::drivers::cuda
{

/// @brief Point structure optimized for CUDA processing
struct CudaNebulaPoint
{
  float x;
  float y;
  float z;
  float distance;
  float azimuth;
  float elevation;
  float intensity;
  uint8_t return_type;
  uint16_t channel;
  uint8_t padding;       // Padding for alignment
  uint32_t entry_id;     // Block group ID for batched processing (used for sorting & filtering)
};

/// @brief Angle correction data for CUDA lookup table
struct CudaAngleCorrectionData
{
  float azimuth_rad;
  float elevation_rad;
  float sin_azimuth;
  float cos_azimuth;
  float sin_elevation;
  float cos_elevation;
};

/// @brief Configuration data for CUDA decoder
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
  uint32_t data_stride;  // Stride between channels in input data (for batched mode)
  uint32_t entry_id;     // Entry ID for batched mode (0 in per-packet mode)
};

/// @brief Main CUDA decoder class for Hesai LiDAR
class HesaiCudaDecoder
{
public:
  HesaiCudaDecoder();
  ~HesaiCudaDecoder();

  /// @brief Initialize decoder with maximum points and channels
  /// @param max_points Maximum number of points in a scan
  /// @param n_channels Number of laser channels
  /// @return true if initialization succeeded
  bool initialize(size_t max_points, uint32_t n_channels);

  /// @brief Upload angle correction lookup table to GPU
  /// @param angle_lut Vector of angle correction data (azimuth_divisions * n_channels)
  /// @param n_azimuths Number of azimuth divisions
  /// @param n_channels Number of laser channels
  /// @return true if upload succeeded
  bool upload_angle_corrections(
    const std::vector<CudaAngleCorrectionData> & angle_lut,
    uint32_t n_azimuths,
    uint32_t n_channels);

  /// @brief Decode a packet on GPU
  /// @param packet_data Raw packet data
  /// @param packet_size Size of packet in bytes
  /// @param config Decoder configuration
  /// @param d_points Device pointer for output points
  /// @param d_count Device pointer for output point count
  /// @param stream CUDA stream for async execution
  /// @return true if decode succeeded
  bool decode_packet(
    const uint8_t * packet_data,
    size_t packet_size,
    const CudaDecoderConfig & config,
    CudaNebulaPoint * d_points,
    uint32_t * d_count,
    cudaStream_t stream);

  /// @brief Get device pointer to angle lookup table
  /// @return Device pointer to angle LUT
  CudaAngleCorrectionData * get_angle_lut() const { return d_angle_lut_; }

private:
  /// @brief Device memory for angle correction lookup table
  CudaAngleCorrectionData * d_angle_lut_ = nullptr;
  /// @brief Number of azimuth divisions
  uint32_t n_azimuths_ = 0;
  /// @brief Number of laser channels
  uint32_t n_channels_ = 0;
  /// @brief Whether decoder is initialized
  bool initialized_ = false;
};

}  // namespace nebula::drivers::cuda
