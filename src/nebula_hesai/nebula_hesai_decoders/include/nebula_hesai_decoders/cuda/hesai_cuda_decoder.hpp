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
  uint8_t in_current_scan;  // 1 = belongs to current scan, 0 = belongs to output/next scan
  uint32_t entry_id;        // Block group ID for batched processing (used for sorting & filtering)
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

/// @brief Maximum number of frames (mirrors) supported for multi-frame sensors like AT128
static constexpr uint32_t MAX_CUDA_FRAMES = 8;

/// @brief Frame angle info for multi-frame sensors (AT128 has 4 frames)
struct CudaFrameAngleInfo
{
  uint32_t fov_start;        // Raw azimuth where FOV starts for this frame
  uint32_t fov_end;          // Raw azimuth where FOV ends for this frame
  uint32_t timestamp_reset;  // Raw azimuth where timestamp resets for this frame
  uint32_t scan_emit;        // Raw azimuth where scan emit occurs for this frame
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
  // Overlap detection parameters (raw azimuth in 0.01 degree units)
  uint32_t timestamp_reset_angle_raw;
  uint32_t emit_angle_raw;
  uint32_t n_azimuths_raw;  // Total azimuth count (e.g., 36000 for 0.01 deg resolution)
  uint32_t max_output_points;  // Maximum output buffer size for sparse indexing (batched mode)

  // Azimuth scaling for sensors with different degree_subdivisions
  // For sensors with degree_subdivisions=100: scale=1 (no scaling needed)
  // For AT128 with degree_subdivisions=25600: scale=256 (raw_azimuth/256 = LUT index)
  uint32_t azimuth_scale;  // Scale factor: raw_azimuth / azimuth_scale = LUT index

  // Multi-frame support for sensors like AT128 (has 4 mirror frames)
  uint32_t n_frames;  // Number of frames (1 for single-frame, 4 for AT128)
  CudaFrameAngleInfo frame_angles[MAX_CUDA_FRAMES];  // Per-frame angle boundaries
  bool is_multi_frame;  // True if sensor has multiple frames (uses frame_angles)
};

/// @brief GPU point cloud structure for zero-copy pipeline
/// Allows downstream CUDA modules to access points directly on GPU without D2H copy
struct GpuPointCloud
{
  /// Device pointer to point array (valid until next flush or decoder destruction)
  const CudaNebulaPoint* d_points = nullptr;
  /// Number of valid points in the buffer
  uint32_t point_count = 0;
  /// Timestamp of the scan in nanoseconds
  uint64_t timestamp_ns = 0;
  /// Whether the data is valid (false if not in GPU pipeline mode or no data available)
  bool valid = false;
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

/// @brief PointCloud2 format constants for PointXYZIRCAEDT
constexpr uint32_t POINTCLOUD2_POINT_STEP = 32;  // sizeof(PointXYZIRCAEDT)

}  // namespace nebula::drivers::cuda

// =============================================================================
// C-linkage function declarations for CUDA kernel launches
// =============================================================================

extern "C" {

/// @brief Launch kernel to decode a single Hesai packet
void launch_decode_hesai_packet(
    const uint16_t* d_distances,
    const uint8_t* d_reflectivities,
    const nebula::drivers::cuda::CudaAngleCorrectionData* d_angle_lut,
    const nebula::drivers::cuda::CudaDecoderConfig& config,
    nebula::drivers::cuda::CudaNebulaPoint* d_points,
    uint32_t* d_count,
    uint32_t n_azimuths,
    uint32_t raw_azimuth,
    cudaStream_t stream);

/// @brief Launch batched kernel to decode entire scan
void launch_decode_hesai_scan_batch(
    const uint16_t* d_distances_batch,
    const uint8_t* d_reflectivities_batch,
    const uint32_t* d_raw_azimuths,
    const uint32_t* d_n_returns,
    const uint32_t* d_last_azimuths,
    const nebula::drivers::cuda::CudaAngleCorrectionData* d_angle_lut,
    const nebula::drivers::cuda::CudaDecoderConfig& config,
    nebula::drivers::cuda::CudaNebulaPoint* d_points,
    uint32_t* d_count,
    uint32_t n_azimuths,
    uint32_t n_packets,
    cudaStream_t stream);

/// @brief Launch kernel to convert CudaNebulaPoint to PointCloud2 format
/// Converts to PointXYZIRCAEDT layout (32 bytes per point)
/// Uses atomic compaction to filter out invalid points
/// @param d_input Input CudaNebulaPoint array
/// @param d_output Output byte buffer (must be at least input_count * 32 bytes)
/// @param d_output_count Output: number of valid points written
/// @param input_count Number of input points
/// @param filter_current_scan If true, only include points with in_current_scan=1
/// @param stream CUDA stream for async execution
void launch_convert_to_pointcloud2(
    const nebula::drivers::cuda::CudaNebulaPoint* d_input,
    uint8_t* d_output,
    uint32_t* d_output_count,
    uint32_t input_count,
    bool filter_current_scan,
    cudaStream_t stream);

/// @brief Launch ordered conversion kernel (preserves point order, no compaction)
/// Faster but may include invalid/zeroed points that need post-filtering
/// @param d_input Input CudaNebulaPoint array
/// @param d_output Output byte buffer
/// @param d_valid_mask Optional output: validity mask (1=valid, 0=filtered)
/// @param input_count Number of input points
/// @param filter_current_scan If true, zero out points with in_current_scan=0
/// @param stream CUDA stream for async execution
void launch_convert_to_pointcloud2_ordered(
    const nebula::drivers::cuda::CudaNebulaPoint* d_input,
    uint8_t* d_output,
    uint8_t* d_valid_mask,
    uint32_t input_count,
    bool filter_current_scan,
    cudaStream_t stream);

}  // extern "C"
