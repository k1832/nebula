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

#include "nebula_hesai_decoders/cuda/hesai_cuda_decoder.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

namespace nebula::drivers::cuda
{

// Device function: Check if angle is between start and end (handles wrap-around)
// Works with both radians and raw azimuth units
__device__ __forceinline__ bool cuda_angle_is_between(float start, float end, float angle)
{
  // Handle wrap-around case: end < start means range crosses 0/2pi boundary
  if (start <= end) {
    return (start <= angle && angle <= end);
  } else {
    return (angle <= end || start <= angle);
  }
}

// Device function: Check if angle is between start and end (uint32 version for raw azimuths)
__device__ __forceinline__ bool cuda_angle_is_between_raw(
  uint32_t start, uint32_t end, uint32_t angle, uint32_t max_angle)
{
  // Normalize angles to [0, max_angle)
  start = start % max_angle;
  end = end % max_angle;
  angle = angle % max_angle;

  if (start <= end) {
    return (start <= angle && angle <= end);
  } else {
    return (angle <= end || start <= angle);
  }
}

// Device function: Check if we're inside the overlap region
// Overlap is the region between timestamp_reset_angle and emit_angle
__device__ __forceinline__ bool cuda_is_inside_overlap(
  uint32_t last_azimuth, uint32_t current_azimuth,
  uint32_t timestamp_reset_angle, uint32_t emit_angle, uint32_t max_angle)
{
  return cuda_angle_is_between_raw(timestamp_reset_angle, emit_angle, current_azimuth, max_angle) ||
         cuda_angle_is_between_raw(timestamp_reset_angle, emit_angle, last_azimuth, max_angle);
}

// CUDA kernel for decoding Hesai LiDAR points
// Highly optimized for:
// - Coalesced memory access patterns
// - Minimal thread divergence
// - Maximum occupancy with register efficiency
__global__ void decode_hesai_packet_kernel(
  const uint16_t * __restrict__ distances,
  const uint8_t * __restrict__ reflectivities,
  const CudaAngleCorrectionData * __restrict__ angle_lut,
  const CudaDecoderConfig config,
  CudaNebulaPoint * __restrict__ output_points,
  uint32_t * __restrict__ output_count,
  uint32_t n_azimuths,
  uint32_t raw_azimuth)
{
  // Thread grid: blockIdx.x = channel blocks, blockIdx.y = return/block index
  // This ensures coalesced memory access
  const uint32_t channel_id = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t block_id = blockIdx.y;

  // Bounds check - early exit for out-of-range threads
  if (channel_id >= config.n_channels || block_id >= config.n_blocks) {
    return;
  }

  // Calculate linear index using data_stride for input buffer access
  // data_stride may differ from n_blocks in batched mode (staging buffer uses max_returns stride)
  const uint32_t data_stride = config.data_stride > 0 ? config.data_stride : config.n_blocks;
  const uint32_t data_idx = channel_id * data_stride + block_id;

  // Load distance and reflectivity with coalesced access
  const uint16_t raw_distance = distances[data_idx];
  const uint8_t reflectivity = reflectivities[data_idx];

  // Early exit for invalid points (distance == 0)
  // Using warp-level branch divergence minimization
  if (raw_distance == 0) {
    return;
  }

  // Convert distance using unit scale
  const float distance = static_cast<float>(raw_distance) * config.dis_unit;

  // Range filtering - use configuration min/max ranges
  if (distance < config.min_range || distance > config.max_range) {
    return;
  }

  // Additional sensor-specific range check
  if (distance < config.sensor_min_range || distance > config.sensor_max_range) {
    return;
  }

  // Calculate azimuth index for lookup table (0.01 degree resolution)
  // raw_azimuth is in 0.01 degree units
  const uint32_t azimuth_idx = raw_azimuth % n_azimuths;

  // Lookup angle corrections with coalesced access
  // LUT is organized as [azimuth][channel] for optimal access pattern
  const uint32_t lut_idx = azimuth_idx * config.n_channels + channel_id;
  const CudaAngleCorrectionData angle_data = angle_lut[lut_idx];

  // Compute Cartesian coordinates using pre-calculated sin/cos
  // This avoids expensive transcendental function calls
  // Note: x uses sin, y uses cos (matching Hesai/nebula convention)
  const float xy_distance = distance * angle_data.cos_elevation;
  const float x = xy_distance * angle_data.sin_azimuth;
  const float y = xy_distance * angle_data.cos_azimuth;
  const float z = distance * angle_data.sin_elevation;

  // Atomic increment to get unique output index
  // Uses hardware atomic for best performance
  const uint32_t output_idx = atomicAdd(output_count, 1);

  // Write output point
  // Struct writes may not be perfectly coalesced, but this is unavoidable
  // with variable number of valid points per packet
  CudaNebulaPoint & out_pt = output_points[output_idx];
  out_pt.x = x;
  out_pt.y = y;
  out_pt.z = z;
  out_pt.distance = distance;
  out_pt.azimuth = angle_data.azimuth_rad;
  out_pt.elevation = angle_data.elevation_rad;
  out_pt.intensity = static_cast<float>(reflectivity);
  out_pt.return_type = static_cast<uint8_t>(block_id);  // block_id represents return index
  out_pt.channel = static_cast<uint16_t>(channel_id);
  out_pt.entry_id = config.entry_id;  // Used for batched mode post-processing
}

/// @brief Batched kernel for processing multiple packets in one launch
/// Processes an entire scan (typically ~3,240 packets) in a single kernel call
/// Includes FOV filtering and overlap/scan assignment on GPU
__global__ void decode_hesai_scan_batch_kernel(
  const uint16_t * __restrict__ d_distances_ring,
  const uint8_t * __restrict__ d_reflectivities_ring,
  const uint32_t * __restrict__ d_raw_azimuths,
  const uint32_t * __restrict__ d_n_returns,
  const uint32_t * __restrict__ d_last_azimuths,  // Per-entry last_azimuth for overlap check
  const CudaAngleCorrectionData * __restrict__ angle_lut,
  const CudaDecoderConfig config,
  CudaNebulaPoint * __restrict__ output_points,
  uint32_t * __restrict__ output_count,
  uint32_t n_azimuths,
  uint32_t n_packets)
{
  // Global thread ID across all blocks
  const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Total work: n_packets * n_channels * max_returns
  const uint32_t total_work = n_packets * config.n_channels * config.max_returns;
  if (global_tid >= total_work) return;

  // Decompose thread ID into (packet_id, channel_id, return_id)
  // Order matches CPU iteration: for each packet, for each channel, for each return/block
  // CPU: outer loop = channel, inner loop = return/block
  // So: packet varies slowest, channel varies middle, return varies fastest
  const uint32_t packet_id = global_tid / (config.n_channels * config.max_returns);
  const uint32_t channel_id = (global_tid / config.max_returns) % config.n_channels;
  const uint32_t return_id = global_tid % config.max_returns;

  // Bounds check for variable return modes (e.g., single/dual/triple return)
  if (return_id >= d_n_returns[packet_id]) return;

  // Calculate buffer index: packet_id * stride + channel_id * max_returns + return_id
  const uint32_t data_idx = packet_id * (config.n_channels * config.max_returns)
                           + channel_id * config.max_returns + return_id;

  // Load distance and reflectivity
  const uint16_t raw_distance = d_distances_ring[data_idx];
  const uint8_t reflectivity = d_reflectivities_ring[data_idx];

  // Early exit for invalid points
  if (raw_distance == 0) return;

  // Convert distance using unit scale
  const float distance = static_cast<float>(raw_distance) * config.dis_unit;

  // Range filtering
  if (distance < config.min_range || distance > config.max_range) return;
  if (distance < config.sensor_min_range || distance > config.sensor_max_range) return;

  // Get raw azimuth for this packet
  const uint32_t raw_azimuth = d_raw_azimuths[packet_id];

  // Calculate azimuth index for lookup table
  const uint32_t azimuth_idx = raw_azimuth % n_azimuths;

  // Lookup angle corrections
  const uint32_t lut_idx = azimuth_idx * config.n_channels + channel_id;
  const CudaAngleCorrectionData angle_data = angle_lut[lut_idx];

  // === FOV FILTERING (GPU) ===
  // Check if azimuth is within configured field of view
  const bool in_fov = cuda_angle_is_between(config.fov_min_rad, config.fov_max_rad,
                                            angle_data.azimuth_rad);
  if (!in_fov) return;

  // === OVERLAP/SCAN ASSIGNMENT (GPU) ===
  // Determine if this point belongs to current scan or output/next scan
  const uint32_t last_azimuth = d_last_azimuths[packet_id];
  uint8_t in_current_scan = 1;  // Default: belongs to current scan

  // Check if we're in the overlap region
  if (cuda_is_inside_overlap(last_azimuth, raw_azimuth,
                             config.timestamp_reset_angle_raw, config.emit_angle_raw,
                             config.n_azimuths_raw)) {
    // In overlap region, check if azimuth is between emit_angle and emit_angle + 20 degrees
    // 20 degrees in radians = 20 * pi / 180 = 0.349066
    constexpr float overlap_margin_rad = 0.349066f;
    const float overlap_end = config.scan_emit_angle_rad + overlap_margin_rad;
    if (cuda_angle_is_between(config.scan_emit_angle_rad, overlap_end, angle_data.azimuth_rad)) {
      in_current_scan = 0;  // Belongs to output/next scan
    }
  }

  // === DUAL-RETURN FILTERING (GPU) - OPTIMIZED ===
  // Matches CPU logic: keep only last of multiple identical/close points
  // This filtering happens BEFORE coordinate computation to avoid wasted work
  const uint32_t n_returns = d_n_returns[packet_id];

  // Last return is always kept - early exit (most common path for last return threads)
  // Single-return mode also skips filtering entirely
  if (return_id >= n_returns - 1) {
    goto compute_coordinates;
  }

  // Calculate base offset for this (packet_id, channel_id) group
  {
    const uint32_t group_base = packet_id * (config.n_channels * config.max_returns)
                               + channel_id * config.max_returns;
    const float threshold = config.dual_return_distance_threshold;

    // OPTIMIZATION: Special-case dual-return (most common mode)
    // Avoids loop overhead - direct comparison with the last return only
    if (n_returns == 2) {
      // return_id must be 0 here (we already checked return_id >= n_returns - 1)
      const uint32_t last_idx = group_base + 1;
      const uint16_t last_raw_distance = d_distances_ring[last_idx];
      const uint8_t last_reflectivity = d_reflectivities_ring[last_idx];

      // IDENTICAL check (same distance AND reflectivity)
      if (raw_distance == last_raw_distance && reflectivity == last_reflectivity) {
        return;  // Filtered: identical to last return
      }

      // Distance threshold check
      const float last_distance = static_cast<float>(last_raw_distance) * config.dis_unit;
      if (fabsf(distance - last_distance) < threshold) {
        return;  // Filtered: too close to last return
      }
    } else {
      // Triple-return or more (rare) - use loop
      for (uint32_t other_ret = 0; other_ret < n_returns; ++other_ret) {
        if (other_ret == return_id) continue;

        const uint16_t other_raw_distance = d_distances_ring[group_base + other_ret];
        const uint8_t other_reflectivity = d_reflectivities_ring[group_base + other_ret];

        // IDENTICAL check
        if (raw_distance == other_raw_distance && reflectivity == other_reflectivity) {
          return;  // Filtered
        }

        // Distance threshold check
        const float other_distance = static_cast<float>(other_raw_distance) * config.dis_unit;
        if (fabsf(distance - other_distance) < threshold) {
          return;  // Filtered
        }
      }
    }
  }

compute_coordinates:

  // Compute Cartesian coordinates
  const float xy_distance = distance * angle_data.cos_elevation;
  const float x = xy_distance * angle_data.sin_azimuth;
  const float y = xy_distance * angle_data.cos_azimuth;
  const float z = distance * angle_data.sin_elevation;

  // DETERMINISTIC OUTPUT: Write to global_tid position for consistent ordering
  // This ensures reproducible results regardless of thread scheduling
  // Invalid/filtered points will have distance=0 (from memset before kernel)
  // A separate compaction pass will pack valid points contiguously

  // Bounds check for output buffer (use global_tid as index)
  // max_output_points is set to sparse_buffer_size (n_entries * n_channels * max_returns)
  if (global_tid >= config.max_output_points) return;

  // Write output point to deterministic position
  CudaNebulaPoint & out_pt = output_points[global_tid];
  out_pt.x = x;
  out_pt.y = y;
  out_pt.z = z;
  out_pt.distance = distance;
  out_pt.azimuth = angle_data.azimuth_rad;
  out_pt.elevation = angle_data.elevation_rad;
  out_pt.intensity = static_cast<float>(reflectivity);
  out_pt.return_type = static_cast<uint8_t>(return_id);
  out_pt.channel = static_cast<uint16_t>(channel_id);
  out_pt.in_current_scan = in_current_scan;
  out_pt.entry_id = packet_id;  // Store block group ID for batched processing

  // Increment count atomically (for total valid point count)
  atomicAdd(output_count, 1);
}

// Constructor
HesaiCudaDecoder::HesaiCudaDecoder()
: d_angle_lut_(nullptr),
  n_azimuths_(0),
  n_channels_(0),
  initialized_(false)
{
}

// Destructor
HesaiCudaDecoder::~HesaiCudaDecoder()
{
  if (d_angle_lut_) {
    cudaFree(d_angle_lut_);
    d_angle_lut_ = nullptr;
  }
}

// Initialize decoder
bool HesaiCudaDecoder::initialize(size_t max_points, uint32_t n_channels)
{
  n_channels_ = n_channels;
  initialized_ = true;
  return true;
}

// Upload angle corrections to GPU
bool HesaiCudaDecoder::upload_angle_corrections(
  const std::vector<CudaAngleCorrectionData> & angle_lut,
  uint32_t n_azimuths,
  uint32_t n_channels)
{
  if (angle_lut.size() != n_azimuths * n_channels) {
    fprintf(stderr, "CUDA: Angle LUT size mismatch: %zu vs expected %u\n",
            angle_lut.size(), n_azimuths * n_channels);
    return false;
  }

  n_azimuths_ = n_azimuths;
  n_channels_ = n_channels;

  // Free existing allocation
  if (d_angle_lut_) {
    cudaFree(d_angle_lut_);
    d_angle_lut_ = nullptr;
  }

  // Allocate device memory
  const size_t lut_size = angle_lut.size() * sizeof(CudaAngleCorrectionData);
  cudaError_t err = cudaMalloc(&d_angle_lut_, lut_size);
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA: Failed to allocate angle LUT: %s\n", cudaGetErrorString(err));
    return false;
  }

  // Copy data to device
  err = cudaMemcpy(d_angle_lut_, angle_lut.data(), lut_size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA: Failed to upload angle LUT: %s\n", cudaGetErrorString(err));
    cudaFree(d_angle_lut_);
    d_angle_lut_ = nullptr;
    return false;
  }

  return true;
}

// Decode packet on GPU
bool HesaiCudaDecoder::decode_packet(
  const uint8_t * packet_data,
  size_t packet_size,
  const CudaDecoderConfig & config,
  CudaNebulaPoint * d_points,
  uint32_t * d_count,
  cudaStream_t stream)
{
  // This method is not used - actual kernel launch happens via launch_decode_hesai_packet
  // The HesaiDecoder class directly calls the C-linkage function for better performance
  return initialized_ && d_angle_lut_ != nullptr;
}

}  // namespace nebula::drivers::cuda

// C-linkage wrapper for launching kernel from hesai_decoder.hpp
// This allows direct access to device pointers without going through the class interface
extern "C" void launch_decode_hesai_packet(
  const uint16_t * d_distances,
  const uint8_t * d_reflectivities,
  const nebula::drivers::cuda::CudaAngleCorrectionData * d_angle_lut,
  const nebula::drivers::cuda::CudaDecoderConfig * d_config,
  nebula::drivers::cuda::CudaNebulaPoint * d_points,
  uint32_t * d_count,
  uint32_t n_azimuths,
  uint32_t raw_azimuth,
  cudaStream_t stream)
{
  // NOTE: d_count is NOT reset here - caller is responsible for resetting it
  // For batched mode, the caller resets once at the start, then accumulates across entries

  // Copy config from device to host to get parameters
  // This is necessary because we need config values for kernel launch
  nebula::drivers::cuda::CudaDecoderConfig config;
  cudaMemcpyAsync(&config, d_config, sizeof(nebula::drivers::cuda::CudaDecoderConfig),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);  // Ensure config is available

  // Calculate optimal grid dimensions
  // Use 128 threads per block - good balance between occupancy and register usage
  const uint32_t threads_per_block = 128;
  const uint32_t n_blocks_x = (config.n_channels + threads_per_block - 1) / threads_per_block;
  const uint32_t n_blocks_y = config.n_blocks;  // One block dimension per return

  dim3 grid(n_blocks_x, n_blocks_y);
  dim3 block(threads_per_block);

  // Launch kernel with optimal configuration
  nebula::drivers::cuda::decode_hesai_packet_kernel<<<grid, block, 0, stream>>>(
    d_distances,
    d_reflectivities,
    d_angle_lut,
    config,
    d_points,
    d_count,
    n_azimuths,
    raw_azimuth);

  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
  }
}

// C-linkage wrapper for launching batched kernel (processes entire scan in one launch)
extern "C" void launch_decode_hesai_scan_batch(
  const uint16_t * d_distances_ring,
  const uint8_t * d_reflectivities_ring,
  const uint32_t * d_raw_azimuths,
  const uint32_t * d_n_returns,
  const uint32_t * d_last_azimuths,  // Per-entry last_azimuth for overlap check
  const nebula::drivers::cuda::CudaAngleCorrectionData * d_angle_lut,
  const nebula::drivers::cuda::CudaDecoderConfig * d_config,
  nebula::drivers::cuda::CudaNebulaPoint * d_points,
  uint32_t * d_count,
  uint32_t n_azimuths,
  uint32_t n_packets,
  cudaStream_t stream)
{
  // Copy config to get parameters
  nebula::drivers::cuda::CudaDecoderConfig config;
  cudaMemcpyAsync(&config, d_config, sizeof(nebula::drivers::cuda::CudaDecoderConfig),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);  // Ensure config is available

  // Calculate grid dimensions for batched processing
  // Total work: n_packets * n_channels * max_returns
  const uint32_t total_work = n_packets * config.n_channels * config.max_returns;
  const uint32_t threads_per_block = 256;  // Larger block size for better occupancy
  const uint32_t n_blocks = (total_work + threads_per_block - 1) / threads_per_block;

  dim3 grid(n_blocks);
  dim3 block(threads_per_block);

  // Launch batched kernel - processes entire scan in one call
  // Now includes FOV filtering and overlap/scan assignment on GPU
  nebula::drivers::cuda::decode_hesai_scan_batch_kernel<<<grid, block, 0, stream>>>(
    d_distances_ring,
    d_reflectivities_ring,
    d_raw_azimuths,
    d_n_returns,
    d_last_azimuths,
    d_angle_lut,
    config,
    d_points,
    d_count,
    n_azimuths,
    n_packets);

  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA batched kernel launch failed: %s\n", cudaGetErrorString(err));
  }

  // COMPACTION PASS: Pack valid points (distance > 0) contiguously
  // This is done on CPU side after D2H copy in hesai_decoder.hpp
  // The kernel outputs sparse data, CPU compacts while copying to point cloud
}

// =============================================================================
// COMPACTION KERNEL: Pack sparse point buffer into contiguous output
// Used after batched decode to remove gaps from deterministic indexing
// =============================================================================

/// @brief Kernel to compact sparse point buffer - counts and compacts valid points
/// Points with distance == 0 are considered invalid/filtered
__global__ void compact_points_kernel(
    const nebula::drivers::cuda::CudaNebulaPoint* __restrict__ d_input,
    nebula::drivers::cuda::CudaNebulaPoint* __restrict__ d_output,
    uint32_t* __restrict__ d_output_count,
    const uint32_t input_size)
{
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= input_size) return;

  const nebula::drivers::cuda::CudaNebulaPoint& pt = d_input[idx];

  // Check if point is valid (distance > 0)
  if (pt.distance > 0.0f) {
    // Atomically allocate output slot
    const uint32_t out_idx = atomicAdd(d_output_count, 1);
    d_output[out_idx] = pt;
  }
}

/// @brief Launch compaction kernel to pack sparse points
/// @param d_sparse_input Sparse input buffer (from batched decode)
/// @param d_compact_output Compacted output buffer
/// @param d_output_count Output count (reset before call)
/// @param input_size Number of potential points (total_work from batched kernel)
/// @param stream CUDA stream
extern "C" void launch_compact_points(
    const nebula::drivers::cuda::CudaNebulaPoint* d_sparse_input,
    nebula::drivers::cuda::CudaNebulaPoint* d_compact_output,
    uint32_t* d_output_count,
    uint32_t input_size,
    cudaStream_t stream)
{
  const uint32_t threads_per_block = 256;
  const uint32_t n_blocks = (input_size + threads_per_block - 1) / threads_per_block;

  compact_points_kernel<<<n_blocks, threads_per_block, 0, stream>>>(
      d_sparse_input, d_compact_output, d_output_count, input_size);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA compact kernel launch failed: %s\n", cudaGetErrorString(err));
  }
}

// =============================================================================
// FORMAT CONVERSION KERNEL FOR CUDA_BLACKBOARD INTEGRATION
// Converts CudaNebulaPoint[] to PointCloud2-compatible byte layout (PointXYZIRCAEDT)
// =============================================================================

namespace nebula::drivers::cuda
{

// PointXYZIRCAEDT field offsets (32 bytes total)
constexpr uint32_t PC2_OFFSET_X = 0;
constexpr uint32_t PC2_OFFSET_Y = 4;
constexpr uint32_t PC2_OFFSET_Z = 8;
constexpr uint32_t PC2_OFFSET_INTENSITY = 12;
constexpr uint32_t PC2_OFFSET_RETURN_TYPE = 13;
constexpr uint32_t PC2_OFFSET_CHANNEL = 14;
constexpr uint32_t PC2_OFFSET_AZIMUTH = 16;
constexpr uint32_t PC2_OFFSET_ELEVATION = 20;
constexpr uint32_t PC2_OFFSET_DISTANCE = 24;
constexpr uint32_t PC2_OFFSET_TIME_STAMP = 28;
constexpr uint32_t PC2_POINT_STEP = 32;

/// @brief CUDA kernel to convert CudaNebulaPoint array to PointCloud2 byte format
/// This kernel:
/// 1. Filters points based on in_current_scan flag
/// 2. Converts float intensity to uint8_t
/// 3. Reorders fields to match PointXYZIRCAEDT layout
/// 4. Computes time_stamp (placeholder - set to 0, computed on CPU or passed in)
///
/// @param d_input Input CudaNebulaPoint array from decoder
/// @param d_output Output byte buffer for PointCloud2 data
/// @param d_output_count Atomic counter for output points (for compaction)
/// @param input_count Number of input points
/// @param filter_current_scan If true, only include points with in_current_scan=1
__global__ void convert_to_pointcloud2_kernel(
    const CudaNebulaPoint* __restrict__ d_input,
    uint8_t* __restrict__ d_output,
    uint32_t* __restrict__ d_output_count,
    const uint32_t input_count,
    const bool filter_current_scan)
{
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= input_count) return;

  const CudaNebulaPoint& pt = d_input[idx];

  // Filter points not in current scan (if filtering enabled)
  if (filter_current_scan && !pt.in_current_scan) {
    return;
  }

  // Atomically allocate output slot
  const uint32_t out_idx = atomicAdd(d_output_count, 1);
  uint8_t* out = d_output + out_idx * PC2_POINT_STEP;

  // Write fields in PointXYZIRCAEDT order
  *reinterpret_cast<float*>(out + PC2_OFFSET_X) = pt.x;
  *reinterpret_cast<float*>(out + PC2_OFFSET_Y) = pt.y;
  *reinterpret_cast<float*>(out + PC2_OFFSET_Z) = pt.z;

  // Convert float intensity to uint8_t with clamping
  out[PC2_OFFSET_INTENSITY] = static_cast<uint8_t>(
      fminf(fmaxf(pt.intensity, 0.0f), 255.0f));

  out[PC2_OFFSET_RETURN_TYPE] = pt.return_type;
  *reinterpret_cast<uint16_t*>(out + PC2_OFFSET_CHANNEL) = pt.channel;
  *reinterpret_cast<float*>(out + PC2_OFFSET_AZIMUTH) = pt.azimuth;
  *reinterpret_cast<float*>(out + PC2_OFFSET_ELEVATION) = pt.elevation;
  *reinterpret_cast<float*>(out + PC2_OFFSET_DISTANCE) = pt.distance;

  // Time stamp: set to 0 for now (can be computed if packet timestamps are available)
  // For full accuracy, this would need per-point timestamp data from the decoder
  *reinterpret_cast<uint32_t*>(out + PC2_OFFSET_TIME_STAMP) = 0;
}

/// @brief Alternative kernel that preserves point order (no compaction)
/// Faster but requires pre-filtering or post-processing to remove invalid points
__global__ void convert_to_pointcloud2_ordered_kernel(
    const CudaNebulaPoint* __restrict__ d_input,
    uint8_t* __restrict__ d_output,
    uint8_t* __restrict__ d_valid_mask,  // Optional: 1 if valid, 0 if filtered
    const uint32_t input_count,
    const bool filter_current_scan)
{
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= input_count) return;

  const CudaNebulaPoint& pt = d_input[idx];
  uint8_t* out = d_output + idx * PC2_POINT_STEP;

  // Check validity
  bool valid = !filter_current_scan || pt.in_current_scan;

  if (d_valid_mask) {
    d_valid_mask[idx] = valid ? 1 : 0;
  }

  if (!valid) {
    // Zero out the point (or could skip writing)
    memset(out, 0, PC2_POINT_STEP);
    return;
  }

  // Write fields in PointXYZIRCAEDT order
  *reinterpret_cast<float*>(out + PC2_OFFSET_X) = pt.x;
  *reinterpret_cast<float*>(out + PC2_OFFSET_Y) = pt.y;
  *reinterpret_cast<float*>(out + PC2_OFFSET_Z) = pt.z;
  out[PC2_OFFSET_INTENSITY] = static_cast<uint8_t>(
      fminf(fmaxf(pt.intensity, 0.0f), 255.0f));
  out[PC2_OFFSET_RETURN_TYPE] = pt.return_type;
  *reinterpret_cast<uint16_t*>(out + PC2_OFFSET_CHANNEL) = pt.channel;
  *reinterpret_cast<float*>(out + PC2_OFFSET_AZIMUTH) = pt.azimuth;
  *reinterpret_cast<float*>(out + PC2_OFFSET_ELEVATION) = pt.elevation;
  *reinterpret_cast<float*>(out + PC2_OFFSET_DISTANCE) = pt.distance;
  *reinterpret_cast<uint32_t*>(out + PC2_OFFSET_TIME_STAMP) = 0;
}

}  // namespace nebula::drivers::cuda

// C-linkage wrapper for format conversion kernel
extern "C" void launch_convert_to_pointcloud2(
    const nebula::drivers::cuda::CudaNebulaPoint* d_input,
    uint8_t* d_output,
    uint32_t* d_output_count,
    uint32_t input_count,
    bool filter_current_scan,
    cudaStream_t stream)
{
  if (input_count == 0) return;

  // Reset output count
  cudaMemsetAsync(d_output_count, 0, sizeof(uint32_t), stream);

  // Launch conversion kernel
  const uint32_t threads_per_block = 256;
  const uint32_t n_blocks = (input_count + threads_per_block - 1) / threads_per_block;

  nebula::drivers::cuda::convert_to_pointcloud2_kernel<<<n_blocks, threads_per_block, 0, stream>>>(
      d_input,
      d_output,
      d_output_count,
      input_count,
      filter_current_scan);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA convert_to_pointcloud2 kernel failed: %s\n", cudaGetErrorString(err));
  }
}

// C-linkage wrapper for ordered conversion (no compaction)
extern "C" void launch_convert_to_pointcloud2_ordered(
    const nebula::drivers::cuda::CudaNebulaPoint* d_input,
    uint8_t* d_output,
    uint8_t* d_valid_mask,
    uint32_t input_count,
    bool filter_current_scan,
    cudaStream_t stream)
{
  if (input_count == 0) return;

  const uint32_t threads_per_block = 256;
  const uint32_t n_blocks = (input_count + threads_per_block - 1) / threads_per_block;

  nebula::drivers::cuda::convert_to_pointcloud2_ordered_kernel<<<n_blocks, threads_per_block, 0, stream>>>(
      d_input,
      d_output,
      d_valid_mask,
      input_count,
      filter_current_scan);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA convert_to_pointcloud2_ordered kernel failed: %s\n", cudaGetErrorString(err));
  }
}
