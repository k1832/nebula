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

#include "nebula_hesai_decoders/cuda/hesai_cuda_decoder.hpp"

#include <cuda_runtime.h>

namespace nebula::drivers::cuda
{

// Device function: Check if angle is between start and end (handles wrap-around)
__device__ __forceinline__ bool cuda_angle_is_between(float start, float end, float angle)
{
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
  start = start % max_angle;
  end = end % max_angle;
  angle = angle % max_angle;

  if (start <= end) {
    return (start <= angle && angle <= end);
  } else {
    return (angle <= end || start <= angle);
  }
}

// Device function: Check if we're inside the overlap region (single-frame version)
__device__ __forceinline__ bool cuda_is_inside_overlap(
  uint32_t last_azimuth, uint32_t current_azimuth,
  uint32_t timestamp_reset_angle, uint32_t emit_angle, uint32_t max_angle)
{
  return cuda_angle_is_between_raw(timestamp_reset_angle, emit_angle, current_azimuth, max_angle) ||
         cuda_angle_is_between_raw(timestamp_reset_angle, emit_angle, last_azimuth, max_angle);
}

// Device function: Check if we're inside the overlap region for multi-frame sensors (AT128)
__device__ __forceinline__ bool cuda_is_inside_overlap_multiframe(
  uint32_t last_azimuth, uint32_t current_azimuth,
  const CudaFrameAngleInfo* frame_angles, uint32_t n_frames, uint32_t max_angle)
{
  for (uint32_t i = 0; i < n_frames; ++i) {
    if (cuda_angle_is_between_raw(frame_angles[i].timestamp_reset, frame_angles[i].scan_emit,
                                  current_azimuth, max_angle) ||
        cuda_angle_is_between_raw(frame_angles[i].timestamp_reset, frame_angles[i].scan_emit,
                                  last_azimuth, max_angle)) {
      return true;
    }
  }
  return false;
}

/// @brief Batched kernel for processing an entire scan in one launch
__global__ void decode_hesai_scan_batch_kernel(
  const uint16_t * __restrict__ d_distances_batch,
  const uint8_t * __restrict__ d_reflectivities_batch,
  const uint32_t * __restrict__ d_raw_azimuths,
  const uint32_t * __restrict__ d_n_returns,
  const uint32_t * __restrict__ d_last_azimuths,
  const CudaAngleCorrectionData * __restrict__ angle_lut,
  const CudaDecoderConfig config,
  CudaNebulaPoint * __restrict__ output_points,
  uint32_t * __restrict__ output_count,
  uint32_t n_azimuths,
  uint32_t n_packets)
{
  const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;

  const uint32_t total_work = n_packets * config.n_channels * config.max_returns;
  if (global_tid >= total_work) return;

  const uint32_t packet_id = global_tid / (config.n_channels * config.max_returns);
  const uint32_t channel_id = (global_tid / config.max_returns) % config.n_channels;
  const uint32_t return_id = global_tid % config.max_returns;

  if (return_id >= d_n_returns[packet_id]) return;

  const uint32_t data_idx = packet_id * (config.n_channels * config.max_returns)
                           + channel_id * config.max_returns + return_id;

  const uint16_t raw_distance = d_distances_batch[data_idx];
  const uint8_t reflectivity = d_reflectivities_batch[data_idx];

  if (raw_distance == 0) return;

  const float distance = static_cast<float>(raw_distance) * config.dis_unit;

  if (distance < config.min_range || distance > config.max_range) return;
  if (distance < config.sensor_min_range || distance > config.sensor_max_range) return;

  const uint32_t raw_azimuth = d_raw_azimuths[packet_id];
  const uint32_t azimuth_idx = (raw_azimuth / config.azimuth_scale) % n_azimuths;
  const uint32_t lut_idx = azimuth_idx * config.n_channels + channel_id;
  const CudaAngleCorrectionData angle_data = angle_lut[lut_idx];

  // FOV filtering
  const bool in_fov = cuda_angle_is_between(config.fov_min_rad, config.fov_max_rad,
                                            angle_data.azimuth_rad);
  if (!in_fov) return;

  // Overlap/scan assignment
  const uint32_t last_azimuth = d_last_azimuths[packet_id];
  uint8_t in_current_scan = 1;

  bool is_in_overlap = false;
  if (config.n_frames > 1) {
    is_in_overlap = cuda_is_inside_overlap_multiframe(
      last_azimuth, raw_azimuth, config.frame_angles, config.n_frames, config.n_azimuths_raw);
  } else {
    is_in_overlap = cuda_is_inside_overlap(
      last_azimuth, raw_azimuth, config.timestamp_reset_angle_raw, config.emit_angle_raw,
      config.n_azimuths_raw);
  }

  if (is_in_overlap) {
    constexpr float overlap_margin_rad = 0.349066f;  // 20 degrees
    const float overlap_end = config.scan_emit_angle_rad + overlap_margin_rad;
    if (cuda_angle_is_between(config.scan_emit_angle_rad, overlap_end, angle_data.azimuth_rad)) {
      in_current_scan = 0;
    }
  }

  // Dual-return filtering
  const uint32_t n_returns = d_n_returns[packet_id];

  if (return_id >= n_returns - 1) {
    goto compute_coordinates;
  }

  {
    const uint32_t group_base = packet_id * (config.n_channels * config.max_returns)
                               + channel_id * config.max_returns;
    const float threshold = config.dual_return_distance_threshold;

    if (n_returns == 2) {
      const uint32_t last_idx = group_base + 1;
      const uint16_t last_raw_distance = d_distances_batch[last_idx];
      const uint8_t last_reflectivity = d_reflectivities_batch[last_idx];

      if (raw_distance == last_raw_distance && reflectivity == last_reflectivity) {
        return;
      }

      const float last_distance = static_cast<float>(last_raw_distance) * config.dis_unit;
      if (fabsf(distance - last_distance) < threshold) {
        return;
      }
    } else {
      for (uint32_t other_ret = 0; other_ret < n_returns; ++other_ret) {
        if (other_ret == return_id) continue;

        const uint16_t other_raw_distance = d_distances_batch[group_base + other_ret];
        const uint8_t other_reflectivity = d_reflectivities_batch[group_base + other_ret];

        if (raw_distance == other_raw_distance && reflectivity == other_reflectivity) {
          return;
        }

        const float other_distance = static_cast<float>(other_raw_distance) * config.dis_unit;
        if (fabsf(distance - other_distance) < threshold) {
          return;
        }
      }
    }
  }

compute_coordinates:

  const float xy_distance = distance * angle_data.cos_elevation;
  const float x = xy_distance * angle_data.sin_azimuth;
  const float y = xy_distance * angle_data.cos_azimuth;
  const float z = distance * angle_data.sin_elevation;

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
  out_pt.entry_id = packet_id;

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

bool HesaiCudaDecoder::initialize(size_t max_points, uint32_t n_channels)
{
  n_channels_ = n_channels;
  initialized_ = true;
  return true;
}

bool HesaiCudaDecoder::upload_angle_corrections(
  const std::vector<CudaAngleCorrectionData> & angle_lut,
  uint32_t n_azimuths,
  uint32_t n_channels)
{
  if (angle_lut.size() != n_azimuths * n_channels) {
    return false;
  }

  n_azimuths_ = n_azimuths;
  n_channels_ = n_channels;

  if (d_angle_lut_) {
    cudaFree(d_angle_lut_);
    d_angle_lut_ = nullptr;
  }

  const size_t lut_size = angle_lut.size() * sizeof(CudaAngleCorrectionData);
  cudaError_t err = cudaMalloc(&d_angle_lut_, lut_size);
  if (err != cudaSuccess) {
    return false;
  }

  err = cudaMemcpy(d_angle_lut_, angle_lut.data(), lut_size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cudaFree(d_angle_lut_);
    d_angle_lut_ = nullptr;
    return false;
  }

  return true;
}

}  // namespace nebula::drivers::cuda

// C-linkage wrapper for batched kernel
extern "C" bool launch_decode_hesai_scan_batch(
  const uint16_t * d_distances_batch,
  const uint8_t * d_reflectivities_batch,
  const uint32_t * d_raw_azimuths,
  const uint32_t * d_n_returns,
  const uint32_t * d_last_azimuths,
  const nebula::drivers::cuda::CudaAngleCorrectionData * d_angle_lut,
  const nebula::drivers::cuda::CudaDecoderConfig & config,
  nebula::drivers::cuda::CudaNebulaPoint * d_points,
  uint32_t * d_count,
  uint32_t n_azimuths,
  uint32_t n_packets,
  cudaStream_t stream)
{
  const uint32_t total_work = n_packets * config.n_channels * config.max_returns;
  const uint32_t threads_per_block = 256;
  const uint32_t n_blocks = (total_work + threads_per_block - 1) / threads_per_block;

  dim3 grid(n_blocks);
  dim3 block(threads_per_block);

  nebula::drivers::cuda::decode_hesai_scan_batch_kernel<<<grid, block, 0, stream>>>(
    d_distances_batch, d_reflectivities_batch, d_raw_azimuths, d_n_returns, d_last_azimuths,
    d_angle_lut, config, d_points, d_count, n_azimuths, n_packets);

  return cudaGetLastError() == cudaSuccess;
}
