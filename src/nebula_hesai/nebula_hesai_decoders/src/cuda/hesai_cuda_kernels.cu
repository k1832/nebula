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

#include "nebula_hesai_decoders/decoders/cuda/hesai_cuda_decoder.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <cstdio>

namespace nebula::drivers::cuda
{

// CUDA error checking macro
#define CUDA_CHECK(call)                                                              \
  do {                                                                                \
    cudaError_t err = call;                                                           \
    if (err != cudaSuccess) {                                                         \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                \
              cudaGetErrorString(err));                                               \
    }                                                                                 \
  } while (0)

/// @brief Kernel to decode Hesai packet and convert to points
/// Each thread processes one channel for one block
__global__ void decodeHesaiPacketKernel(
  const uint8_t * __restrict__ packet_data,
  const CudaDecoderConfig * __restrict__ config,
  const CudaAngleCorrectionData * __restrict__ angle_corrections,
  uint32_t n_azimuths,
  uint32_t n_channels,
  CudaNebulaPoint * __restrict__ output_points,
  uint32_t * __restrict__ output_count,
  uint32_t raw_azimuth,
  uint16_t * __restrict__ distances,
  uint8_t * __restrict__ reflectivities)
{
  // Each thread handles one channel
  const uint32_t channel_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (channel_id >= config->n_channels) {
    return;
  }

  // Process each return block
  for (uint32_t block_offset = 0; block_offset < config->max_returns; ++block_offset) {
    const uint32_t point_idx = channel_id * config->max_returns + block_offset;

    // Get distance and reflectivity from pre-extracted data
    const uint16_t raw_distance = distances[point_idx];
    const uint8_t reflectivity = reflectivities[point_idx];

    // Skip zero distance points
    if (raw_distance == 0) {
      continue;
    }

    // Calculate actual distance
    const float distance = static_cast<float>(raw_distance) * config->dis_unit;

    // Range check
    if (distance < config->sensor_min_range || distance > config->sensor_max_range ||
        distance < config->min_range || distance > config->max_range) {
      continue;
    }

    // Get angle correction data from lookup table
    // Index into the flattened 2D array: [azimuth][channel]
    const uint32_t azimuth_idx = raw_azimuth % n_azimuths;
    const uint32_t angle_idx = azimuth_idx * n_channels + channel_id;
    const CudaAngleCorrectionData & angle_data = angle_corrections[angle_idx];

    // Check FOV
    const float azimuth_rad = angle_data.azimuth_rad;

    // Simple FOV check (can be improved with proper angle wrapping)
    if (azimuth_rad < config->fov_min_rad || azimuth_rad > config->fov_max_rad) {
      continue;
    }

    // Calculate point coordinates
    const float xy_distance = distance * angle_data.cos_elevation;
    const float x = xy_distance * angle_data.sin_azimuth;
    const float y = xy_distance * angle_data.cos_azimuth;
    const float z = distance * angle_data.sin_elevation;

    // Atomically allocate output slot
    const uint32_t out_idx = atomicAdd(output_count, 1);

    // Write output point
    CudaNebulaPoint & point = output_points[out_idx];
    point.x = x;
    point.y = y;
    point.z = z;
    point.distance = distance;
    point.azimuth = azimuth_rad;
    point.elevation = angle_data.elevation_rad;
    point.intensity = reflectivity;
    point.return_type = static_cast<uint8_t>(block_offset);  // Simplified
    point.channel = static_cast<uint16_t>(channel_id);
    point.time_stamp = 0;  // TODO: Calculate proper timestamp
  }
}

/// @brief Simple kernel to extract distances and reflectivities from packet
/// This is a simplified version - real implementation needs sensor-specific parsing
__global__ void extractPacketDataKernel(
  const uint8_t * __restrict__ packet_data,
  uint32_t n_channels,
  uint32_t n_blocks,
  uint32_t block_size,
  uint32_t unit_size,
  uint16_t * __restrict__ distances,
  uint8_t * __restrict__ reflectivities,
  uint32_t * __restrict__ azimuth)
{
  // Thread 0 extracts azimuth from first block
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // Simplified: actual offset depends on sensor model
    *azimuth = *reinterpret_cast<const uint16_t *>(packet_data + 2);
  }

  // Each thread extracts one unit's data
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t total_units = n_channels * n_blocks;

  if (idx < total_units) {
    const uint32_t block_id = idx / n_channels;
    const uint32_t channel_id = idx % n_channels;

    // Calculate offset in packet (simplified, needs sensor-specific offsets)
    const uint32_t block_offset = 4 + block_id * block_size;  // Skip header
    const uint32_t unit_offset = block_offset + 4 + channel_id * unit_size;  // Skip azimuth

    if (unit_offset + unit_size <= 1500) {  // Bounds check
      distances[idx] = *reinterpret_cast<const uint16_t *>(packet_data + unit_offset);
      reflectivities[idx] = packet_data[unit_offset + 2];
    }
  }
}

// Implementation of HesaiCudaDecoder class

HesaiCudaDecoder::HesaiCudaDecoder() = default;

HesaiCudaDecoder::~HesaiCudaDecoder()
{
  if (d_output_points_) cudaFree(d_output_points_);
  if (d_output_count_) cudaFree(d_output_count_);
  if (d_packet_buffer_) cudaFree(d_packet_buffer_);
  if (d_angle_corrections_) cudaFree(d_angle_corrections_);
  if (h_pinned_packet_buffer_) cudaFreeHost(h_pinned_packet_buffer_);
}

bool HesaiCudaDecoder::initialize(size_t max_points, uint32_t n_channels)
{
  if (initialized_) {
    return true;
  }

  max_points_ = max_points;
  n_channels_ = n_channels;
  packet_buffer_size_ = 2048;  // Max UDP packet size

  // Allocate device memory
  CUDA_CHECK(cudaMalloc(&d_output_points_, max_points_ * sizeof(CudaNebulaPoint)));
  CUDA_CHECK(cudaMalloc(&d_output_count_, sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_packet_buffer_, packet_buffer_size_));

  // Allocate pinned host memory for async transfers
  CUDA_CHECK(cudaMallocHost(&h_pinned_packet_buffer_, packet_buffer_size_));

  initialized_ = true;
  return true;
}

void HesaiCudaDecoder::upload_angle_corrections(
  const std::vector<CudaAngleCorrectionData> & angle_data,
  uint32_t n_azimuths,
  uint32_t n_channels)
{
  n_azimuths_ = n_azimuths;
  n_channels_ = n_channels;

  const size_t size = angle_data.size() * sizeof(CudaAngleCorrectionData);

  if (d_angle_corrections_) {
    cudaFree(d_angle_corrections_);
  }

  CUDA_CHECK(cudaMalloc(&d_angle_corrections_, size));
  CUDA_CHECK(cudaMemcpy(d_angle_corrections_, angle_data.data(), size, cudaMemcpyHostToDevice));
}

void HesaiCudaDecoder::decode_packet(
  const uint8_t * packet_data,
  size_t packet_size,
  const CudaDecoderConfig & config,
  CudaNebulaPoint * output_points,
  uint32_t * output_count,
  cudaStream_t stream)
{
  // Copy packet to pinned memory then to device
  memcpy(h_pinned_packet_buffer_, packet_data, packet_size);
  CUDA_CHECK(cudaMemcpyAsync(d_packet_buffer_, h_pinned_packet_buffer_, packet_size,
                             cudaMemcpyHostToDevice, stream));

  // Reset output count
  CUDA_CHECK(cudaMemsetAsync(output_count, 0, sizeof(uint32_t), stream));

  // Allocate temporary device memory for config
  CudaDecoderConfig * d_config;
  CUDA_CHECK(cudaMalloc(&d_config, sizeof(CudaDecoderConfig)));
  CUDA_CHECK(cudaMemcpyAsync(d_config, &config, sizeof(CudaDecoderConfig),
                             cudaMemcpyHostToDevice, stream));

  // Allocate temporary buffers for extracted data
  uint16_t * d_distances;
  uint8_t * d_reflectivities;
  uint32_t * d_azimuth;
  const size_t n_units = config.n_channels * config.max_returns;
  CUDA_CHECK(cudaMalloc(&d_distances, n_units * sizeof(uint16_t)));
  CUDA_CHECK(cudaMalloc(&d_reflectivities, n_units * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(&d_azimuth, sizeof(uint32_t)));

  // Extract packet data
  const int extract_threads = 256;
  const int extract_blocks = (n_units + extract_threads - 1) / extract_threads;
  extractPacketDataKernel<<<extract_blocks, extract_threads, 0, stream>>>(
    d_packet_buffer_, config.n_channels, config.n_blocks,
    64,  // block_size (simplified)
    4,   // unit_size (simplified)
    d_distances, d_reflectivities, d_azimuth);

  // Copy azimuth back (needed for angle lookup)
  uint32_t raw_azimuth;
  CUDA_CHECK(cudaMemcpyAsync(&raw_azimuth, d_azimuth, sizeof(uint32_t),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Launch decode kernel
  const int threads_per_block = 128;
  const int blocks = (config.n_channels + threads_per_block - 1) / threads_per_block;

  decodeHesaiPacketKernel<<<blocks, threads_per_block, 0, stream>>>(
    d_packet_buffer_, d_config, d_angle_corrections_,
    n_azimuths_, n_channels_,
    output_points, output_count,
    raw_azimuth, d_distances, d_reflectivities);

  // Cleanup temporary allocations
  cudaFree(d_config);
  cudaFree(d_distances);
  cudaFree(d_reflectivities);
  cudaFree(d_azimuth);
}

size_t HesaiCudaDecoder::copy_results_to_host(
  const CudaNebulaPoint * d_points,
  const uint32_t * d_count,
  CudaNebulaPoint * h_points,
  size_t max_points,
  cudaStream_t stream)
{
  uint32_t count;
  CUDA_CHECK(cudaMemcpyAsync(&count, d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  const size_t copy_count = (count < max_points) ? count : max_points;

  if (copy_count > 0) {
    CUDA_CHECK(cudaMemcpyAsync(h_points, d_points, copy_count * sizeof(CudaNebulaPoint),
                               cudaMemcpyDeviceToHost, stream));
  }

  return copy_count;
}

void HesaiCudaDecoder::synchronize(cudaStream_t stream)
{
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

void launch_decode_hesai_packet_kernel(
  const uint8_t * d_packet,
  size_t packet_size,
  const CudaDecoderConfig * d_config,
  const CudaAngleCorrectionData * d_angle_corrections,
  uint32_t n_azimuths,
  uint32_t n_channels,
  CudaNebulaPoint * d_output_points,
  uint32_t * d_output_count,
  cudaStream_t stream)
{
  // This is a wrapper for external calls
  // The actual kernel launch is done in decode_packet()
}

}  // namespace nebula::drivers::cuda
