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

  // Calculate linear index: channel_id * n_blocks + block_id
  // This layout ensures coalesced reads when threads in a warp process consecutive channels
  const uint32_t data_idx = channel_id * config.n_blocks + block_id;

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
  // Zero out the output count asynchronously
  cudaMemsetAsync(d_count, 0, sizeof(uint32_t), stream);

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
