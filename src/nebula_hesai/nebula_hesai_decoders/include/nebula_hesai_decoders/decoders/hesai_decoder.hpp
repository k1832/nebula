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

#include "nebula_core_decoders/angles.hpp"
#include "nebula_core_decoders/point_filters/blockage_mask.hpp"
#include "nebula_core_decoders/point_filters/downsample_mask.hpp"
#include "nebula_hesai_decoders/decoders/angle_corrector.hpp"
#include "nebula_hesai_decoders/decoders/functional_safety.hpp"
#include "nebula_hesai_decoders/decoders/hesai_packet.hpp"
#include "nebula_hesai_decoders/decoders/hesai_scan_decoder.hpp"
#include "nebula_hesai_decoders/decoders/packet_loss_detector.hpp"

#ifdef NEBULA_CUDA_ENABLED
#include "nebula_hesai_decoders/cuda/hesai_cuda_decoder.hpp"

// C-linkage kernel launcher declaration
extern "C" void launch_decode_hesai_packet(
  const uint16_t * d_distances,
  const uint8_t * d_reflectivities,
  const nebula::drivers::cuda::CudaAngleCorrectionData * d_angle_lut,
  const nebula::drivers::cuda::CudaDecoderConfig * d_config,
  nebula::drivers::cuda::CudaNebulaPoint * d_points,
  uint32_t * d_count,
  uint32_t n_azimuths,
  uint32_t raw_azimuth,
  cudaStream_t stream);

// C-linkage batched kernel launcher declaration (processes entire scan)
extern "C" void launch_decode_hesai_scan_batch(
  const uint16_t * d_distances_ring,
  const uint8_t * d_reflectivities_ring,
  const uint32_t * d_raw_azimuths,
  const uint32_t * d_n_returns,
  const nebula::drivers::cuda::CudaAngleCorrectionData * d_angle_lut,
  const nebula::drivers::cuda::CudaDecoderConfig * d_config,
  nebula::drivers::cuda::CudaNebulaPoint * d_points,
  uint32_t * d_count,
  uint32_t n_azimuths,
  uint32_t n_packets,
  cudaStream_t stream);
#endif

#include <nebula_core_common/loggers/logger.hpp>
#include <nebula_core_common/nebula_common.hpp>
#include <nebula_core_common/point_types.hpp>
#include <nebula_core_common/util/stopwatch.hpp>
#include <nebula_hesai_common/hesai_common.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace nebula::drivers
{

template <typename SensorT>
class HesaiDecoder : public HesaiScanDecoder
{
private:
  struct ScanCutAngles
  {
    float fov_min;
    float fov_max;
    float scan_emit_angle;
  };

  struct DecodeFrame
  {
    NebulaPointCloudPtr pointcloud;
    uint64_t scan_timestamp_ns{0};
    std::optional<point_filters::BlockageMask> blockage_mask;
  };

  /// @brief Configuration for this decoder
  const std::shared_ptr<const drivers::HesaiSensorConfiguration> sensor_configuration_;

  /// @brief The sensor definition, used for return mode and time offset handling
  SensorT sensor_{};

  /// @brief A function that is called on each decoded pointcloud frame
  pointcloud_callback_t pointcloud_callback_;

  /// @brief Decodes azimuth/elevation angles given calibration/correction data
  typename SensorT::angle_corrector_t angle_corrector_;

  /// @brief Decodes functional safety data for supported sensors
  std::shared_ptr<FunctionalSafetyDecoderTypedBase<typename SensorT::packet_t>>
    functional_safety_decoder_;

  std::shared_ptr<PacketLossDetectorTypedBase<typename SensorT::packet_t>> packet_loss_detector_;

  /// @brief The last decoded packet
  typename SensorT::packet_t packet_;

  ScanCutAngles scan_cut_angles_;
  uint32_t last_azimuth_ = 0;

  std::shared_ptr<loggers::Logger> logger_;

  /// @brief For each channel, its firing offset relative to the block in nanoseconds
  std::array<int, SensorT::packet_t::n_channels> channel_firing_offset_ns_;
  /// @brief For each return mode, the firing offset of each block relative to its packet in
  /// nanoseconds
  std::array<std::array<int, SensorT::packet_t::n_blocks>, SensorT::packet_t::max_returns>
    block_firing_offset_ns_;

  std::optional<point_filters::DownsampleMaskFilter> mask_filter_;

  std::shared_ptr<point_filters::BlockageMaskPlugin> blockage_mask_plugin_;

  /// @brief Decoded data of the frame currently being decoded to
  DecodeFrame decode_frame_;
  /// @brief Decoded data of the frame currently being output
  DecodeFrame output_frame_;

#ifdef NEBULA_CUDA_ENABLED
  /// @brief CUDA decoder for GPU-accelerated point cloud processing
  std::unique_ptr<cuda::HesaiCudaDecoder> cuda_decoder_;
  /// @brief CUDA stream for async operations
  cudaStream_t cuda_stream_ = nullptr;
  /// @brief Whether CUDA decoding is enabled and initialized
  bool cuda_enabled_ = false;
  /// @brief Number of azimuth divisions for angle lookup table
  static constexpr uint32_t cuda_n_azimuths_ = 36000;  // 0.01 degree resolution
  /// @brief Device memory for output points
  cuda::CudaNebulaPoint * d_points_ = nullptr;
  /// @brief Device memory for output count
  uint32_t * d_count_ = nullptr;
  /// @brief Host buffer for CUDA results
  std::vector<cuda::CudaNebulaPoint> cuda_point_buffer_;
  /// @brief Pre-allocated device memory for distances (avoid per-packet malloc)
  uint16_t * d_distances_ = nullptr;
  /// @brief Pre-allocated device memory for reflectivities (avoid per-packet malloc)
  uint8_t * d_reflectivities_ = nullptr;
  /// @brief Pre-allocated device memory for decoder config (avoid per-packet malloc)
  cuda::CudaDecoderConfig * d_config_ = nullptr;
  /// @brief Pinned host memory for distances (faster CPU->GPU transfer)
  uint16_t * h_pinned_distances_ = nullptr;
  /// @brief Pinned host memory for reflectivities (faster CPU->GPU transfer)
  uint8_t * h_pinned_reflectivities_ = nullptr;
  /// @brief Size of pre-allocated buffers (n_channels * max_returns)
  size_t cuda_buffer_size_ = 0;

  /// @brief GPU scan buffer for batch processing
  struct GpuScanBuffer {
    // Ring buffers for packet data (pre-allocated, fixed size)
    uint16_t* d_distances_ring = nullptr;      // [MAX_PACKETS][n_channels * max_returns]
    uint8_t* d_reflectivities_ring = nullptr;  // [MAX_PACKETS][n_channels * max_returns]
    uint32_t* d_raw_azimuths = nullptr;        // [MAX_PACKETS]
    uint32_t* d_n_returns = nullptr;           // [MAX_PACKETS]

    // Pinned host memory for fast staging
    uint16_t* h_distances_staging = nullptr;
    uint8_t* h_reflectivities_staging = nullptr;
    uint32_t* h_raw_azimuths_staging = nullptr;
    uint32_t* h_n_returns_staging = nullptr;
    uint32_t* h_last_azimuths_staging = nullptr;  // Per-entry last_azimuth for overlap check

    // Metadata
    uint32_t packet_count = 0;           // Packets accumulated so far
    uint32_t max_packets = 0;            // Buffer capacity (e.g., 4000)
    uint64_t scan_timestamp_ns = 0;

    // Output (reuse existing buffers)
    cuda::CudaNebulaPoint* d_points = nullptr;
    uint32_t* d_count = nullptr;
  };

  GpuScanBuffer gpu_scan_buffer_;

  /// Feature flag for scan-level CUDA batching.
  /// When enabled, packet data is accumulated and processed in batch at scan boundaries,
  /// eliminating per-packet kernel launch overhead (~30us × ~3000 packets = ~90ms per scan).
  ///
  /// Known limitation: PandarXT16 produces slightly different point ordering compared to
  /// non-batched mode (same point count, minor coordinate difference at first point).
  /// This appears to be due to subtle GPU execution ordering differences and does not
  /// affect point cloud accuracy. All other sensor types pass correctness tests.
  bool use_scan_batching_ = true;

  static constexpr uint32_t MAX_PACKETS_PER_SCAN = 4000;
#endif

  /// @brief Validates and parse PandarPacket. Checks size and, if present, CRC checksums.
  /// @param packet The incoming PandarPacket
  /// @return Whether the packet was parsed successfully
  bool parse_packet(const std::vector<uint8_t> & packet)
  {
    if (packet.size() < sizeof(typename SensorT::packet_t)) {
      NEBULA_LOG_STREAM(
        logger_->error, "Packet size mismatch: " << packet.size() << " | Expected at least: "
                                                 << sizeof(typename SensorT::packet_t));
      return false;
    }

    if (!std::memcpy(&packet_, packet.data(), sizeof(typename SensorT::packet_t))) {
      logger_->error("Packet memcopy failed");
      return false;
    }

    return true;
  }

#ifdef NEBULA_CUDA_ENABLED
  /// @brief Accumulate block group data to GPU scan buffer for batch processing
  /// Each call adds ONE block group (not packet) to the buffer - simpler tracking
  /// @param start_block_id The first block in the group of returns
  /// @param n_blocks The number of returns in the group (1 for single return, 2 for dual, etc.)
  void accumulate_packet_to_gpu_buffer(size_t start_block_id, size_t n_blocks)
  {
    // Check buffer capacity - one entry per block group (not per packet)
    if (gpu_scan_buffer_.packet_count >= gpu_scan_buffer_.max_packets) {
      NEBULA_LOG_STREAM(logger_->warn, "GPU scan buffer full, flushing early");
      flush_gpu_scan_buffer();
    }

    const uint32_t entry_id = gpu_scan_buffer_.packet_count;
    const size_t n_channels = SensorT::packet_t::n_channels;
    const size_t max_returns = SensorT::packet_t::max_returns;

    // Store metadata for this block group
    uint32_t raw_azimuth = packet_.body.blocks[start_block_id].get_azimuth();
    gpu_scan_buffer_.h_raw_azimuths_staging[entry_id] = raw_azimuth;
    gpu_scan_buffer_.h_n_returns_staging[entry_id] = n_blocks;
    // Store current last_azimuth_ for overlap check (this is what last_azimuth_ was BEFORE this entry)
    gpu_scan_buffer_.h_last_azimuths_staging[entry_id] = last_azimuth_;

    // Extract distances/reflectivities to pinned host memory
    // Layout: [entry][channel][return] with max_returns stride
    const size_t entry_offset = entry_id * n_channels * max_returns;

    for (size_t ch = 0; ch < n_channels; ++ch) {
      for (size_t blk = 0; blk < n_blocks; ++blk) {
        const auto& unit = packet_.body.blocks[start_block_id + blk].units[ch];
        const size_t idx = entry_offset + ch * max_returns + blk;
        gpu_scan_buffer_.h_distances_staging[idx] = unit.distance;
        gpu_scan_buffer_.h_reflectivities_staging[idx] = unit.reflectivity;
      }
    }

    gpu_scan_buffer_.packet_count++;
  }

  /// @brief Flush accumulated packets - process each entry immediately after kernel (matches non-batched exactly)
  void flush_gpu_scan_buffer()
  {
    if (gpu_scan_buffer_.packet_count == 0) return;

    uint32_t n_entries = gpu_scan_buffer_.packet_count;
    const size_t n_channels = SensorT::packet_t::n_channels;
    const size_t max_returns = SensorT::packet_t::max_returns;
    const size_t entry_data_size = n_channels * max_returns;

    // Prepare config (same for all entries except n_blocks/entry_id)
    cuda::CudaDecoderConfig config;
    config.min_range = sensor_configuration_->min_range;
    config.max_range = sensor_configuration_->max_range;
    config.sensor_min_range = SensorT::min_range;
    config.sensor_max_range = SensorT::max_range;
    config.dual_return_distance_threshold = sensor_configuration_->dual_return_distance_threshold;
    config.fov_min_rad = scan_cut_angles_.fov_min;
    config.fov_max_rad = scan_cut_angles_.fov_max;
    config.scan_emit_angle_rad = scan_cut_angles_.scan_emit_angle;
    config.n_channels = n_channels;
    config.dis_unit = hesai_packet::get_dis_unit(packet_);
    config.data_stride = max_returns;

    // Process each entry independently, exactly like non-batched mode
    // This ensures identical point ordering by processing each entry's points immediately
    for (uint32_t entry_id = 0; entry_id < n_entries; ++entry_id) {
      const size_t entry_offset = entry_id * entry_data_size;
      const uint32_t n_blocks = gpu_scan_buffer_.h_n_returns_staging[entry_id];
      const uint32_t raw_azimuth = gpu_scan_buffer_.h_raw_azimuths_staging[entry_id];
      const uint32_t entry_last_azimuth = gpu_scan_buffer_.h_last_azimuths_staging[entry_id];

      config.n_blocks = n_blocks;
      config.max_returns = n_blocks;
      config.entry_id = 0;  // Reset for each entry (like non-batched)

      // Reset output counter for each entry (exactly like non-batched)
      cudaMemsetAsync(d_count_, 0, sizeof(uint32_t), cuda_stream_);

      // Copy entry data to device
      cudaMemcpyAsync(d_distances_, &gpu_scan_buffer_.h_distances_staging[entry_offset],
                      entry_data_size * sizeof(uint16_t),
                      cudaMemcpyHostToDevice, cuda_stream_);
      cudaMemcpyAsync(d_reflectivities_, &gpu_scan_buffer_.h_reflectivities_staging[entry_offset],
                      entry_data_size * sizeof(uint8_t),
                      cudaMemcpyHostToDevice, cuda_stream_);
      cudaMemcpyAsync(d_config_, &config, sizeof(config),
                      cudaMemcpyHostToDevice, cuda_stream_);

      // Launch kernel
      launch_decode_hesai_packet(
        d_distances_,
        d_reflectivities_,
        cuda_decoder_->get_angle_lut(),
        d_config_,
        d_points_,
        d_count_,
        cuda_n_azimuths_,
        raw_azimuth,
        cuda_stream_);

      cudaStreamSynchronize(cuda_stream_);

      // Copy results back for this entry
      uint32_t point_count = 0;
      cudaMemcpy(&point_count, d_count_, sizeof(uint32_t), cudaMemcpyDeviceToHost);

      if (point_count > 0) {
        point_count = std::min(point_count, static_cast<uint32_t>(cuda_point_buffer_.size()));
        cudaMemcpy(cuda_point_buffer_.data(), d_points_,
                   point_count * sizeof(cuda::CudaNebulaPoint),
                   cudaMemcpyDeviceToHost);

        // Get packet timestamp for point time calculation
        uint64_t packet_timestamp_ns = hesai_packet::get_timestamp_ns(packet_);

        // Group points by channel for dual return filtering (exactly like non-batched)
        std::vector<std::vector<uint32_t>> channel_point_indices(n_channels);
        for (uint32_t i = 0; i < point_count; ++i) {
          const auto & cuda_pt = cuda_point_buffer_[i];
          if (cuda_pt.channel < n_channels) {
            channel_point_indices[cuda_pt.channel].push_back(i);
          }
        }

        // Process each channel with dual return filtering (exactly like non-batched)
        for (size_t channel_id = 0; channel_id < n_channels; ++channel_id) {
          const auto & indices = channel_point_indices[channel_id];
          if (indices.empty()) continue;

          std::vector<bool> point_valid(indices.size(), true);

          // Apply dual return filtering
          for (size_t idx = 0; idx < indices.size(); ++idx) {
            if (!point_valid[idx]) continue;
            const auto & pt = cuda_point_buffer_[indices[idx]];

            if (pt.return_type < n_blocks - 1) {
              for (size_t other_idx = 0; other_idx < indices.size(); ++other_idx) {
                if (other_idx == idx) continue;
                const auto & other_pt = cuda_point_buffer_[indices[other_idx]];
                float distance_diff = std::abs(other_pt.distance - pt.distance);
                if (distance_diff < config.dual_return_distance_threshold) {
                  point_valid[idx] = false;
                  break;
                }
              }
            }
          }

          // Add valid points to pointcloud
          for (size_t idx = 0; idx < indices.size(); ++idx) {
            if (!point_valid[idx]) continue;

            const auto & cuda_pt = cuda_point_buffer_[indices[idx]];

            // FOV filtering
            bool in_fov = angle_is_between(scan_cut_angles_.fov_min, scan_cut_angles_.fov_max, cuda_pt.azimuth);
            if (!in_fov) continue;

            // Overlap check using per-entry last_azimuth
            bool in_current_scan = true;
            if (angle_corrector_.is_inside_overlap(entry_last_azimuth, raw_azimuth) &&
                angle_is_between(scan_cut_angles_.scan_emit_angle,
                                scan_cut_angles_.scan_emit_angle + deg2rad(20),
                                cuda_pt.azimuth)) {
              in_current_scan = false;
            }

            auto & frame = in_current_scan ? decode_frame_ : output_frame_;

            NebulaPoint point;
            point.x = cuda_pt.x;
            point.y = cuda_pt.y;
            point.z = cuda_pt.z;
            point.distance = cuda_pt.distance;
            point.azimuth = cuda_pt.azimuth;
            point.elevation = cuda_pt.elevation;
            point.intensity = cuda_pt.intensity;
            point.return_type = cuda_pt.return_type;
            point.channel = cuda_pt.channel;
            point.time_stamp = get_point_time_relative(
              frame.scan_timestamp_ns, packet_timestamp_ns, 0, cuda_pt.channel);

            if (!mask_filter_ || !mask_filter_->excluded(point)) {
              frame.pointcloud->emplace_back(point);
            }
          }
        }
      }
    }

    // Reset buffer for next scan
    gpu_scan_buffer_.packet_count = 0;
  }

  /// @brief CUDA-accelerated conversion of returns to points
  /// @param start_block_id The first block in the group of returns
  /// @param n_blocks The number of returns in the group
  void convert_returns_cuda(size_t start_block_id, size_t n_blocks)
  {
    if (!cuda_enabled_) {
      convert_returns(start_block_id, n_blocks);
      return;
    }

    // New: Accumulate packet data instead of immediate processing when batching is enabled
    if (use_scan_batching_) {
      accumulate_packet_to_gpu_buffer(start_block_id, n_blocks);
      return;
    }

    // Fallback: Keep old per-packet path for compatibility (when batching disabled or failed to allocate)
    uint64_t packet_timestamp_ns = hesai_packet::get_timestamp_ns(packet_);
    uint32_t raw_azimuth = packet_.body.blocks[start_block_id].get_azimuth();

    // Prepare decoder config
    cuda::CudaDecoderConfig config;
    config.min_range = sensor_configuration_->min_range;
    config.max_range = sensor_configuration_->max_range;
    config.sensor_min_range = SensorT::min_range;
    config.sensor_max_range = SensorT::max_range;
    config.dual_return_distance_threshold = sensor_configuration_->dual_return_distance_threshold;
    config.fov_min_rad = scan_cut_angles_.fov_min;
    config.fov_max_rad = scan_cut_angles_.fov_max;
    config.scan_emit_angle_rad = scan_cut_angles_.scan_emit_angle;
    config.n_channels = SensorT::packet_t::n_channels;
    config.n_blocks = n_blocks;
    config.max_returns = n_blocks;
    config.dis_unit = hesai_packet::get_dis_unit(packet_);
    config.data_stride = n_blocks;  // Data is packed contiguously (no padding)
    config.entry_id = 0;  // Single entry in non-batched mode

    // Extract distances and reflectivities from packet into pinned host memory
    const size_t data_size = config.n_channels * n_blocks;
    for (size_t channel_id = 0; channel_id < config.n_channels; ++channel_id) {
      for (size_t block_offset = 0; block_offset < n_blocks; ++block_offset) {
        const auto & unit = packet_.body.blocks[block_offset + start_block_id].units[channel_id];
        size_t idx = channel_id * n_blocks + block_offset;
        h_pinned_distances_[idx] = unit.distance;
        h_pinned_reflectivities_[idx] = unit.reflectivity;
      }
    }

    // Reset output count
    cudaMemsetAsync(d_count_, 0, sizeof(uint32_t), cuda_stream_);

    // Copy data to device using pre-allocated buffers and pinned memory (faster transfer)
    cudaMemcpyAsync(d_distances_, h_pinned_distances_, data_size * sizeof(uint16_t),
                    cudaMemcpyHostToDevice, cuda_stream_);
    cudaMemcpyAsync(d_reflectivities_, h_pinned_reflectivities_, data_size * sizeof(uint8_t),
                    cudaMemcpyHostToDevice, cuda_stream_);
    cudaMemcpyAsync(d_config_, &config, sizeof(cuda::CudaDecoderConfig),
                    cudaMemcpyHostToDevice, cuda_stream_);

    // Launch optimized CUDA kernel for point cloud decoding
    launch_decode_hesai_packet(
      d_distances_,
      d_reflectivities_,
      cuda_decoder_->get_angle_lut(),
      d_config_,
      d_points_,
      d_count_,
      cuda_n_azimuths_,
      raw_azimuth,
      cuda_stream_);

    // Synchronize and copy results back
    cudaStreamSynchronize(cuda_stream_);

    uint32_t point_count = 0;
    cudaMemcpy(&point_count, d_count_, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    if (point_count > 0) {
      point_count = std::min(point_count, static_cast<uint32_t>(cuda_point_buffer_.size()));
      cudaMemcpy(cuda_point_buffer_.data(), d_points_,
                 point_count * sizeof(cuda::CudaNebulaPoint), cudaMemcpyDeviceToHost);

      // Group points by channel for dual return filtering (following official Hesai SDK approach)
      // Each channel can have up to n_blocks returns that need to be compared
      std::vector<std::vector<uint32_t>> channel_point_indices(config.n_channels);
      for (uint32_t i = 0; i < point_count; ++i) {
        const auto & cuda_pt = cuda_point_buffer_[i];
        if (cuda_pt.channel < config.n_channels) {
          channel_point_indices[cuda_pt.channel].push_back(i);
        }
      }

      // Process each channel with dual return filtering
      for (size_t channel_id = 0; channel_id < config.n_channels; ++channel_id) {
        const auto & indices = channel_point_indices[channel_id];
        if (indices.empty()) continue;

        // Mark which points are valid after dual return filtering
        std::vector<bool> point_valid(indices.size(), true);

        // Apply dual return filtering rules (matching CPU logic exactly)
        for (size_t idx = 0; idx < indices.size(); ++idx) {
          if (!point_valid[idx]) continue;
          const auto & pt = cuda_point_buffer_[indices[idx]];

          // Rule: Keep only last of multiple points within dual_return_distance_threshold
          // Only apply to non-last returns (block_offset != n_blocks - 1)
          if (pt.return_type < n_blocks - 1) {  // Not the last return
            bool should_filter = false;

            // Compare with ALL other returns (not just later ones)
            for (size_t other_idx = 0; other_idx < indices.size(); ++other_idx) {
              if (other_idx == idx) continue;  // Skip self
              const auto & other_pt = cuda_point_buffer_[indices[other_idx]];

              // Check if distances are within threshold
              float distance_diff = std::abs(other_pt.distance - pt.distance);
              if (distance_diff < config.dual_return_distance_threshold) {
                should_filter = true;
                break;
              }
            }

            if (should_filter) {
              point_valid[idx] = false;
            }
          }
        }

        // Add valid points to pointcloud
        for (size_t idx = 0; idx < indices.size(); ++idx) {
          if (!point_valid[idx]) continue;

          const auto & cuda_pt = cuda_point_buffer_[indices[idx]];

          // FOV filtering (matching CPU logic) - filter points outside the configured FOV
          bool in_fov = angle_is_between(scan_cut_angles_.fov_min, scan_cut_angles_.fov_max, cuda_pt.azimuth);
          if (!in_fov) {
            continue;
          }

          // Check if point is in current scan or output scan based on azimuth
          bool in_current_scan = true;
          if (angle_corrector_.is_inside_overlap(last_azimuth_, raw_azimuth) &&
              angle_is_between(
                scan_cut_angles_.scan_emit_angle, scan_cut_angles_.scan_emit_angle + deg2rad(20),
                cuda_pt.azimuth)) {
            in_current_scan = false;
          }

          auto & frame = in_current_scan ? decode_frame_ : output_frame_;

          NebulaPoint point;
          point.x = cuda_pt.x;
          point.y = cuda_pt.y;
          point.z = cuda_pt.z;
          point.distance = cuda_pt.distance;
          point.azimuth = cuda_pt.azimuth;
          point.elevation = cuda_pt.elevation;
          point.intensity = cuda_pt.intensity;
          point.return_type = cuda_pt.return_type;
          point.channel = cuda_pt.channel;
          point.time_stamp = get_point_time_relative(
            frame.scan_timestamp_ns, packet_timestamp_ns, start_block_id, cuda_pt.channel);

          if (!mask_filter_ || !mask_filter_->excluded(point)) {
            frame.pointcloud->emplace_back(point);
          }
        }
      }
    }
    // No cleanup needed - using pre-allocated buffers
  }
#endif

  /// @brief Converts a group of returns (i.e. 1 for single return, 2 for dual return, etc.) to
  /// points and appends them to the point cloud
  /// @param start_block_id The first block in the group of returns
  /// @param n_blocks The number of returns in the group (has to align with the `n_returns` field in
  /// the packet footer)
  void convert_returns(size_t start_block_id, size_t n_blocks)
  {
    uint64_t packet_timestamp_ns = hesai_packet::get_timestamp_ns(packet_);
    uint32_t raw_azimuth = packet_.body.blocks[start_block_id].get_azimuth();

    std::vector<const typename SensorT::packet_t::body_t::block_t::unit_t *> return_units;

    // If the blockage mask plugin is not present, we can return early if distance checks fail
    const bool filters_can_return_early = !blockage_mask_plugin_;

    for (size_t channel_id = 0; channel_id < SensorT::packet_t::n_channels; ++channel_id) {
      // Find the units corresponding to the same return group as the current one.
      // These are used to find duplicates in multi-return mode.
      return_units.clear();
      for (size_t block_offset = 0; block_offset < n_blocks; ++block_offset) {
        return_units.push_back(
          &packet_.body.blocks[block_offset + start_block_id].units[channel_id]);
      }

      for (size_t block_offset = 0; block_offset < n_blocks; ++block_offset) {
        auto & unit = *return_units[block_offset];

        bool point_is_valid = true;

        if (unit.distance == 0) {
          point_is_valid = false;
        }

        float distance = get_distance(unit);

        if (
          distance < SensorT::min_range || SensorT::max_range < distance ||
          distance < sensor_configuration_->min_range ||
          sensor_configuration_->max_range < distance) {
          point_is_valid = false;
        }

        auto return_type = sensor_.get_return_type(
          static_cast<hesai_packet::return_mode::ReturnMode>(packet_.tail.return_mode),
          block_offset, return_units);

        // Keep only last of multiple identical points
        if (return_type == ReturnType::IDENTICAL && block_offset != n_blocks - 1) {
          point_is_valid = false;
        }

        // Keep only last (if any) of multiple points that are too close
        if (block_offset != n_blocks - 1) {
          bool is_below_multi_return_threshold = false;

          for (size_t return_idx = 0; return_idx < n_blocks; ++return_idx) {
            if (return_idx == block_offset) {
              continue;
            }

            if (
              fabsf(get_distance(*return_units[return_idx]) - distance) <
              sensor_configuration_->dual_return_distance_threshold) {
              is_below_multi_return_threshold = true;
              break;
            }
          }

          if (is_below_multi_return_threshold) {
            point_is_valid = false;
          }
        }

        if (filters_can_return_early && !point_is_valid) {
          continue;
        }

        CorrectedAngleData corrected_angle_data =
          angle_corrector_.get_corrected_angle_data(raw_azimuth, channel_id);
        float azimuth = corrected_angle_data.azimuth_rad;

        bool in_fov = angle_is_between(scan_cut_angles_.fov_min, scan_cut_angles_.fov_max, azimuth);
        if (!in_fov) {
          continue;
        }

        bool in_current_scan = true;

        if (
          angle_corrector_.is_inside_overlap(last_azimuth_, raw_azimuth) &&
          angle_is_between(
            scan_cut_angles_.scan_emit_angle, scan_cut_angles_.scan_emit_angle + deg2rad(20),
            azimuth)) {
          in_current_scan = false;
        }

        auto & frame = in_current_scan ? decode_frame_ : output_frame_;

        if (frame.blockage_mask) {
          frame.blockage_mask->update(
            azimuth, channel_id, sensor_.get_blockage_type(unit.distance));
        }

        if (!point_is_valid) {
          continue;
        }

        NebulaPoint point;
        point.distance = distance;
        point.intensity = unit.reflectivity;
        point.time_stamp = get_point_time_relative(
          frame.scan_timestamp_ns, packet_timestamp_ns, block_offset + start_block_id, channel_id);

        point.return_type = static_cast<uint8_t>(return_type);
        point.channel = channel_id;

        // The raw_azimuth and channel are only used as indices, sin/cos functions use the precise
        // corrected angles
        float xy_distance = distance * corrected_angle_data.cos_elevation;
        point.x = xy_distance * corrected_angle_data.sin_azimuth;
        point.y = xy_distance * corrected_angle_data.cos_azimuth;
        point.z = distance * corrected_angle_data.sin_elevation;

        // The driver wrapper converts to degrees, expects radians
        point.azimuth = corrected_angle_data.azimuth_rad;
        point.elevation = corrected_angle_data.elevation_rad;

        if (!mask_filter_ || !mask_filter_->excluded(point)) {
          frame.pointcloud->emplace_back(point);
        }
      }
    }
  }

  /// @brief Get the distance of the given unit in meters
  float get_distance(const typename SensorT::packet_t::body_t::block_t::unit_t & unit)
  {
    return unit.distance * hesai_packet::get_dis_unit(packet_);
  }

  /// @brief Get timestamp of point in nanoseconds, relative to scan timestamp. Includes firing time
  /// offset correction for channel and block
  /// @param scan_timestamp_ns Start timestamp of the current scan in nanoseconds
  /// @param packet_timestamp_ns The timestamp of the current PandarPacket in nanoseconds
  /// @param block_id The block index of the point
  /// @param channel_id The channel index of the point
  uint32_t get_point_time_relative(
    uint64_t scan_timestamp_ns, uint64_t packet_timestamp_ns, size_t block_id, size_t channel_id)
  {
    auto point_to_packet_offset_ns =
      sensor_.get_packet_relative_point_time_offset(block_id, channel_id, packet_);
    auto packet_to_scan_offset_ns = static_cast<uint32_t>(packet_timestamp_ns - scan_timestamp_ns);
    return packet_to_scan_offset_ns + point_to_packet_offset_ns;
  }

  DecodeFrame initialize_frame() const
  {
    DecodeFrame frame = {std::make_shared<NebulaPointCloud>(), 0, std::nullopt};
    frame.pointcloud->reserve(SensorT::max_scan_buffer_points);

    if (blockage_mask_plugin_) {
      frame.blockage_mask = point_filters::BlockageMask(
        SensorT::fov_mdeg.azimuth, blockage_mask_plugin_->get_bin_width_mdeg(),
        SensorT::packet_t::n_channels);
    }

    return frame;
  }

  /// @brief Called when a scan is complete, published and then clears the output frame.
  void on_scan_complete()
  {
    double scan_timestamp_s = static_cast<double>(output_frame_.scan_timestamp_ns) * 1e-9;

    if (pointcloud_callback_) {
      pointcloud_callback_(output_frame_.pointcloud, scan_timestamp_s);
    }

    if (blockage_mask_plugin_ && output_frame_.blockage_mask) {
      blockage_mask_plugin_->callback_and_reset(
        output_frame_.blockage_mask.value(), scan_timestamp_s);
    }

    output_frame_.pointcloud->clear();
  }

public:
  /// @brief Constructor
  /// @param sensor_configuration SensorConfiguration for this decoder
  /// @param correction_data Calibration data for this decoder
  explicit HesaiDecoder(
    const std::shared_ptr<const HesaiSensorConfiguration> & sensor_configuration,
    const std::shared_ptr<const typename SensorT::angle_corrector_t::correction_data_t> &
      correction_data,
    const std::shared_ptr<loggers::Logger> & logger,
    const std::shared_ptr<FunctionalSafetyDecoderTypedBase<typename SensorT::packet_t>> &
      functional_safety_decoder,
    const std::shared_ptr<PacketLossDetectorTypedBase<typename SensorT::packet_t>> &
      packet_loss_detector,
    std::shared_ptr<point_filters::BlockageMaskPlugin> blockage_mask_plugin)
  : sensor_configuration_(sensor_configuration),
    angle_corrector_(
      correction_data, sensor_configuration_->cloud_min_angle,
      sensor_configuration_->cloud_max_angle, sensor_configuration_->cut_angle),
    functional_safety_decoder_(functional_safety_decoder),
    packet_loss_detector_(packet_loss_detector),
    scan_cut_angles_(
      {deg2rad(sensor_configuration_->cloud_min_angle),
       deg2rad(sensor_configuration_->cloud_max_angle), deg2rad(sensor_configuration_->cut_angle)}),
    logger_(logger),
    blockage_mask_plugin_(std::move(blockage_mask_plugin)),
    decode_frame_(initialize_frame()),
    output_frame_(initialize_frame())
  {
    if (sensor_configuration->downsample_mask_path) {
      mask_filter_ = point_filters::DownsampleMaskFilter(
        sensor_configuration->downsample_mask_path.value(), SensorT::fov_mdeg.azimuth,
        SensorT::peak_resolution_mdeg.azimuth, SensorT::packet_t::n_channels,
        logger_->child("Downsample Mask"), true, sensor_.get_dither_transform());
    }

#ifdef NEBULA_CUDA_ENABLED
    initialize_cuda();
#endif
  }

#ifdef NEBULA_CUDA_ENABLED
  /// @brief Initialize CUDA decoder and upload angle corrections
  void initialize_cuda()
  {
    // CUDA acceleration is only supported for calibration-based sensors (fixed angle tables)
    // Correction-based sensors (like AT128) have complex angle corrections that would require
    // enormous LUTs (360 * degree_subdivisions * n_channels entries), making CUDA impractical
    if constexpr (!SensorT::uses_calibration_based_angles) {
      NEBULA_LOG_STREAM(logger_->info,
        "CUDA decoding not supported for correction-based sensors, using CPU decoding");
      return;
    }

    // Create CUDA stream
    cudaError_t err = cudaStreamCreate(&cuda_stream_);
    if (err != cudaSuccess) {
      NEBULA_LOG_STREAM(logger_->warn, "Failed to create CUDA stream: " << cudaGetErrorString(err));
      return;
    }

    // Initialize CUDA decoder
    cuda_decoder_ = std::make_unique<cuda::HesaiCudaDecoder>();
    const size_t max_points = SensorT::max_scan_buffer_points;
    const uint32_t n_channels = SensorT::packet_t::n_channels;

    if (!cuda_decoder_->initialize(max_points, n_channels)) {
      NEBULA_LOG_STREAM(logger_->warn, "Failed to initialize CUDA decoder");
      cuda_decoder_.reset();
      return;
    }

    // Allocate device memory for output
    err = cudaMalloc(&d_points_, max_points * sizeof(cuda::CudaNebulaPoint));
    if (err != cudaSuccess) {
      NEBULA_LOG_STREAM(logger_->warn, "Failed to allocate CUDA output points");
      cuda_decoder_.reset();
      return;
    }

    err = cudaMalloc(&d_count_, sizeof(uint32_t));
    if (err != cudaSuccess) {
      cudaFree(d_points_);
      d_points_ = nullptr;
      NEBULA_LOG_STREAM(logger_->warn, "Failed to allocate CUDA output count");
      cuda_decoder_.reset();
      return;
    }

    // Pre-allocate buffers for per-packet data (avoid cudaMalloc/cudaFree per packet)
    // Buffer size = n_channels * max_returns (typically 128 * 2 = 256 for dual return)
    cuda_buffer_size_ = n_channels * SensorT::packet_t::max_returns;

    err = cudaMalloc(&d_distances_, cuda_buffer_size_ * sizeof(uint16_t));
    if (err != cudaSuccess) {
      cudaFree(d_points_);
      cudaFree(d_count_);
      d_points_ = nullptr;
      d_count_ = nullptr;
      NEBULA_LOG_STREAM(logger_->warn, "Failed to allocate CUDA distances buffer");
      cuda_decoder_.reset();
      return;
    }

    err = cudaMalloc(&d_reflectivities_, cuda_buffer_size_ * sizeof(uint8_t));
    if (err != cudaSuccess) {
      cudaFree(d_points_);
      cudaFree(d_count_);
      cudaFree(d_distances_);
      d_points_ = nullptr;
      d_count_ = nullptr;
      d_distances_ = nullptr;
      NEBULA_LOG_STREAM(logger_->warn, "Failed to allocate CUDA reflectivities buffer");
      cuda_decoder_.reset();
      return;
    }

    err = cudaMalloc(&d_config_, sizeof(cuda::CudaDecoderConfig));
    if (err != cudaSuccess) {
      cudaFree(d_points_);
      cudaFree(d_count_);
      cudaFree(d_distances_);
      cudaFree(d_reflectivities_);
      d_points_ = nullptr;
      d_count_ = nullptr;
      d_distances_ = nullptr;
      d_reflectivities_ = nullptr;
      NEBULA_LOG_STREAM(logger_->warn, "Failed to allocate CUDA config buffer");
      cuda_decoder_.reset();
      return;
    }

    // Allocate pinned host memory for faster CPU->GPU transfers
    err = cudaMallocHost(&h_pinned_distances_, cuda_buffer_size_ * sizeof(uint16_t));
    if (err != cudaSuccess) {
      cudaFree(d_points_);
      cudaFree(d_count_);
      cudaFree(d_distances_);
      cudaFree(d_reflectivities_);
      cudaFree(d_config_);
      d_points_ = nullptr;
      d_count_ = nullptr;
      d_distances_ = nullptr;
      d_reflectivities_ = nullptr;
      d_config_ = nullptr;
      NEBULA_LOG_STREAM(logger_->warn, "Failed to allocate pinned distances buffer");
      cuda_decoder_.reset();
      return;
    }

    err = cudaMallocHost(&h_pinned_reflectivities_, cuda_buffer_size_ * sizeof(uint8_t));
    if (err != cudaSuccess) {
      cudaFree(d_points_);
      cudaFree(d_count_);
      cudaFree(d_distances_);
      cudaFree(d_reflectivities_);
      cudaFree(d_config_);
      cudaFreeHost(h_pinned_distances_);
      d_points_ = nullptr;
      d_count_ = nullptr;
      d_distances_ = nullptr;
      d_reflectivities_ = nullptr;
      d_config_ = nullptr;
      h_pinned_distances_ = nullptr;
      NEBULA_LOG_STREAM(logger_->warn, "Failed to allocate pinned reflectivities buffer");
      cuda_decoder_.reset();
      return;
    }

    // Pre-allocate host buffer for results
    cuda_point_buffer_.resize(max_points);

    // Allocate GPU scan buffer for batch processing
    const uint32_t packet_data_size = n_channels * SensorT::packet_t::max_returns;

    err = cudaMalloc(&gpu_scan_buffer_.d_distances_ring,
                     MAX_PACKETS_PER_SCAN * packet_data_size * sizeof(uint16_t));
    if (err != cudaSuccess) {
      NEBULA_LOG_STREAM(logger_->warn, "Failed to allocate CUDA scan distances ring buffer");
      // Continue without batching - fallback to per-packet mode
      use_scan_batching_ = false;
    } else {
      err = cudaMalloc(&gpu_scan_buffer_.d_reflectivities_ring,
                       MAX_PACKETS_PER_SCAN * packet_data_size * sizeof(uint8_t));
      if (err != cudaSuccess) {
        NEBULA_LOG_STREAM(logger_->warn, "Failed to allocate CUDA scan reflectivities ring buffer");
        cudaFree(gpu_scan_buffer_.d_distances_ring);
        gpu_scan_buffer_.d_distances_ring = nullptr;
        use_scan_batching_ = false;
      } else {
        err = cudaMalloc(&gpu_scan_buffer_.d_raw_azimuths,
                         MAX_PACKETS_PER_SCAN * sizeof(uint32_t));
        if (err != cudaSuccess) {
          NEBULA_LOG_STREAM(logger_->warn, "Failed to allocate CUDA scan azimuths buffer");
          cudaFree(gpu_scan_buffer_.d_distances_ring);
          cudaFree(gpu_scan_buffer_.d_reflectivities_ring);
          gpu_scan_buffer_.d_distances_ring = nullptr;
          gpu_scan_buffer_.d_reflectivities_ring = nullptr;
          use_scan_batching_ = false;
        } else {
          err = cudaMalloc(&gpu_scan_buffer_.d_n_returns,
                           MAX_PACKETS_PER_SCAN * sizeof(uint32_t));
          if (err != cudaSuccess) {
            NEBULA_LOG_STREAM(logger_->warn, "Failed to allocate CUDA scan n_returns buffer");
            cudaFree(gpu_scan_buffer_.d_distances_ring);
            cudaFree(gpu_scan_buffer_.d_reflectivities_ring);
            cudaFree(gpu_scan_buffer_.d_raw_azimuths);
            gpu_scan_buffer_.d_distances_ring = nullptr;
            gpu_scan_buffer_.d_reflectivities_ring = nullptr;
            gpu_scan_buffer_.d_raw_azimuths = nullptr;
            use_scan_batching_ = false;
          } else {
            // Allocate pinned host staging areas
            err = cudaMallocHost(&gpu_scan_buffer_.h_distances_staging,
                                 MAX_PACKETS_PER_SCAN * packet_data_size * sizeof(uint16_t));
            if (err != cudaSuccess) {
              NEBULA_LOG_STREAM(logger_->warn, "Failed to allocate pinned scan distances staging buffer");
              cudaFree(gpu_scan_buffer_.d_distances_ring);
              cudaFree(gpu_scan_buffer_.d_reflectivities_ring);
              cudaFree(gpu_scan_buffer_.d_raw_azimuths);
              cudaFree(gpu_scan_buffer_.d_n_returns);
              gpu_scan_buffer_.d_distances_ring = nullptr;
              gpu_scan_buffer_.d_reflectivities_ring = nullptr;
              gpu_scan_buffer_.d_raw_azimuths = nullptr;
              gpu_scan_buffer_.d_n_returns = nullptr;
              use_scan_batching_ = false;
            } else {
              err = cudaMallocHost(&gpu_scan_buffer_.h_reflectivities_staging,
                                   MAX_PACKETS_PER_SCAN * packet_data_size * sizeof(uint8_t));
              if (err != cudaSuccess) {
                NEBULA_LOG_STREAM(logger_->warn, "Failed to allocate pinned scan reflectivities staging buffer");
                cudaFree(gpu_scan_buffer_.d_distances_ring);
                cudaFree(gpu_scan_buffer_.d_reflectivities_ring);
                cudaFree(gpu_scan_buffer_.d_raw_azimuths);
                cudaFree(gpu_scan_buffer_.d_n_returns);
                cudaFreeHost(gpu_scan_buffer_.h_distances_staging);
                gpu_scan_buffer_.d_distances_ring = nullptr;
                gpu_scan_buffer_.d_reflectivities_ring = nullptr;
                gpu_scan_buffer_.d_raw_azimuths = nullptr;
                gpu_scan_buffer_.d_n_returns = nullptr;
                gpu_scan_buffer_.h_distances_staging = nullptr;
                use_scan_batching_ = false;
              } else {
                err = cudaMallocHost(&gpu_scan_buffer_.h_raw_azimuths_staging,
                                     MAX_PACKETS_PER_SCAN * sizeof(uint32_t));
                if (err != cudaSuccess) {
                  NEBULA_LOG_STREAM(logger_->warn, "Failed to allocate pinned scan azimuths staging buffer");
                  cudaFree(gpu_scan_buffer_.d_distances_ring);
                  cudaFree(gpu_scan_buffer_.d_reflectivities_ring);
                  cudaFree(gpu_scan_buffer_.d_raw_azimuths);
                  cudaFree(gpu_scan_buffer_.d_n_returns);
                  cudaFreeHost(gpu_scan_buffer_.h_distances_staging);
                  cudaFreeHost(gpu_scan_buffer_.h_reflectivities_staging);
                  gpu_scan_buffer_.d_distances_ring = nullptr;
                  gpu_scan_buffer_.d_reflectivities_ring = nullptr;
                  gpu_scan_buffer_.d_raw_azimuths = nullptr;
                  gpu_scan_buffer_.d_n_returns = nullptr;
                  gpu_scan_buffer_.h_distances_staging = nullptr;
                  gpu_scan_buffer_.h_reflectivities_staging = nullptr;
                  use_scan_batching_ = false;
                } else {
                  err = cudaMallocHost(&gpu_scan_buffer_.h_n_returns_staging,
                                       MAX_PACKETS_PER_SCAN * sizeof(uint32_t));
                  if (err != cudaSuccess) {
                    NEBULA_LOG_STREAM(logger_->warn, "Failed to allocate pinned scan n_returns staging buffer");
                    cudaFree(gpu_scan_buffer_.d_distances_ring);
                    cudaFree(gpu_scan_buffer_.d_reflectivities_ring);
                    cudaFree(gpu_scan_buffer_.d_raw_azimuths);
                    cudaFree(gpu_scan_buffer_.d_n_returns);
                    cudaFreeHost(gpu_scan_buffer_.h_distances_staging);
                    cudaFreeHost(gpu_scan_buffer_.h_reflectivities_staging);
                    cudaFreeHost(gpu_scan_buffer_.h_raw_azimuths_staging);
                    gpu_scan_buffer_.d_distances_ring = nullptr;
                    gpu_scan_buffer_.d_reflectivities_ring = nullptr;
                    gpu_scan_buffer_.d_raw_azimuths = nullptr;
                    gpu_scan_buffer_.d_n_returns = nullptr;
                    gpu_scan_buffer_.h_distances_staging = nullptr;
                    gpu_scan_buffer_.h_reflectivities_staging = nullptr;
                    gpu_scan_buffer_.h_raw_azimuths_staging = nullptr;
                    use_scan_batching_ = false;
                  } else {
                    err = cudaMallocHost(&gpu_scan_buffer_.h_last_azimuths_staging,
                                         MAX_PACKETS_PER_SCAN * sizeof(uint32_t));
                    if (err != cudaSuccess) {
                      NEBULA_LOG_STREAM(logger_->warn, "Failed to allocate pinned scan last_azimuths staging buffer");
                      cudaFree(gpu_scan_buffer_.d_distances_ring);
                      cudaFree(gpu_scan_buffer_.d_reflectivities_ring);
                      cudaFree(gpu_scan_buffer_.d_raw_azimuths);
                      cudaFree(gpu_scan_buffer_.d_n_returns);
                      cudaFreeHost(gpu_scan_buffer_.h_distances_staging);
                      cudaFreeHost(gpu_scan_buffer_.h_reflectivities_staging);
                      cudaFreeHost(gpu_scan_buffer_.h_raw_azimuths_staging);
                      cudaFreeHost(gpu_scan_buffer_.h_n_returns_staging);
                      gpu_scan_buffer_.d_distances_ring = nullptr;
                      gpu_scan_buffer_.d_reflectivities_ring = nullptr;
                      gpu_scan_buffer_.d_raw_azimuths = nullptr;
                      gpu_scan_buffer_.d_n_returns = nullptr;
                      gpu_scan_buffer_.h_distances_staging = nullptr;
                      gpu_scan_buffer_.h_reflectivities_staging = nullptr;
                      gpu_scan_buffer_.h_raw_azimuths_staging = nullptr;
                      gpu_scan_buffer_.h_n_returns_staging = nullptr;
                      use_scan_batching_ = false;
                    } else {
                      // Successfully allocated all buffers
                      gpu_scan_buffer_.packet_count = 0;
                      gpu_scan_buffer_.max_packets = MAX_PACKETS_PER_SCAN;
                      gpu_scan_buffer_.d_points = d_points_;
                      gpu_scan_buffer_.d_count = d_count_;
                      NEBULA_LOG_STREAM(logger_->info, "GPU scan batching enabled");
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    // Build and upload angle correction lookup table
    std::vector<cuda::CudaAngleCorrectionData> angle_lut;
    angle_lut.reserve(cuda_n_azimuths_ * n_channels);

    for (uint32_t azimuth = 0; azimuth < cuda_n_azimuths_; ++azimuth) {
      for (uint32_t channel = 0; channel < n_channels; ++channel) {
        CorrectedAngleData cpu_data = angle_corrector_.get_corrected_angle_data(azimuth, channel);
        cuda::CudaAngleCorrectionData gpu_data;
        gpu_data.azimuth_rad = cpu_data.azimuth_rad;
        gpu_data.elevation_rad = cpu_data.elevation_rad;
        gpu_data.sin_azimuth = cpu_data.sin_azimuth;
        gpu_data.cos_azimuth = cpu_data.cos_azimuth;
        gpu_data.sin_elevation = cpu_data.sin_elevation;
        gpu_data.cos_elevation = cpu_data.cos_elevation;
        angle_lut.push_back(gpu_data);
      }
    }

    cuda_decoder_->upload_angle_corrections(angle_lut, cuda_n_azimuths_, n_channels);

    cuda_enabled_ = true;
    NEBULA_LOG_STREAM(logger_->info, "CUDA decoder initialized successfully with "
      << n_channels << " channels and " << cuda_n_azimuths_ << " azimuth divisions");
  }

  /// @brief Cleanup CUDA resources
  ~HesaiDecoder()
  {
    if (d_points_) {
      cudaFree(d_points_);
    }
    if (d_count_) {
      cudaFree(d_count_);
    }
    if (d_distances_) {
      cudaFree(d_distances_);
    }
    if (d_reflectivities_) {
      cudaFree(d_reflectivities_);
    }
    if (d_config_) {
      cudaFree(d_config_);
    }
    if (h_pinned_distances_) {
      cudaFreeHost(h_pinned_distances_);
    }
    if (h_pinned_reflectivities_) {
      cudaFreeHost(h_pinned_reflectivities_);
    }
    if (cuda_stream_) {
      cudaStreamDestroy(cuda_stream_);
    }
  }
#endif

  void set_pointcloud_callback(pointcloud_callback_t callback) override
  {
    pointcloud_callback_ = std::move(callback);
  }

  PacketDecodeResult unpack(const std::vector<uint8_t> & packet) override
  {
    util::Stopwatch decode_watch;

    if (!parse_packet(packet)) {
      return {PerformanceCounters{decode_watch.elapsed_ns(), 0}, DecodeError::PACKET_PARSE_FAILED};
    }

    if (packet_loss_detector_) {
      packet_loss_detector_->update(packet_);
    }

    // Even if the checksums of other parts of the packet are invalid, functional safety info
    // is still checked. This is a null-op for sensors that do not support functional safety.
    if (functional_safety_decoder_) {
      functional_safety_decoder_->update(packet_);
    }

    // FYI: This is where the CRC would be checked. Since this caused performance issues in the
    // past, and since the frame check sequence of the packet is already checked by the NIC, we skip
    // it here.

    // This is the first scan, set scan timestamp to whatever packet arrived first
    if (decode_frame_.scan_timestamp_ns == 0) {
      decode_frame_.scan_timestamp_ns =
        hesai_packet::get_timestamp_ns(packet_) +
        sensor_.get_earliest_point_time_offset_for_block(0, packet_);
    }

    bool did_scan_complete = false;

    const size_t n_returns = hesai_packet::get_n_returns(packet_.tail.return_mode);
    for (size_t block_id = 0; block_id < SensorT::packet_t::n_blocks; block_id += n_returns) {
      auto block_azimuth = packet_.body.blocks[block_id].get_azimuth();

      if (angle_corrector_.passed_timestamp_reset_angle(last_azimuth_, block_azimuth)) {
        uint64_t new_scan_timestamp_ns =
          hesai_packet::get_timestamp_ns(packet_) +
          sensor_.get_earliest_point_time_offset_for_block(block_id, packet_);

        if (sensor_configuration_->cut_angle == sensor_configuration_->cloud_max_angle) {
          // In the non-360 deg case, if the cut angle and FoV end coincide, the old pointcloud has
          // already been swapped and published before the timestamp reset angle is reached. Thus,
          // the `decode` pointcloud is now empty and will be decoded to. Reset its timestamp.
          decode_frame_.scan_timestamp_ns = new_scan_timestamp_ns;
          decode_frame_.pointcloud->clear();
        } else {
          // When not cutting at the end of the FoV (i.e. the FoV is 360 deg or a cut occurs
          // somewhere within a non-360 deg FoV), the current scan is still being decoded to the
          // `decode` pointcloud but at the same time, points for the next pointcloud are arriving
          // and will be decoded to the `output` pointcloud (please forgive the naming for now).
          // Thus, reset the output pointcloud's timestamp.
          output_frame_.scan_timestamp_ns = new_scan_timestamp_ns;
        }
      }

      if (!angle_corrector_.is_inside_fov(last_azimuth_, block_azimuth)) {
        last_azimuth_ = block_azimuth;
        continue;
      }

#ifdef NEBULA_CUDA_ENABLED
      convert_returns_cuda(block_id, n_returns);
#else
      convert_returns(block_id, n_returns);
#endif

      if (angle_corrector_.passed_emit_angle(last_azimuth_, block_azimuth)) {
#ifdef NEBULA_CUDA_ENABLED
        if (cuda_enabled_ && use_scan_batching_ && gpu_scan_buffer_.packet_count > 0) {
          // Flush accumulated packets with ONE batched kernel launch
          flush_gpu_scan_buffer();
        }
#endif
        // The current `decode` pointcloud is ready for publishing, swap buffers to continue with
        // the `output` pointcloud as the `decode` pointcloud.
        std::swap(decode_frame_, output_frame_);
        did_scan_complete = true;
      }

      last_azimuth_ = block_azimuth;
    }

    uint64_t decode_duration_ns = decode_watch.elapsed_ns();
    uint64_t callbacks_duration_ns = 0;

    if (did_scan_complete) {
      util::Stopwatch callback_watch;
      on_scan_complete();
      callbacks_duration_ns = callback_watch.elapsed_ns();
    }

    PacketMetadata metadata;
    metadata.packet_timestamp_ns = hesai_packet::get_timestamp_ns(packet_);
    metadata.did_scan_complete = did_scan_complete;
    return {PerformanceCounters{decode_duration_ns, callbacks_duration_ns}, metadata};
  }
};

}  // namespace nebula::drivers
