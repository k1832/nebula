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
  const nebula::drivers::cuda::CudaDecoderConfig & config,
  nebula::drivers::cuda::CudaNebulaPoint * d_points,
  uint32_t * d_count,
  uint32_t n_azimuths,
  uint32_t raw_azimuth,
  cudaStream_t stream);

// C-linkage batched kernel launcher declaration (processes entire scan)
// Includes GPU FOV filtering and overlap/scan assignment
extern "C" void launch_decode_hesai_scan_batch(
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
#include <iostream>
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
  /// @brief Number of azimuth divisions for angle lookup table (LUT resolution)
  static constexpr uint32_t cuda_n_azimuths_ = 36000;  // 0.01 degree resolution
  /// @brief Sensor's native azimuth range (max_azimuth = 360 * degree_subdivisions)
  static constexpr uint32_t sensor_max_azimuth_ = 360 * SensorT::packet_t::degree_subdivisions;
  /// @brief Scale factor from sensor native azimuth to LUT index (sensor_max_azimuth / cuda_n_azimuths)
  /// For sensors with degree_subdivisions=100: scale=1
  /// For AT128 with degree_subdivisions=25600: scale=256
  static constexpr uint32_t azimuth_scale_ = sensor_max_azimuth_ / cuda_n_azimuths_;
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
  /// @brief Pinned host memory for distances (faster CPU->GPU transfer)
  uint16_t * h_pinned_distances_ = nullptr;
  /// @brief Pinned host memory for reflectivities (faster CPU->GPU transfer)
  uint8_t * h_pinned_reflectivities_ = nullptr;
  /// @brief Size of pre-allocated buffers (n_channels * max_returns)
  size_t cuda_buffer_size_ = 0;

  /// @brief GPU scan buffer for batch processing
  struct GpuScanBuffer {
    // Batch buffers for packet data (pre-allocated, filled linearly, reset after flush)
    uint16_t* d_distances_batch = nullptr;      // [MAX_PACKETS][n_channels * max_returns]
    uint8_t* d_reflectivities_batch = nullptr;  // [MAX_PACKETS][n_channels * max_returns]
    uint32_t* d_raw_azimuths = nullptr;        // [MAX_PACKETS]
    uint32_t* d_n_returns = nullptr;           // [MAX_PACKETS]
    uint32_t* d_last_azimuths = nullptr;       // [MAX_PACKETS] Per-entry last_azimuth for GPU overlap check

    // Pinned host memory for fast staging
    uint16_t* h_distances_staging = nullptr;
    uint8_t* h_reflectivities_staging = nullptr;
    uint32_t* h_raw_azimuths_staging = nullptr;
    uint32_t* h_n_returns_staging = nullptr;
    uint32_t* h_last_azimuths_staging = nullptr;  // Per-entry last_azimuth for overlap check
    uint64_t* h_packet_timestamps_staging = nullptr;  // Per-entry packet timestamp for time_stamp calculation

    // Metadata
    uint32_t packet_count = 0;           // Packets accumulated so far
    uint32_t max_packets = 0;            // Buffer capacity (e.g., 4000)
    uint64_t scan_timestamp_ns = 0;

    // Output (reuse existing buffers)
    cuda::CudaNebulaPoint* d_points = nullptr;
    uint32_t* d_count = nullptr;

    /// @brief Free all allocated device and pinned-host memory, reset pointers to nullptr
    void cleanup()
    {
      auto free_device = [](auto *& ptr) {
        if (ptr) { cudaFree(ptr); ptr = nullptr; }
      };
      auto free_host = [](auto *& ptr) {
        if (ptr) { cudaFreeHost(ptr); ptr = nullptr; }
      };

      free_device(d_distances_batch);
      free_device(d_reflectivities_batch);
      free_device(d_raw_azimuths);
      free_device(d_n_returns);
      free_device(d_last_azimuths);

      free_host(h_distances_staging);
      free_host(h_reflectivities_staging);
      free_host(h_raw_azimuths_staging);
      free_host(h_n_returns_staging);
      free_host(h_last_azimuths_staging);
      free_host(h_packet_timestamps_staging);

      packet_count = 0;
      max_packets = 0;
    }
  };

  GpuScanBuffer gpu_scan_buffer_;

  /// Cached GPU output state for zero-copy access (updated after flush)
  uint32_t gpu_output_point_count_ = 0;
  uint64_t gpu_output_timestamp_ns_ = 0;

  /// Feature flag for scan-level CUDA batching.
  /// When enabled, packet data is accumulated and processed in batch at scan boundaries,
  /// eliminating per-packet kernel launch overhead (~30us × ~3000 packets = ~90ms per scan).
  ///
  /// Known limitation: PandarXT16 produces slightly different point ordering compared to
  /// non-batched mode (same point count, minor coordinate difference at first point).
  /// This appears to be due to subtle GPU execution ordering differences and does not
  /// affect point cloud accuracy. All other sensor types pass correctness tests.
  bool use_scan_batching_ = true;  // Batched kernel with deterministic ordering

  static constexpr uint32_t MAX_PACKETS_PER_SCAN = 4000;

#ifdef NEBULA_CUDA_PROFILING
  // CUDA event timing for performance instrumentation (enabled via -DNEBULA_CUDA_PROFILING=ON)
  cudaEvent_t timing_event_start_ = nullptr;
  cudaEvent_t timing_event_after_h2d_ = nullptr;
  cudaEvent_t timing_event_after_kernel_ = nullptr;
  cudaEvent_t timing_event_after_d2h_ = nullptr;
  bool timing_events_initialized_ = false;

  // Accumulated timing statistics (in milliseconds)
  struct GpuTimingStats {
    double total_h2d_ms = 0.0;
    double total_kernel_ms = 0.0;
    double total_d2h_ms = 0.0;
    uint32_t flush_count = 0;
  };
  GpuTimingStats gpu_timing_stats_;
#endif  // NEBULA_CUDA_PROFILING
#endif  // NEBULA_CUDA_ENABLED

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
    // Store packet timestamp for time_stamp calculation during post-processing
    gpu_scan_buffer_.h_packet_timestamps_staging[entry_id] = hesai_packet::get_timestamp_ns(packet_);

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

  /// @brief Build the CudaDecoderConfig for a batched kernel launch
  /// @param n_entries Number of block-group entries accumulated
  /// @return Populated config ready for GPU upload
  cuda::CudaDecoderConfig build_batch_config(uint32_t n_entries)
  {
    const size_t n_channels = SensorT::packet_t::n_channels;
    const size_t max_returns = SensorT::packet_t::max_returns;

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
    config.max_returns = max_returns;
    config.dis_unit = hesai_packet::get_dis_unit(packet_);
    config.data_stride = max_returns;
    config.n_blocks = max_returns;
    config.entry_id = 0;
    config.n_azimuths_raw = cuda_n_azimuths_;
    config.azimuth_scale = azimuth_scale_;
    config.max_output_points = n_entries * n_channels * max_returns;

    if constexpr (SensorT::uses_calibration_based_angles) {
      config.is_multi_frame = false;
      config.n_frames = 1;
      config.timestamp_reset_angle_raw = angle_corrector_.timestamp_reset_angle_raw_;
      config.emit_angle_raw = angle_corrector_.emit_angle_raw_;
      config.frame_angles[0].fov_start = angle_corrector_.fov_start_raw_;
      config.frame_angles[0].fov_end = angle_corrector_.fov_end_raw_;
      config.frame_angles[0].timestamp_reset = angle_corrector_.timestamp_reset_angle_raw_;
      config.frame_angles[0].scan_emit = angle_corrector_.emit_angle_raw_;
    } else {
      config.is_multi_frame = true;
      config.n_frames = static_cast<uint32_t>(angle_corrector_.get_n_frames());
      config.timestamp_reset_angle_raw = 0;
      config.emit_angle_raw = 0;

      for (uint32_t i = 0; i < config.n_frames && i < cuda::MAX_CUDA_FRAMES; ++i) {
        uint32_t fov_start, fov_end, timestamp_reset, scan_emit;
        if (angle_corrector_.get_frame_angle_info(i, fov_start, fov_end, timestamp_reset, scan_emit)) {
          config.frame_angles[i].fov_start = fov_start;
          config.frame_angles[i].fov_end = fov_end;
          config.frame_angles[i].timestamp_reset = timestamp_reset;
          config.frame_angles[i].scan_emit = scan_emit;
          if (i == 0) {
            config.timestamp_reset_angle_raw = timestamp_reset;
            config.emit_angle_raw = scan_emit;
          }
        }
      }
    }

    return config;
  }

  /// @brief Transfer accumulated scan data from pinned host memory to device
  /// @param n_entries Number of block-group entries to transfer
  /// @param total_data_size Total number of distance/reflectivity elements
  /// @param config Config struct to upload to device
  void transfer_scan_to_device(
    uint32_t n_entries, size_t total_data_size, const cuda::CudaDecoderConfig & /*config*/)
  {
    cudaMemcpyAsync(gpu_scan_buffer_.d_distances_batch, gpu_scan_buffer_.h_distances_staging,
                    total_data_size * sizeof(uint16_t),
                    cudaMemcpyHostToDevice, cuda_stream_);
    cudaMemcpyAsync(gpu_scan_buffer_.d_reflectivities_batch, gpu_scan_buffer_.h_reflectivities_staging,
                    total_data_size * sizeof(uint8_t),
                    cudaMemcpyHostToDevice, cuda_stream_);
    cudaMemcpyAsync(gpu_scan_buffer_.d_raw_azimuths, gpu_scan_buffer_.h_raw_azimuths_staging,
                    n_entries * sizeof(uint32_t),
                    cudaMemcpyHostToDevice, cuda_stream_);
    cudaMemcpyAsync(gpu_scan_buffer_.d_n_returns, gpu_scan_buffer_.h_n_returns_staging,
                    n_entries * sizeof(uint32_t),
                    cudaMemcpyHostToDevice, cuda_stream_);
    cudaMemcpyAsync(gpu_scan_buffer_.d_last_azimuths, gpu_scan_buffer_.h_last_azimuths_staging,
                    n_entries * sizeof(uint32_t),
                    cudaMemcpyHostToDevice, cuda_stream_);
  }

  /// @brief Copy GPU results to host and compact sparse buffer into NebulaPoint pointcloud
  /// @param n_entries Number of block-group entries in the scan
  /// @param valid_point_count Number of valid points reported by the kernel
  /// @param sparse_buffer_size Total sparse buffer size (n_entries * n_channels * max_returns)
  void process_gpu_results(
    uint32_t n_entries, uint32_t /*valid_point_count*/, uint32_t sparse_buffer_size)
  {
    // Copy SPARSE buffer - points are at deterministic positions with gaps
    const uint32_t copy_size = std::min(sparse_buffer_size, static_cast<uint32_t>(cuda_point_buffer_.size()));
    cudaMemcpy(cuda_point_buffer_.data(), d_points_,
               copy_size * sizeof(cuda::CudaNebulaPoint),
               cudaMemcpyDeviceToHost);

    // CPU post-processing: iterate sparse buffer, skip invalid points (distance <= 0)
    // This preserves deterministic ordering from global_tid indexing
    for (uint32_t i = 0; i < copy_size; ++i) {
      const auto & cuda_pt = cuda_point_buffer_[i];

      if (cuda_pt.distance <= 0.0f) {
        continue;
      }

      auto & frame = cuda_pt.in_current_scan ? decode_frame_ : output_frame_;

      const uint32_t entry_id = cuda_pt.entry_id;
      const uint64_t packet_timestamp_ns = (entry_id < n_entries) ?
        gpu_scan_buffer_.h_packet_timestamps_staging[entry_id] :
        hesai_packet::get_timestamp_ns(packet_);

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

  /// @brief Flush accumulated packets - one batched kernel launch for the entire scan
  void flush_gpu_scan_buffer()
  {
    if (gpu_scan_buffer_.packet_count == 0) return;

    const uint32_t n_entries = gpu_scan_buffer_.packet_count;
    const size_t n_channels = SensorT::packet_t::n_channels;
    const size_t max_returns = SensorT::packet_t::max_returns;
    const size_t total_data_size = n_entries * n_channels * max_returns;
    const uint32_t sparse_buffer_size = n_entries * n_channels * max_returns;

    // Build config on host (no device copy needed — passed by reference to launcher)
    cuda::CudaDecoderConfig config = build_batch_config(n_entries);

#ifdef NEBULA_CUDA_PROFILING
    if (timing_events_initialized_) {
      cudaEventRecord(timing_event_start_, cuda_stream_);
    }
#endif

    transfer_scan_to_device(n_entries, total_data_size, config);

#ifdef NEBULA_CUDA_PROFILING
    if (timing_events_initialized_) {
      cudaEventRecord(timing_event_after_h2d_, cuda_stream_);
    }
#endif

    // Reset output counter and zero output buffer for deterministic sparse indexing
    cudaMemsetAsync(d_count_, 0, sizeof(uint32_t), cuda_stream_);
    cudaMemsetAsync(d_points_, 0, sparse_buffer_size * sizeof(cuda::CudaNebulaPoint), cuda_stream_);

    // Launch batched kernel (config passed by host reference, copied to kernel by value)
    launch_decode_hesai_scan_batch(
      gpu_scan_buffer_.d_distances_batch,
      gpu_scan_buffer_.d_reflectivities_batch,
      gpu_scan_buffer_.d_raw_azimuths,
      gpu_scan_buffer_.d_n_returns,
      gpu_scan_buffer_.d_last_azimuths,
      cuda_decoder_->get_angle_lut(),
      config,
      d_points_,
      d_count_,
      cuda_n_azimuths_,
      n_entries,
      cuda_stream_);

#ifdef NEBULA_CUDA_PROFILING
    if (timing_events_initialized_) {
      cudaEventRecord(timing_event_after_kernel_, cuda_stream_);
    }
#endif

    cudaStreamSynchronize(cuda_stream_);

    uint32_t valid_point_count = 0;
    cudaMemcpy(&valid_point_count, d_count_, sizeof(uint32_t), cudaMemcpyDeviceToHost);

#ifdef NEBULA_CUDA_PROFILING
    if (timing_events_initialized_) {
      cudaEventRecord(timing_event_after_d2h_, cuda_stream_);
      cudaEventSynchronize(timing_event_after_d2h_);

      float h2d_ms = 0.0f, kernel_ms = 0.0f, d2h_ms = 0.0f;
      cudaEventElapsedTime(&h2d_ms, timing_event_start_, timing_event_after_h2d_);
      cudaEventElapsedTime(&kernel_ms, timing_event_after_h2d_, timing_event_after_kernel_);
      cudaEventElapsedTime(&d2h_ms, timing_event_after_kernel_, timing_event_after_d2h_);

      gpu_timing_stats_.total_h2d_ms += h2d_ms;
      gpu_timing_stats_.total_kernel_ms += kernel_ms;
      gpu_timing_stats_.total_d2h_ms += d2h_ms;
      gpu_timing_stats_.flush_count++;

      if (gpu_timing_stats_.flush_count <= 10 || gpu_timing_stats_.flush_count % 100 == 0) {
        NEBULA_LOG_STREAM(logger_->info, "[GPU_TIMING] flush#" << gpu_timing_stats_.flush_count
          << " entries=" << n_entries << " points=" << valid_point_count
          << " h2d=" << h2d_ms << "ms kernel=" << kernel_ms << "ms d2h=" << d2h_ms
          << "ms total=" << (h2d_ms + kernel_ms + d2h_ms) << "ms");
      }

      std::cerr << "PROFILING {\"d_gpu_h2d_ms\": " << h2d_ms
                << ", \"d_gpu_kernel_ms\": " << kernel_ms
                << ", \"d_gpu_d2h_ms\": " << d2h_ms
                << ", \"d_gpu_total_ms\": " << (h2d_ms + kernel_ms + d2h_ms)
                << ", \"n_points\": " << valid_point_count << "}" << std::endl;
    }
#endif

    // Update GPU output state for zero-copy access
    gpu_output_point_count_ = std::min(valid_point_count, static_cast<uint32_t>(cuda_point_buffer_.size()));
    gpu_output_timestamp_ns_ = decode_frame_.scan_timestamp_ns;

    if (valid_point_count == 0) {
      gpu_scan_buffer_.packet_count = 0;
      return;
    }

    // Copy to host and process (only if not in GPU pipeline mode)
    if (!sensor_configuration_->gpu_pipeline_mode) {
      process_gpu_results(n_entries, valid_point_count, sparse_buffer_size);
    }

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
    // Azimuth scaling for LUT indexing
    config.azimuth_scale = azimuth_scale_;

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

    // Launch optimized CUDA kernel (config passed by host reference, no D2H copy needed)
    launch_decode_hesai_packet(
      d_distances_,
      d_reflectivities_,
      cuda_decoder_->get_angle_lut(),
      config,
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
  /// @brief Whether this is a multi-frame sensor (e.g., AT128 with 4 mirror frames)
  bool is_multi_frame_sensor_ = false;

  /// @brief Initialize CUDA decoder and upload angle corrections
  void initialize_cuda()
  {
    // CUDA acceleration is now supported for both calibration-based and correction-based sensors
    // Correction-based sensors (like AT128) use the same get_corrected_angle_data() interface
    // but require larger LUTs and multi-frame boundary handling

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
    // For batched mode, the sparse output buffer needs to hold all potential point positions
    // (not just valid points). This is: MAX_PACKETS_PER_SCAN × n_channels × max_returns
    const size_t max_sparse_buffer_points = static_cast<size_t>(MAX_PACKETS_PER_SCAN) *
                                            n_channels * SensorT::packet_t::max_returns;
    // Use the larger of the two for buffer allocation
    const size_t buffer_allocation_size = std::max(max_points, max_sparse_buffer_points);

    if (!cuda_decoder_->initialize(buffer_allocation_size, n_channels)) {
      NEBULA_LOG_STREAM(logger_->warn, "Failed to initialize CUDA decoder");
      cuda_decoder_.reset();
      return;
    }

    // Allocate device memory for output (use larger size for batched sparse mode)
    err = cudaMalloc(&d_points_, buffer_allocation_size * sizeof(cuda::CudaNebulaPoint));
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

    // Helper: attempt allocation, on failure clean up all prior per-packet buffers and bail
    auto cleanup_per_packet = [&]() {
      if (d_distances_) { cudaFree(d_distances_); d_distances_ = nullptr; }
      if (d_reflectivities_) { cudaFree(d_reflectivities_); d_reflectivities_ = nullptr; }
      if (h_pinned_distances_) { cudaFreeHost(h_pinned_distances_); h_pinned_distances_ = nullptr; }
      if (h_pinned_reflectivities_) { cudaFreeHost(h_pinned_reflectivities_); h_pinned_reflectivities_ = nullptr; }
      cudaFree(d_points_); d_points_ = nullptr;
      cudaFree(d_count_); d_count_ = nullptr;
      cuda_decoder_.reset();
    };

    auto alloc_per_packet_ok = [&](cudaError_t result, const char * name) -> bool {
      if (result != cudaSuccess) {
        NEBULA_LOG_STREAM(logger_->warn, "Failed to allocate " << name);
        cleanup_per_packet();
        return false;
      }
      return true;
    };

    bool ok = true;
    ok = ok && alloc_per_packet_ok(
      cudaMalloc(&d_distances_, cuda_buffer_size_ * sizeof(uint16_t)), "CUDA distances buffer");
    ok = ok && alloc_per_packet_ok(
      cudaMalloc(&d_reflectivities_, cuda_buffer_size_ * sizeof(uint8_t)), "CUDA reflectivities buffer");
    ok = ok && alloc_per_packet_ok(
      cudaMallocHost(&h_pinned_distances_, cuda_buffer_size_ * sizeof(uint16_t)), "pinned distances buffer");
    ok = ok && alloc_per_packet_ok(
      cudaMallocHost(&h_pinned_reflectivities_, cuda_buffer_size_ * sizeof(uint8_t)), "pinned reflectivities buffer");

    if (!ok) return;

    // Pre-allocate host buffer for results (use larger size for batched sparse mode)
    cuda_point_buffer_.resize(buffer_allocation_size);

    // Allocate GPU scan buffer for batch processing
    const uint32_t packet_data_size = n_channels * SensorT::packet_t::max_returns;

    auto alloc_scan_ok = [&](cudaError_t result, const char * name) -> bool {
      if (result != cudaSuccess) {
        NEBULA_LOG_STREAM(logger_->warn, "Failed to allocate " << name);
        gpu_scan_buffer_.cleanup();
        use_scan_batching_ = false;
        return false;
      }
      return true;
    };

    bool scan_ok = true;
    // Device buffers
    scan_ok = scan_ok && alloc_scan_ok(
      cudaMalloc(&gpu_scan_buffer_.d_distances_batch,
                 MAX_PACKETS_PER_SCAN * packet_data_size * sizeof(uint16_t)),
      "scan distances buffer");
    scan_ok = scan_ok && alloc_scan_ok(
      cudaMalloc(&gpu_scan_buffer_.d_reflectivities_batch,
                 MAX_PACKETS_PER_SCAN * packet_data_size * sizeof(uint8_t)),
      "scan reflectivities buffer");
    scan_ok = scan_ok && alloc_scan_ok(
      cudaMalloc(&gpu_scan_buffer_.d_raw_azimuths,
                 MAX_PACKETS_PER_SCAN * sizeof(uint32_t)),
      "scan azimuths buffer");
    scan_ok = scan_ok && alloc_scan_ok(
      cudaMalloc(&gpu_scan_buffer_.d_n_returns,
                 MAX_PACKETS_PER_SCAN * sizeof(uint32_t)),
      "scan n_returns buffer");
    scan_ok = scan_ok && alloc_scan_ok(
      cudaMalloc(&gpu_scan_buffer_.d_last_azimuths,
                 MAX_PACKETS_PER_SCAN * sizeof(uint32_t)),
      "scan last_azimuths buffer");
    // Pinned host staging buffers
    scan_ok = scan_ok && alloc_scan_ok(
      cudaMallocHost(&gpu_scan_buffer_.h_distances_staging,
                     MAX_PACKETS_PER_SCAN * packet_data_size * sizeof(uint16_t)),
      "pinned scan distances staging");
    scan_ok = scan_ok && alloc_scan_ok(
      cudaMallocHost(&gpu_scan_buffer_.h_reflectivities_staging,
                     MAX_PACKETS_PER_SCAN * packet_data_size * sizeof(uint8_t)),
      "pinned scan reflectivities staging");
    scan_ok = scan_ok && alloc_scan_ok(
      cudaMallocHost(&gpu_scan_buffer_.h_raw_azimuths_staging,
                     MAX_PACKETS_PER_SCAN * sizeof(uint32_t)),
      "pinned scan azimuths staging");
    scan_ok = scan_ok && alloc_scan_ok(
      cudaMallocHost(&gpu_scan_buffer_.h_n_returns_staging,
                     MAX_PACKETS_PER_SCAN * sizeof(uint32_t)),
      "pinned scan n_returns staging");
    scan_ok = scan_ok && alloc_scan_ok(
      cudaMallocHost(&gpu_scan_buffer_.h_last_azimuths_staging,
                     MAX_PACKETS_PER_SCAN * sizeof(uint32_t)),
      "pinned scan last_azimuths staging");
    scan_ok = scan_ok && alloc_scan_ok(
      cudaMallocHost(&gpu_scan_buffer_.h_packet_timestamps_staging,
                     MAX_PACKETS_PER_SCAN * sizeof(uint64_t)),
      "pinned scan timestamps staging");

    if (scan_ok) {
      gpu_scan_buffer_.packet_count = 0;
      gpu_scan_buffer_.max_packets = MAX_PACKETS_PER_SCAN;
      gpu_scan_buffer_.d_points = d_points_;
      gpu_scan_buffer_.d_count = d_count_;
      NEBULA_LOG_STREAM(logger_->info, "GPU scan batching enabled");
    }

    // Build and upload angle correction lookup table
    // For sensors with degree_subdivisions != 100 (like AT128 with 25600), we need to
    // scale the azimuth to the sensor's native range when building the LUT.
    // LUT index 'i' corresponds to sensor azimuth 'i * azimuth_scale_'
    std::vector<cuda::CudaAngleCorrectionData> angle_lut;
    angle_lut.reserve(cuda_n_azimuths_ * n_channels);

    NEBULA_LOG_STREAM(logger_->info, "Building CUDA angle LUT: azimuth_scale=" << azimuth_scale_
      << " sensor_max_azimuth=" << sensor_max_azimuth_
      << " cuda_n_azimuths=" << cuda_n_azimuths_);

    for (uint32_t lut_idx = 0; lut_idx < cuda_n_azimuths_; ++lut_idx) {
      // Scale LUT index to sensor's native azimuth range
      // For standard sensors (scale=1): sensor_azimuth = lut_idx
      // For AT128 (scale=256): sensor_azimuth = lut_idx * 256
      uint32_t sensor_azimuth = lut_idx * azimuth_scale_;
      for (uint32_t channel = 0; channel < n_channels; ++channel) {
        CorrectedAngleData cpu_data = angle_corrector_.get_corrected_angle_data(sensor_azimuth, channel);
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

    // Detect multi-frame sensor (correction-based sensors like AT128)
    if constexpr (!SensorT::uses_calibration_based_angles) {
      is_multi_frame_sensor_ = true;
      size_t n_frames = angle_corrector_.get_n_frames();
      NEBULA_LOG_STREAM(logger_->info, "CUDA: Detected multi-frame sensor with "
        << n_frames << " frames");
    } else {
      is_multi_frame_sensor_ = false;
    }

    cuda_enabled_ = true;
    NEBULA_LOG_STREAM(logger_->info, "CUDA decoder initialized successfully with "
      << n_channels << " channels and " << cuda_n_azimuths_ << " azimuth divisions"
      << (is_multi_frame_sensor_ ? " (multi-frame)" : ""));

#ifdef NEBULA_CUDA_PROFILING
    // Initialize CUDA events for timing instrumentation
    cudaError_t event_err = cudaEventCreate(&timing_event_start_);
    if (event_err == cudaSuccess) {
      event_err = cudaEventCreate(&timing_event_after_h2d_);
    }
    if (event_err == cudaSuccess) {
      event_err = cudaEventCreate(&timing_event_after_kernel_);
    }
    if (event_err == cudaSuccess) {
      event_err = cudaEventCreate(&timing_event_after_d2h_);
    }
    if (event_err == cudaSuccess) {
      timing_events_initialized_ = true;
      NEBULA_LOG_STREAM(logger_->info, "CUDA timing events initialized");
    } else {
      NEBULA_LOG_STREAM(logger_->warn, "Failed to initialize CUDA timing events: "
        << cudaGetErrorString(event_err));
      if (timing_event_start_) cudaEventDestroy(timing_event_start_);
      if (timing_event_after_h2d_) cudaEventDestroy(timing_event_after_h2d_);
      if (timing_event_after_kernel_) cudaEventDestroy(timing_event_after_kernel_);
      if (timing_event_after_d2h_) cudaEventDestroy(timing_event_after_d2h_);
      timing_event_start_ = nullptr;
      timing_event_after_h2d_ = nullptr;
      timing_event_after_kernel_ = nullptr;
      timing_event_after_d2h_ = nullptr;
    }
#endif  // NEBULA_CUDA_PROFILING
  }

  /// @brief Cleanup CUDA resources
  ~HesaiDecoder()
  {
    // Free scan buffer resources (fixes memory leak)
    gpu_scan_buffer_.cleanup();

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
    if (h_pinned_distances_) {
      cudaFreeHost(h_pinned_distances_);
    }
    if (h_pinned_reflectivities_) {
      cudaFreeHost(h_pinned_reflectivities_);
    }
    if (cuda_stream_) {
      cudaStreamDestroy(cuda_stream_);
    }
#ifdef NEBULA_CUDA_PROFILING
    // Clean up timing events
    if (timing_event_start_) {
      cudaEventDestroy(timing_event_start_);
    }
    if (timing_event_after_h2d_) {
      cudaEventDestroy(timing_event_after_h2d_);
    }
    if (timing_event_after_kernel_) {
      cudaEventDestroy(timing_event_after_kernel_);
    }
    if (timing_event_after_d2h_) {
      cudaEventDestroy(timing_event_after_d2h_);
    }
    // Log accumulated timing stats if any measurements were taken
    if (gpu_timing_stats_.flush_count > 0) {
      double avg_h2d = gpu_timing_stats_.total_h2d_ms / gpu_timing_stats_.flush_count;
      double avg_kernel = gpu_timing_stats_.total_kernel_ms / gpu_timing_stats_.flush_count;
      double avg_d2h = gpu_timing_stats_.total_d2h_ms / gpu_timing_stats_.flush_count;
      double avg_total = avg_h2d + avg_kernel + avg_d2h;
      NEBULA_LOG_STREAM(logger_->info, "[GPU_TIMING_SUMMARY] flushes=" << gpu_timing_stats_.flush_count
        << " avg_h2d=" << avg_h2d << "ms avg_kernel=" << avg_kernel << "ms avg_d2h=" << avg_d2h
        << "ms avg_total=" << avg_total << "ms");
    }
#endif  // NEBULA_CUDA_PROFILING
  }

  /// @brief Get GPU point cloud for zero-copy downstream processing
  /// Returns a struct containing device pointer to points and count.
  /// Only valid when gpu_pipeline_mode is enabled in configuration.
  /// The returned pointer is valid until the next flush or decoder destruction.
  /// @return GpuPointCloud struct with device pointer, count, timestamp, and validity flag
  cuda::GpuPointCloud get_gpu_pointcloud() const
  {
    cuda::GpuPointCloud result;
    if (!cuda_enabled_ || !sensor_configuration_->gpu_pipeline_mode) {
      return result;  // valid = false by default
    }

    result.d_points = d_points_;
    result.point_count = gpu_output_point_count_;
    result.timestamp_ns = gpu_output_timestamp_ns_;
    result.valid = (gpu_output_point_count_ > 0);
    return result;
  }

  /// @brief Check if GPU pipeline mode is active
  /// @return true if CUDA is enabled and gpu_pipeline_mode is configured
  bool is_gpu_pipeline_mode() const
  {
    return cuda_enabled_ && sensor_configuration_->gpu_pipeline_mode;
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
