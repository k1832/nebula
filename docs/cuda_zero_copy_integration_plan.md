# CUDA Zero-Copy Integration Plan for Nebula

## Executive Summary

This document outlines the plan to implement zero-copy GPU data passing from the Nebula LiDAR driver to downstream Autoware perception nodes using ROS2 Type Adaptation and cuda_blackboard.

**Goal:** Eliminate the CPU-GPU memory transfer bottleneck by keeping decoded point clouds on GPU throughout the entire pipeline.

---

## Current Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CURRENT PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Nebula HesaiDecoder                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ GPU Kernel (decode + filter) → cudaMemcpy D2H → CPU NebulaPoint[]   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                              │
│                    pcl::toROSMsg() conversion                               │
│                              ↓                                              │
│                 sensor_msgs::msg::PointCloud2                               │
│                              ↓                                              │
│                    ROS2 Publisher (CPU)                                     │
│                              ↓                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  CUDA Pointcloud Preprocessor                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Receive PointCloud2 → cudaMemcpy H2D → GPU preprocessing            │   │
│  │ (distortion correction, filtering, cropping)                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                              │
│                    CudaPointCloud2 (on GPU)                                 │
│                              ↓                                              │
│                 cuda_blackboard publish                                     │
│                              ↓                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  GPU Perception Nodes (CenterPoint, Transfusion, BEVFusion, PTv3)          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ cuda_blackboard subscribe → GPU inference (zero-copy)               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

BOTTLENECKS:
  [1] cudaMemcpy D2H in Nebula (~1-2ms for 100k+ points)
  [2] cudaMemcpy H2D in CUDA Preprocessor (~1-2ms)
```

---

## Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PROPOSED PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Nebula HesaiDecoder (gpu_pipeline_mode=true)                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ GPU Kernel (decode + filter) → Format conversion kernel (GPU)       │   │
│  │ → CudaPointCloud2 (stays on GPU)                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                              │
│                 cuda_blackboard publish (zero-copy)                         │
│                              ↓                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  CUDA Pointcloud Preprocessor (optional, can be bypassed)                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ cuda_blackboard subscribe → GPU preprocessing (zero-copy)           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                              │
│                 cuda_blackboard publish (zero-copy)                         │
│                              ↓                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  GPU Perception Nodes (CenterPoint, Transfusion, BEVFusion, PTv3)          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ cuda_blackboard subscribe → GPU inference (zero-copy)               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

ELIMINATED BOTTLENECKS:
  [1] ✅ No D2H copy in Nebula
  [2] ✅ No H2D copy in CUDA Preprocessor
```

---

## How ROS2 Type Adaptation + cuda_blackboard Works

### ROS2 Type Adaptation (TypeAdapter)

ROS2 Type Adaptation allows custom C++ types to be used with ROS2 publish/subscribe while maintaining compatibility with standard ROS message types.

```cpp
// Definition in cuda_adaptation.hpp
template <>
struct rclcpp::TypeAdapter<cuda_blackboard::CudaPointCloud2, sensor_msgs::msg::PointCloud2>
{
  using is_specialized = std::true_type;
  using custom_type = cuda_blackboard::CudaPointCloud2;
  using ros_message_type = sensor_msgs::msg::PointCloud2;

  // Called when publishing to non-CUDA subscribers
  static void convert_to_ros_message(
    const custom_type & source,
    ros_message_type & destination)
  {
    // Copy metadata (header, width, height, fields, etc.)
    static_cast<ros_message_type&>(destination) = source;
    // Copy data from GPU to CPU
    destination.data.resize(source.row_step * source.height);
    cudaMemcpy(destination.data.data(), source.data.get(),
               destination.data.size(), cudaMemcpyDeviceToHost);
  }

  // Called when receiving from non-CUDA publishers
  static void convert_to_custom(
    const ros_message_type & source,
    custom_type & destination)
  {
    // Copy metadata
    static_cast<ros_message_type&>(destination) = source;
    // Allocate GPU memory and copy data
    destination.data = make_cuda_unique<uint8_t[]>(source.data.size());
    cudaMemcpy(destination.data.get(), source.data.data(),
               source.data.size(), cudaMemcpyHostToDevice);
  }
};

// Enable as ROS message type
RCLCPP_USING_CUSTOM_TYPE_AS_ROS_MESSAGE_TYPE(
  cuda_blackboard::CudaPointCloud2,
  sensor_msgs::msg::PointCloud2);
```

### cuda_blackboard Architecture

The cuda_blackboard provides zero-copy GPU memory sharing between ROS2 nodes:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CUDA BLACKBOARD MECHANISM                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PUBLISHER NODE                          SUBSCRIBER NODE                    │
│  ┌─────────────────────┐                ┌─────────────────────┐            │
│  │ CudaBlackboard-     │                │ CudaBlackboard-     │            │
│  │ Publisher           │                │ Subscriber          │            │
│  └─────────────────────┘                └─────────────────────┘            │
│           │                                      │                          │
│           │ 1. Register data                     │                          │
│           ↓                                      │                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      CUDA BLACKBOARD (Singleton)                     │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │ Instance ID → GPU Data Pointer + Reference Count (Tickets)  │    │   │
│  │  │                                                             │    │   │
│  │  │ id=12345 → { d_ptr=0x7f..., tickets=2 }                    │    │   │
│  │  │ id=12346 → { d_ptr=0x7f..., tickets=1 }                    │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│           │                                      │                          │
│           │ 2. Publish instance ID               │ 3. Query by instance ID  │
│           ↓                                      ↓                          │
│  ┌─────────────────────┐                ┌─────────────────────┐            │
│  │ negotiated_pub_     │ ──────────────→│ negotiated_sub_     │            │
│  │ (publishes uint64)  │   Instance ID  │ (receives uint64)   │            │
│  └─────────────────────┘                └─────────────────────┘            │
│                                                  │                          │
│                                                  │ 4. Get GPU pointer       │
│                                                  ↓                          │
│                                         ┌─────────────────────┐            │
│                                         │ Callback with       │            │
│                                         │ CudaPointCloud2     │            │
│                                         │ (GPU data, no copy) │            │
│                                         └─────────────────────┘            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

KEY FEATURES:
  - Reference counting via "tickets" prevents premature deallocation
  - Negotiated publisher handles mixed CUDA/non-CUDA subscribers
  - Automatic fallback to standard ROS messages for non-CUDA nodes
```

### Negotiation Protocol

```cpp
// Publisher checks subscriber types and publishes accordingly
void publish(CudaPointCloud2::UniquePtr cuda_msg_ptr) {
  // Count CUDA subscribers (intra-process)
  size_t cuda_subs = publisher->get_intra_process_subscription_count();

  // Count standard ROS subscribers
  size_t ros_subs = publisher->get_subscription_count() - cuda_subs;

  if (cuda_subs > 0) {
    // Register in blackboard with ticket count
    uint64_t instance_id = blackboard.registerData(
      producer_name, std::move(cuda_msg_ptr), cuda_subs);

    // Publish only the instance ID (8 bytes, not megabytes!)
    std_msgs::msg::UInt64 id_msg;
    id_msg.data = instance_id;
    negotiated_pub_->publish(id_msg);
  }

  if (ros_subs > 0) {
    // Convert and publish standard ROS message for non-CUDA subscribers
    sensor_msgs::msg::PointCloud2 ros_msg;
    rclcpp::TypeAdapter<CudaPointCloud2, PointCloud2>::convert_to_ros_message(
      *cuda_msg_ptr, ros_msg);
    ros_pub_->publish(ros_msg);
  }
}
```

---

## Downstream Nodes Inventory

### GPU-Accelerated Perception Nodes (Use cuda_blackboard)

| Node | Package | Subscription | Interface |
|------|---------|--------------|-----------|
| LiDAR CenterPoint | `autoware_lidar_centerpoint` | `~/input/pointcloud` | `CudaPointCloud2` |
| LiDAR Transfusion | `autoware_lidar_transfusion` | `~/input/pointcloud` | `CudaPointCloud2` |
| BEVFusion | `autoware_bevfusion` | `~/input/pointcloud` | `CudaPointCloud2` |
| PTv3 | `autoware_ptv3` | `~/input/pointcloud` | `CudaPointCloud2` |
| Occupancy Grid Map | `autoware_probabilistic_occupancy_grid_map` | `~/input/obstacle_pointcloud` | `PointCloud2` (CUDA kernels internally) |

### GPU Preprocessing Nodes (Use cuda_blackboard)

| Node | Package | Input | Output |
|------|---------|-------|--------|
| CUDA Preprocessor | `autoware_cuda_pointcloud_preprocessor` | `CudaPointCloud2` | `CudaPointCloud2` |
| CUDA Voxel Grid Downsample | `autoware_cuda_pointcloud_preprocessor` | `CudaPointCloud2` | `CudaPointCloud2` |
| CUDA Concatenate & Sync | `autoware_cuda_pointcloud_preprocessor` | `CudaPointCloud2` | `CudaPointCloud2` |

### CPU-Based Nodes (Standard ROS2 interface)

| Node | Package | Interface |
|------|---------|-----------|
| Euclidean Cluster | `autoware_euclidean_cluster` | `sensor_msgs::msg::PointCloud2` |
| Ground Segmentation | `autoware_ground_segmentation` | `sensor_msgs::msg::PointCloud2` |
| Compare Map Filter | `autoware_compare_map_segmentation` | `sensor_msgs::msg::PointCloud2` |
| CPU Preprocessor | `autoware_pointcloud_preprocessor` | `sensor_msgs::msg::PointCloud2` |

---

## Data Format Analysis

### Current CudaNebulaPoint (Kernel Output)

```cpp
struct CudaNebulaPoint {
  float x;              // 4 bytes, offset 0
  float y;              // 4 bytes, offset 4
  float z;              // 4 bytes, offset 8
  float distance;       // 4 bytes, offset 12
  float azimuth;        // 4 bytes, offset 16
  float elevation;      // 4 bytes, offset 20
  float intensity;      // 4 bytes, offset 24 (FLOAT!)
  uint8_t return_type;  // 1 byte,  offset 28
  uint16_t channel;     // 2 bytes, offset 30
  uint8_t in_current_scan; // 1 byte, offset 32 (internal use)
  uint32_t entry_id;    // 4 bytes, offset 36 (internal use)
};  // Total: ~40 bytes with padding
```

### Required PointXYZIRCAEDT (PointCloud2 Format)

```cpp
struct PointXYZIRCAEDT {
  float x;              // 4 bytes, offset 0
  float y;              // 4 bytes, offset 4
  float z;              // 4 bytes, offset 8
  uint8_t intensity;    // 1 byte,  offset 12 (UINT8!)
  uint8_t return_type;  // 1 byte,  offset 13
  uint16_t channel;     // 2 bytes, offset 14
  float azimuth;        // 4 bytes, offset 16
  float elevation;      // 4 bytes, offset 20
  float distance;       // 4 bytes, offset 24
  uint32_t time_stamp;  // 4 bytes, offset 28 (relative to scan start)
};  // Total: 32 bytes
```

### Conversion Requirements

| Field | CudaNebulaPoint | PointXYZIRCAEDT | Conversion |
|-------|-----------------|-----------------|------------|
| x, y, z | float | float | Direct copy |
| intensity | float | uint8_t | Cast + clamp to [0, 255] |
| return_type | uint8_t | uint8_t | Direct copy |
| channel | uint16_t | uint16_t | Direct copy |
| azimuth | float | float | Direct copy |
| elevation | float | float | Direct copy |
| distance | float | float | Direct copy |
| time_stamp | N/A | uint32_t | **Must compute on GPU** |
| in_current_scan | uint8_t | N/A | Filter out (only current scan) |
| entry_id | uint32_t | N/A | Discard |

---

## Implementation Plan

### Phase 4a: Internal GPU API (COMPLETED)

| Task | Status | Description |
|------|--------|-------------|
| GpuPointCloud struct | ✅ Done | Device pointer + count + timestamp |
| get_gpu_pointcloud() API | ✅ Done | Returns GpuPointCloud from decoder |
| gpu_pipeline_mode config | ✅ Done | Skip D2H cudaMemcpy when enabled |
| Skip cudaMemcpy in flush | ✅ Done | Conditional based on config |

### Phase 4b: ROS2 Integration (TODO)

| Task | Complexity | Est. Effort | Description |
|------|------------|-------------|-------------|
| **4b-1: GPU Format Conversion Kernel** | High | 2-3 days | CUDA kernel to convert CudaNebulaPoint[] → PointCloud2 byte layout |
| **4b-2: GPU Time Stamp Computation** | Medium | 1 day | Move get_point_time_relative() logic to GPU |
| **4b-3: Integrate cuda_blackboard** | Medium | 1-2 days | Add cuda_blackboard dependency, create CudaBlackboardPublisher |
| **4b-4: Modify decoder_wrapper.cpp** | Medium | 1 day | Conditional CUDA/ROS publishing based on gpu_pipeline_mode |
| **4b-5: Add ROS Parameter** | Low | 0.5 day | Wire gpu_pipeline_mode to launch configuration |
| **4b-6: Testing & Validation** | Medium | 2 days | Integration tests with downstream nodes |

**Total Estimated Effort:** 7-10 days

---

## Detailed Task Breakdown

### Task 4b-1: GPU Format Conversion Kernel

```cuda
// New kernel in hesai_cuda_kernels.cu

__global__ void convert_to_pointcloud2_format(
    const CudaNebulaPoint* __restrict__ d_input,
    uint8_t* __restrict__ d_output,  // Raw byte buffer for PointCloud2
    const uint32_t point_count,
    const uint64_t scan_timestamp_ns,
    const uint32_t point_step)       // 32 bytes for XYZIRCAEDT
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= point_count) return;

    const CudaNebulaPoint& in = d_input[idx];

    // Skip points not in current scan
    if (!in.in_current_scan) return;

    uint8_t* out = d_output + idx * point_step;

    // Write in PointXYZIRCAEDT layout
    *reinterpret_cast<float*>(out + 0)  = in.x;
    *reinterpret_cast<float*>(out + 4)  = in.y;
    *reinterpret_cast<float*>(out + 8)  = in.z;
    out[12] = static_cast<uint8_t>(fminf(fmaxf(in.intensity, 0.0f), 255.0f));
    out[13] = in.return_type;
    *reinterpret_cast<uint16_t*>(out + 14) = in.channel;
    *reinterpret_cast<float*>(out + 16) = in.azimuth;
    *reinterpret_cast<float*>(out + 20) = in.elevation;
    *reinterpret_cast<float*>(out + 24) = in.distance;
    // time_stamp computed separately or passed in
    *reinterpret_cast<uint32_t*>(out + 28) = compute_time_stamp(in, scan_timestamp_ns);
}
```

### Task 4b-3: cuda_blackboard Integration

```cpp
// In decoder_wrapper.hpp

#include <cuda_blackboard/cuda_blackboard_publisher.hpp>
#include <cuda_blackboard/cuda_pointcloud2.hpp>

class HesaiDecoderWrapper {
  // Existing publishers (for CPU path)
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr nebula_points_pub_;

  // New CUDA publisher (for GPU path)
#ifdef NEBULA_CUDA_ENABLED
  std::unique_ptr<cuda_blackboard::CudaBlackboardPublisher<
    cuda_blackboard::CudaPointCloud2>> cuda_points_pub_;
#endif
};
```

### Task 4b-4: Conditional Publishing

```cpp
// In decoder_wrapper.cpp

void HesaiDecoderWrapper::on_pointcloud_decoded(
    const drivers::NebulaPointCloudPtr& pointcloud,
    double timestamp_s)
{
#ifdef NEBULA_CUDA_ENABLED
    if (sensor_cfg_->gpu_pipeline_mode) {
        // GPU PATH: Get GPU point cloud and publish via cuda_blackboard
        auto gpu_cloud = driver_ptr_->get_gpu_pointcloud();
        if (gpu_cloud.valid) {
            publish_cuda_pointcloud(gpu_cloud, timestamp_s);
            return;
        }
    }
#endif

    // CPU PATH: Existing logic
    auto ros_pc_msg_ptr = std::make_unique<sensor_msgs::msg::PointCloud2>();
    pcl::toROSMsg(*pointcloud, *ros_pc_msg_ptr);
    // ... publish via standard ROS publisher
}

void HesaiDecoderWrapper::publish_cuda_pointcloud(
    const cuda::GpuPointCloud& gpu_cloud,
    double timestamp_s)
{
    auto cuda_msg = std::make_unique<cuda_blackboard::CudaPointCloud2>();

    // Set metadata
    cuda_msg->header.stamp = rclcpp::Time(seconds_to_chrono_nano_seconds(timestamp_s).count());
    cuda_msg->header.frame_id = sensor_cfg_->frame_id;
    cuda_msg->height = 1;
    cuda_msg->width = gpu_cloud.point_count;
    cuda_msg->point_step = 32;  // sizeof(PointXYZIRCAEDT)
    cuda_msg->row_step = cuda_msg->point_step * cuda_msg->width;
    cuda_msg->is_dense = true;
    cuda_msg->is_bigendian = false;

    // Set field descriptors
    cuda_msg->fields = create_xyzircaedt_fields();

    // Allocate GPU buffer and run conversion kernel
    cuda_msg->data = cuda_blackboard::make_cuda_unique<uint8_t[]>(cuda_msg->row_step);

    launch_convert_to_pointcloud2_format(
        gpu_cloud.d_points,
        cuda_msg->data.get(),
        gpu_cloud.point_count,
        gpu_cloud.timestamp_ns,
        cuda_msg->point_step,
        cuda_stream_);

    cudaStreamSynchronize(cuda_stream_);

    // Publish via cuda_blackboard (zero-copy to GPU subscribers)
    cuda_points_pub_->publish(std::move(cuda_msg));
}
```

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Format conversion adds latency | Medium | Kernel is lightweight (~0.5ms), still faster than D2H+H2D |
| Downstream compatibility | High | Maintain CPU path as fallback, test with all perception nodes |
| Memory management complexity | Medium | Use cuda_blackboard's reference counting |
| Time stamp accuracy on GPU | Low | Port existing CPU logic carefully |

---

## Testing Strategy

1. **Unit Tests**
   - Format conversion kernel correctness
   - Time stamp computation accuracy
   - Memory allocation/deallocation

2. **Integration Tests**
   - Nebula → CUDA Preprocessor → CenterPoint pipeline
   - Mixed CUDA/non-CUDA subscriber scenarios
   - Fallback to CPU path when gpu_pipeline_mode=false

3. **Performance Benchmarks**
   - End-to-end latency comparison (current vs proposed)
   - GPU memory usage
   - CPU utilization reduction

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Nebula decode + publish latency | ~5-10ms | ~2-3ms |
| GPU→CPU→GPU transfers per scan | 2 | 0 |
| CPU utilization (decode thread) | ~30% | ~10% |

---

## Dependencies

| Package | Version | Notes |
|---------|---------|-------|
| cuda_blackboard | Existing | `/src/universe/external/cuda_blackboard/` |
| negotiated | Existing | Used by cuda_blackboard |
| CUDA Toolkit | 11.4+ | Already required for NEBULA_CUDA_ENABLED |

---

## File Changes Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `hesai_cuda_kernels.cu` | Modify | Add format conversion kernel |
| `hesai_cuda_kernels.hpp` | Modify | Add kernel declaration |
| `hesai_decoder.hpp` | Modify | Add conversion buffer allocation |
| `decoder_wrapper.hpp` | Modify | Add CudaBlackboardPublisher |
| `decoder_wrapper.cpp` | Modify | Add conditional CUDA publishing |
| `CMakeLists.txt` | Modify | Add cuda_blackboard dependency |
| `package.xml` | Modify | Add cuda_blackboard dependency |

---

## References

- [ROS2 Type Adaptation](https://docs.ros.org/en/humble/Concepts/About-Type-Adaptation.html)
- [cuda_blackboard source](/home/keitamorisaki/autoware/src/universe/external/cuda_blackboard/)
- [Phase 4a Implementation](/home/keitamorisaki/autoware/src/sensor_component/external/nebula/src/nebula_hesai/nebula_hesai_decoders/include/nebula_hesai_decoders/cuda/hesai_cuda_decoder.hpp)
