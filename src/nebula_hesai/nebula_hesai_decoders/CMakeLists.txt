cmake_minimum_required(VERSION 3.20)
project(nebula_hesai_decoders)

find_package(autoware_cmake REQUIRED)
autoware_package()

# Option to enable CUDA support
option(BUILD_CUDA "Build with CUDA support" ON)

# CUDA Configuration
if(BUILD_CUDA)
  find_package(CUDA)
  if(NOT ${CUDA_FOUND})
    message(WARNING "CUDA not found, building without CUDA support")
    set(BUILD_CUDA OFF)
  else()
    message(STATUS "CUDA found: ${CUDA_VERSION}")
    message(STATUS "CUDA include dirs: ${CUDA_INCLUDE_DIRS}")

    # CUDA compiler flags
    list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
    list(APPEND CUDA_NVCC_FLAGS "-diag-suppress 20012")

    # GPU architectures
    list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_75,code=sm_75")  # Turing
    list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_86,code=sm_86")  # Ampere
    list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_87,code=sm_87")  # Jetson Orin
    list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_89,code=sm_89")  # Ada
    list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_89,code=compute_89")  # PTX

    # Extended lambda support
    list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")

    # Optimization flags
    list(APPEND CUDA_NVCC_FLAGS "-O3")

    # Check if CUDA kernel source exists
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/hesai_cuda_kernels.cu")
      set(CUDA_KERNELS_EXIST TRUE)
      message(STATUS "CUDA kernel source found, will build CUDA decoder")
    else()
      set(CUDA_KERNELS_EXIST FALSE)
      message(WARNING "CUDA kernel source (src/cuda/hesai_cuda_kernels.cu) not found")
      message(WARNING "Building without CUDA decoder support. To enable CUDA:")
      message(WARNING "  1. Create the src/cuda/ directory")
      message(WARNING "  2. Implement hesai_cuda_kernels.cu with CUDA kernels")
    endif()
  endif()
endif()

# Hesai Decoder library (CPU)
add_library(nebula_hesai_decoders SHARED src/hesai_driver.cpp)

target_include_directories(
  nebula_hesai_decoders
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include>)

target_link_libraries(
  nebula_hesai_decoders
  PUBLIC nebula_core_common::nebula_core_common
         nebula_hesai_common::nebula_hesai_common
         nebula_core_decoders::nebula_core_decoders rclcpp::rclcpp)

# CUDA Decoder library
if(BUILD_CUDA AND CUDA_KERNELS_EXIST)
  # Use modern CMake CUDA support
  enable_language(CUDA)

  # Set CUDA architectures
  set(CMAKE_CUDA_ARCHITECTURES "75;86;87;89")

  # Set CUDA-specific flags (avoid passing C++ flags to nvcc)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --expt-relaxed-constexpr")

  add_library(nebula_hesai_decoders_cuda SHARED
    src/cuda/hesai_cuda_kernels.cu
  )

  set_target_properties(nebula_hesai_decoders_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    # Don't pass host compiler flags to nvcc
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )

  # Clear any inherited compile options that might be incompatible with CUDA
  set_property(TARGET nebula_hesai_decoders_cuda PROPERTY COMPILE_OPTIONS "")

  target_include_directories(
    nebula_hesai_decoders_cuda
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include>)

  target_link_libraries(nebula_hesai_decoders_cuda
    PRIVATE cudart)

  target_compile_definitions(nebula_hesai_decoders_cuda PUBLIC NEBULA_CUDA_ENABLED)
  target_compile_definitions(nebula_hesai_decoders PUBLIC NEBULA_CUDA_ENABLED)

  # Add CUDA include directories to main library for header resolution
  target_include_directories(nebula_hesai_decoders
    PUBLIC ${CUDA_INCLUDE_DIRS})

  # Link CUDA library to main library
  target_link_libraries(nebula_hesai_decoders PUBLIC nebula_hesai_decoders_cuda ${CUDA_LIBRARIES})

  install(TARGETS nebula_hesai_decoders_cuda EXPORT export_nebula_hesai_decoders_cuda)
  ament_export_targets(export_nebula_hesai_decoders_cuda)
endif()

install(TARGETS nebula_hesai_decoders EXPORT export_nebula_hesai_decoders)
install(DIRECTORY include/${PROJECT_NAME}/ DESTINATION include/${PROJECT_NAME})

install(DIRECTORY calibration DESTINATION share/${PROJECT_NAME})

ament_export_include_directories("include/${PROJECT_NAME}")
ament_export_targets(export_nebula_hesai_decoders)

ament_export_dependencies(nebula_core_common nebula_hesai_common
                          nebula_core_decoders rclcpp)

ament_package()
