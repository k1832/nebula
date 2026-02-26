#!/bin/bash
# Nebula CPU vs GPU Benchmark Runner
# Usage: ./benchmark_runner.bash [--cpu|--gpu] [options]

set -eo pipefail

# === Configuration ===
# Auto-detect rosbag path (different in CPU vs GPU branches)
if [[ -d "$HOME/autoware/src/sensor_component/external/nebula/nebula_tests/data/hesai/ot128/1730271167765338806" ]]; then
    DEFAULT_ROSBAG="$HOME/autoware/src/sensor_component/external/nebula/nebula_tests/data/hesai/ot128/1730271167765338806"
elif [[ -d "$HOME/autoware/src/sensor_component/external/nebula/src/nebula_hesai/nebula_hesai/test_resources/decoder_ground_truth/ot128/1730271167765338806" ]]; then
    DEFAULT_ROSBAG="$HOME/autoware/src/sensor_component/external/nebula/src/nebula_hesai/nebula_hesai/test_resources/decoder_ground_truth/ot128/1730271167765338806"
else
    DEFAULT_ROSBAG=""
fi
ROSBAG_PATH="${ROSBAG_PATH:-$DEFAULT_ROSBAG}"
SENSOR_MODEL="${SENSOR_MODEL:-Pandar128E4X}"
N_ITERATIONS="${N_ITERATIONS:-10}"
RUNTIME_SECS="${RUNTIME_SECS:-30}"
OUTPUT_DIR="${OUTPUT_DIR:-./benchmark_output}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODE="cpu"  # cpu or gpu
NEBULA_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)
AUTOWARE_DIR="${AUTOWARE_DIR:-$HOME/autoware}"
N_CORES=$(nproc --all)
TASKSET_CORES="${TASKSET_CORES:-0-$((N_CORES - 1))}"
MAXFREQ="${MAXFREQ:-2500000}"

print_usage() {
    echo "Nebula CPU vs GPU Benchmark Runner"
    echo ""
    echo "Usage: $0 [--cpu|--gpu] [options]"
    echo ""
    echo "Options:"
    echo "  --cpu              Run CPU benchmark (default)"
    echo "  --gpu              Run GPU benchmark"
    echo "  -m, --sensor-model Sensor model (default: $SENSOR_MODEL)"
    echo "  -n, --n-iterations Number of iterations (default: $N_ITERATIONS)"
    echo "  -t, --runtime      Runtime per iteration in seconds (default: $RUNTIME_SECS)"
    echo "  -o, --output-dir   Output directory (default: $OUTPUT_DIR)"
    echo "  -b, --rosbag-path  Path to rosbag (default: $ROSBAG_PATH)"
    echo "  -f, --maxfreq      CPU frequency in Hz (default: $MAXFREQ)"
    echo "  -c, --taskset-cores Cores to pin to (default: $TASKSET_CORES)"
    echo "  -h, --help         Show this help"
    echo ""
    echo "Example:"
    echo "  $0 --cpu -n 5 -t 20"
    echo "  $0 --gpu -m Pandar128E4X"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu)
            MODE="cpu"
            shift
            ;;
        --gpu)
            MODE="gpu"
            shift
            ;;
        -m|--sensor-model)
            SENSOR_MODEL="$2"
            shift 2
            ;;
        -n|--n-iterations)
            N_ITERATIONS="$2"
            shift 2
            ;;
        -t|--runtime)
            RUNTIME_SECS="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -b|--rosbag-path)
            ROSBAG_PATH="$2"
            shift 2
            ;;
        -f|--maxfreq)
            MAXFREQ="$2"
            shift 2
            ;;
        -c|--taskset-cores)
            TASKSET_CORES="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Validate rosbag path
if [[ ! -d "$ROSBAG_PATH" && ! -f "$ROSBAG_PATH" ]]; then
    echo "Error: Rosbag path does not exist: $ROSBAG_PATH"
    exit 1
fi

# Create output directory
OUTPUT_DIR="${OUTPUT_DIR}_${MODE}"
mkdir -p "$OUTPUT_DIR"

echo "=== Nebula Benchmark Configuration ==="
echo "Mode:        $MODE"
echo "Sensor:      $SENSOR_MODEL"
echo "Iterations:  $N_ITERATIONS"
echo "Runtime/iter: ${RUNTIME_SECS}s"
echo "Output:      $OUTPUT_DIR"
echo "Rosbag:      $ROSBAG_PATH"
echo "CPU Freq:    $MAXFREQ Hz"
echo "Cores:       $TASKSET_CORES"
echo ""

lock_cpu() {
    echo "Locking CPU frequency to $MAXFREQ Hz..."
    for policy in /sys/devices/system/cpu/cpufreq/policy*; do
        echo userspace | sudo tee "$policy/scaling_governor" >/dev/null 2>&1 || true
        echo "$MAXFREQ" | sudo tee "$policy/scaling_setspeed" >/dev/null 2>&1 || true
    done
    sudo sh -c "echo 0 > /sys/devices/system/cpu/cpufreq/boost" 2>/dev/null || true
    echo "CPU frequency locked"
}

unlock_cpu() {
    echo "Unlocking CPU frequency..."
    for policy in /sys/devices/system/cpu/cpufreq/policy*; do
        echo schedutil | sudo tee "$policy/scaling_governor" >/dev/null 2>&1 || true
    done
    sudo sh -c "echo 1 > /sys/devices/system/cpu/cpufreq/boost" 2>/dev/null || true
    echo "CPU frequency unlocked"
}

run_iteration() {
    local iter=$1
    local output_file="$OUTPUT_DIR/${MODE}_${TIMESTAMP}_iter${iter}.log"

    echo "  Iteration $iter/$N_ITERATIONS -> $output_file"

    # Detect launch file: GPU branch uses different path
    local launch_cmd
    if ros2 pkg list 2>/dev/null | grep -q "^nebula_ros$"; then
        # CPU branch: nebula_ros package
        launch_cmd="ros2 launch nebula_ros nebula_launch.py"
    elif [[ -f "$NEBULA_DIR/src/nebula/launch/nebula_launch.py" ]]; then
        # GPU branch: launch from source
        launch_cmd="ros2 launch $NEBULA_DIR/src/nebula/launch/nebula_launch.py"
    else
        echo "Error: Cannot find nebula launch file"
        return 1
    fi

    # Start nebula driver with stderr captured (where PROFILING goes)
    timeout -s INT "$RUNTIME_SECS" taskset -c "$TASKSET_CORES" \
        $launch_cmd \
        sensor_model:="$SENSOR_MODEL" \
        launch_hw:=false \
        > "$output_file" 2>&1 &
    local driver_pid=$!

    sleep 5  # Wait for initialization (GPU needs more time)

    # Play rosbag in loop
    timeout -s KILL $((RUNTIME_SECS - 7)) ros2 bag play -l "$ROSBAG_PATH" >/dev/null 2>&1 &
    local bag_pid=$!

    wait $driver_pid $bag_pid 2>/dev/null || true
}

# === Main Execution ===
trap unlock_cpu EXIT

# Source workspace
if [[ -f "$AUTOWARE_DIR/install/setup.bash" ]]; then
    source "$AUTOWARE_DIR/install/setup.bash"
elif [[ -f "$NEBULA_DIR/install/setup.bash" ]]; then
    source "$NEBULA_DIR/install/setup.bash"
else
    echo "Error: Cannot find setup.bash in $AUTOWARE_DIR/install or $NEBULA_DIR/install"
    exit 1
fi

# Restart ROS2 daemon
ros2 daemon stop 2>/dev/null || true
ros2 daemon start

# Lock CPU frequency for consistent results
lock_cpu

echo "=== Running $N_ITERATIONS iterations ($MODE mode) ==="
for ((i = 1; i <= N_ITERATIONS; i++)); do
    run_iteration $i
    sleep 2
done

echo ""
echo "=== Benchmark Complete ==="
echo "Logs saved to: $OUTPUT_DIR"
echo ""
echo "PROFILING entries found:"
grep -c "PROFILING" "$OUTPUT_DIR"/*.log 2>/dev/null | tail -5 || echo "  (none found)"
echo ""
echo "Next steps:"
echo "  1. Verify PROFILING output: grep PROFILING $OUTPUT_DIR/*.log | head"
echo "  2. Analyze results: python3 scripts/analyze_benchmark.py --log-dir $OUTPUT_DIR"
