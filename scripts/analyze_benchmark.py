#!/usr/bin/env python3
"""
Nebula CPU vs GPU Benchmark Analysis Script

Parses PROFILING logs from benchmark runs and computes comparison metrics.

Usage:
    python3 analyze_benchmark.py --log-dir ./benchmark_output_cpu --output cpu_results.json
    python3 analyze_benchmark.py --log-dir ./benchmark_output_gpu --output gpu_results.json
    python3 analyze_benchmark.py --cpu-results cpu_results.json --gpu-results gpu_results.json --compare
"""

import argparse
import glob
import json
import os
import re
import sys
from typing import Dict, List, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: numpy not available, using basic statistics", file=sys.stderr)


def parse_logs(log_dir: str, pattern: str = "*.log") -> List[Dict]:
    """Parse PROFILING entries from log files."""
    records = []
    log_files = glob.glob(os.path.join(log_dir, pattern))

    if not log_files:
        print(f"Warning: No log files found matching {log_dir}/{pattern}", file=sys.stderr)
        return records

    for path in log_files:
        with open(path, 'r') as f:
            for line in f:
                if "PROFILING" in line:
                    match = re.search(r'PROFILING\s+(\{.*?\})', line)
                    if match:
                        try:
                            record = json.loads(match.group(1))
                            record['_source_file'] = os.path.basename(path)
                            records.append(record)
                        except json.JSONDecodeError as e:
                            print(f"Warning: Failed to parse JSON in {path}: {e}", file=sys.stderr)

    print(f"Parsed {len(records)} PROFILING entries from {len(log_files)} log files", file=sys.stderr)
    return records


def compute_stats(values: List[float]) -> Dict:
    """Compute statistical metrics for a list of values."""
    if not values:
        return {'error': 'no data'}

    values = sorted(values)
    n = len(values)

    if HAS_NUMPY:
        arr = np.array(values)
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'p50': float(np.percentile(arr, 50)),
            'p95': float(np.percentile(arr, 95)),
            'p99': float(np.percentile(arr, 99)),
            'max': float(np.max(arr)),
            'count': n
        }
    else:
        # Basic statistics without numpy
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / n
        std = variance ** 0.5

        def percentile(vals, p):
            k = (len(vals) - 1) * p / 100
            f = int(k)
            c = f + 1 if f + 1 < len(vals) else f
            return vals[f] + (vals[c] - vals[f]) * (k - f)

        return {
            'mean': mean,
            'std': std,
            'min': values[0],
            'p50': percentile(values, 50),
            'p95': percentile(values, 95),
            'p99': percentile(values, 99),
            'max': values[-1],
            'count': n
        }


def analyze_cpu_logs(records: List[Dict]) -> Dict:
    """Analyze CPU benchmark logs."""
    results = {'type': 'cpu'}

    # Extract CPU metrics
    unpack_ms = [r['d_cpu_unpack_ms'] for r in records if 'd_cpu_unpack_ms' in r]
    convert_ms = [r['d_cpu_convert_ms'] for r in records if 'd_cpu_convert_ms' in r]
    n_points = [r['n_points'] for r in records if 'n_points' in r]

    if unpack_ms:
        results['cpu_unpack_ms'] = compute_stats(unpack_ms)
    if convert_ms:
        results['cpu_convert_ms'] = compute_stats(convert_ms)
    if n_points:
        results['n_points'] = compute_stats(n_points)

    # CPU total is the unpack time (accumulated across all packets in scan)
    # This is comparable to GPU flush_gpu_scan_buffer (H2D + Kernel + D2H)
    if unpack_ms:
        results['cpu_total_ms'] = results['cpu_unpack_ms']

    return results


def analyze_gpu_logs(records: List[Dict]) -> Dict:
    """Analyze GPU benchmark logs."""
    results = {'type': 'gpu'}

    # Extract GPU metrics
    h2d_ms = [r['d_gpu_h2d_ms'] for r in records if 'd_gpu_h2d_ms' in r]
    kernel_ms = [r['d_gpu_kernel_ms'] for r in records if 'd_gpu_kernel_ms' in r]
    d2h_ms = [r['d_gpu_d2h_ms'] for r in records if 'd_gpu_d2h_ms' in r]
    total_ms = [r['d_gpu_total_ms'] for r in records if 'd_gpu_total_ms' in r]
    n_points = [r['n_points'] for r in records if 'n_points' in r]

    if h2d_ms:
        results['gpu_h2d_ms'] = compute_stats(h2d_ms)
    if kernel_ms:
        results['gpu_kernel_ms'] = compute_stats(kernel_ms)
    if d2h_ms:
        results['gpu_d2h_ms'] = compute_stats(d2h_ms)
    if total_ms:
        results['gpu_total_ms'] = compute_stats(total_ms)
    if n_points:
        results['n_points'] = compute_stats(n_points)

    # Compute zero-copy effective latency (total - d2h)
    if total_ms and d2h_ms and len(total_ms) == len(d2h_ms):
        effective_ms = [t - d for t, d in zip(total_ms, d2h_ms)]
        results['gpu_effective_zerocopy_ms'] = compute_stats(effective_ms)

    return results


def compare_results(cpu_results: Dict, gpu_results: Dict) -> Dict:
    """Compare CPU and GPU results and compute speedup metrics."""
    comparison = {
        'cpu_summary': cpu_results,
        'gpu_summary': gpu_results,
        'comparison': {}
    }

    cpu_mean = cpu_results.get('cpu_total_ms', {}).get('mean')
    gpu_mean = gpu_results.get('gpu_total_ms', {}).get('mean')
    zerocopy_mean = gpu_results.get('gpu_effective_zerocopy_ms', {}).get('mean')

    if cpu_mean and gpu_mean:
        comparison['comparison']['speedup'] = cpu_mean / gpu_mean
        comparison['comparison']['cpu_mean_ms'] = cpu_mean
        comparison['comparison']['gpu_mean_ms'] = gpu_mean

    if cpu_mean and zerocopy_mean:
        comparison['comparison']['zerocopy_speedup'] = cpu_mean / zerocopy_mean
        comparison['comparison']['zerocopy_mean_ms'] = zerocopy_mean

    # Add breakdown percentages for GPU
    if 'gpu_h2d_ms' in gpu_results and 'gpu_kernel_ms' in gpu_results and gpu_mean:
        h2d_mean = gpu_results['gpu_h2d_ms'].get('mean', 0)
        kernel_mean = gpu_results['gpu_kernel_ms'].get('mean', 0)
        d2h_mean = gpu_results.get('gpu_d2h_ms', {}).get('mean', 0)

        comparison['comparison']['gpu_breakdown'] = {
            'h2d_pct': (h2d_mean / gpu_mean) * 100 if gpu_mean else 0,
            'd2h_pct': (d2h_mean / gpu_mean) * 100 if gpu_mean else 0,
            'kernel_pct': (kernel_mean / gpu_mean) * 100 if gpu_mean else 0,
        }

    return comparison


def print_summary(results: Dict, title: str = "Results"):
    """Print a formatted summary of results."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")

    if results.get('type') == 'cpu':
        if 'cpu_total_ms' in results:
            stats = results['cpu_total_ms']
            print(f"CPU Total Latency:")
            print(f"  Mean:  {stats['mean']:.3f} ms")
            print(f"  Std:   {stats['std']:.3f} ms")
            print(f"  P50:   {stats['p50']:.3f} ms")
            print(f"  P95:   {stats['p95']:.3f} ms")
            print(f"  P99:   {stats['p99']:.3f} ms")
            print(f"  Max:   {stats['max']:.3f} ms")
            print(f"  Count: {stats['count']}")

    elif results.get('type') == 'gpu':
        if 'gpu_total_ms' in results:
            stats = results['gpu_total_ms']
            print(f"GPU Total Latency:")
            print(f"  Mean:  {stats['mean']:.3f} ms")
            print(f"  Std:   {stats['std']:.3f} ms")
            print(f"  P50:   {stats['p50']:.3f} ms")
            print(f"  P95:   {stats['p95']:.3f} ms")
            print(f"  P99:   {stats['p99']:.3f} ms")
            print(f"  Max:   {stats['max']:.3f} ms")
            print(f"  Count: {stats['count']}")

        print(f"\nGPU Breakdown:")
        for key in ['gpu_h2d_ms', 'gpu_kernel_ms', 'gpu_d2h_ms']:
            if key in results:
                label = key.replace('gpu_', '').replace('_ms', '').upper()
                print(f"  {label}: {results[key]['mean']:.3f} ms")

        if 'gpu_effective_zerocopy_ms' in results:
            print(f"\nZero-Copy Effective (H2D + Kernel):")
            print(f"  Mean: {results['gpu_effective_zerocopy_ms']['mean']:.3f} ms")

    elif 'comparison' in results:
        comp = results['comparison']
        print(f"CPU Mean:           {comp.get('cpu_mean_ms', 'N/A'):.3f} ms")
        print(f"GPU Mean:           {comp.get('gpu_mean_ms', 'N/A'):.3f} ms")
        print(f"Zero-Copy Mean:     {comp.get('zerocopy_mean_ms', 'N/A'):.3f} ms")
        print(f"\nSpeedup (CPU/GPU):  {comp.get('speedup', 'N/A'):.2f}x")
        print(f"Zero-Copy Speedup:  {comp.get('zerocopy_speedup', 'N/A'):.2f}x")

        if 'gpu_breakdown' in comp:
            print(f"\nGPU Time Breakdown:")
            print(f"  H2D Copy:    {comp['gpu_breakdown']['h2d_pct']:.1f}%")
            print(f"  Kernel:      {comp['gpu_breakdown']['kernel_pct']:.1f}%")
            print(f"  D2H Copy:    {comp['gpu_breakdown']['d2h_pct']:.1f}%")

    print()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Nebula CPU vs GPU benchmark results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze CPU benchmark logs
  %(prog)s --log-dir ./benchmark_output_cpu --output cpu_results.json

  # Analyze GPU benchmark logs
  %(prog)s --log-dir ./benchmark_output_gpu --output gpu_results.json

  # Compare CPU and GPU results
  %(prog)s --cpu-results cpu_results.json --gpu-results gpu_results.json --compare
"""
    )

    parser.add_argument('--log-dir',
                        help='Directory containing benchmark log files')
    parser.add_argument('--output', '-o', default='benchmark_results.json',
                        help='Output JSON file for results')
    parser.add_argument('--cpu-results',
                        help='CPU results JSON file (for comparison)')
    parser.add_argument('--gpu-results',
                        help='GPU results JSON file (for comparison)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare CPU and GPU results')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress summary output')

    args = parser.parse_args()

    # Comparison mode
    if args.compare:
        if not args.cpu_results or not args.gpu_results:
            parser.error('--compare requires both --cpu-results and --gpu-results')

        with open(args.cpu_results) as f:
            cpu_results = json.load(f)
        with open(args.gpu_results) as f:
            gpu_results = json.load(f)

        results = compare_results(cpu_results, gpu_results)

        if not args.quiet:
            print_summary(results, "CPU vs GPU Comparison")

        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Comparison results saved to: {args.output}")
        return

    # Single analysis mode
    if not args.log_dir:
        parser.error('--log-dir is required for single analysis mode')

    records = parse_logs(args.log_dir)

    if not records:
        print("Error: No PROFILING records found in log files", file=sys.stderr)
        sys.exit(1)

    # Detect whether this is CPU or GPU data
    sample = records[0]
    if 'd_cpu_unpack_ms' in sample:
        results = analyze_cpu_logs(records)
        mode = 'CPU'
    elif 'd_gpu_total_ms' in sample:
        results = analyze_gpu_logs(records)
        mode = 'GPU'
    else:
        print("Error: Unrecognized PROFILING format", file=sys.stderr)
        print(f"Sample record: {sample}", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print_summary(results, f"{mode} Benchmark Results")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()
