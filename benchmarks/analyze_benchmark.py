#!/usr/bin/env python3
"""
Analyze and visualize PackBoost kernel benchmark results.

Usage:
    python analyze_benchmark.py benchmark_results.json
    python analyze_benchmark.py --compare results1.json results2.json
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping visualizations")


def load_results(filename: str) -> Dict:
    """Load benchmark results from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)


def print_summary(data: Dict, title: str = "Benchmark Results"):
    """Print text summary of results"""
    print("\n" + "="*80)
    print(title)
    print("="*80)

    config = data['config']
    print(f"Configuration: N={config['N']}, F={config['F']}, "
          f"K0={config['K0']}, K1={config['K1']}, max_depth={config['max_depth']}")
    print(f"Device: {config['device']}")
    print("-"*80)

    results = data['results']

    # Sort by mean time
    sorted_kernels = sorted(results.items(), key=lambda x: x[1]['mean_ms'])

    total_time = sum(r['mean_ms'] for r in results.values())

    print(f"{'Kernel':20s} {'Mean (ms)':>10s} {'Std (ms)':>10s} "
          f"{'Min (ms)':>10s} {'Max (ms)':>10s} {'% Total':>8s} {'BW (GB/s)':>12s}")
    print("-"*80)

    for name, result in sorted_kernels:
        pct = (result['mean_ms'] / total_time * 100) if total_time > 0 else 0
        bw = result.get('throughput_gb_s', None)
        bw_str = f"{bw:8.2f}" if bw else "N/A"

        print(f"{name:20s} {result['mean_ms']:10.3f} {result['std_ms']:10.3f} "
              f"{result['min_ms']:10.3f} {result['max_ms']:10.3f} {pct:7.1f}% {bw_str:>12s}")

    print("-"*80)
    print(f"{'TOTAL':20s} {total_time:10.3f} {'':10s} {'':10s} {'':10s} {'100.0%':>8s}")
    print("="*80)

    # Print hotspots
    print("\nTop 3 Hotspots:")
    for i, (name, result) in enumerate(sorted_kernels[-3:][::-1], 1):
        pct = (result['mean_ms'] / total_time * 100)
        print(f"  {i}. {name:20s} {result['mean_ms']:8.3f} ms ({pct:5.1f}%)")


def plot_timing_breakdown(data: Dict, output_file: str = "timing_breakdown.png"):
    """Create pie chart of kernel timings"""
    if not HAS_MATPLOTLIB:
        return

    results = data['results']
    names = list(results.keys())
    times = [results[name]['mean_ms'] for name in names]

    # Sort by time
    sorted_indices = np.argsort(times)[::-1]
    names = [names[i] for i in sorted_indices]
    times = [times[i] for i in sorted_indices]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Pie chart
    colors = plt.cm.Set3(range(len(names)))
    wedges, texts, autotexts = ax1.pie(times, labels=names, autopct='%1.1f%%',
                                        colors=colors, startangle=90)
    ax1.set_title(f"Kernel Execution Time Breakdown\nN={data['config']['N']}, "
                  f"F={data['config']['F']}")

    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(9)

    # Bar chart
    y_pos = np.arange(len(names))
    ax2.barh(y_pos, times, color=colors)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names)
    ax2.invert_yaxis()
    ax2.set_xlabel('Execution Time (ms)')
    ax2.set_title('Kernel Execution Time (ms)')
    ax2.grid(axis='x', alpha=0.3)

    # Add values on bars
    for i, v in enumerate(times):
        ax2.text(v + 0.5, i, f'{v:.2f}', va='center')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"\nSaved timing breakdown to {output_file}")


def plot_bandwidth_analysis(data: Dict, output_file: str = "bandwidth_analysis.png"):
    """Create bar chart of memory bandwidth utilization"""
    if not HAS_MATPLOTLIB:
        return

    results = data['results']

    # Filter kernels with bandwidth data
    kernels_with_bw = {name: res for name, res in results.items()
                       if res.get('throughput_gb_s') is not None}

    if not kernels_with_bw:
        print("No bandwidth data available")
        return

    names = list(kernels_with_bw.keys())
    bandwidths = [kernels_with_bw[name]['throughput_gb_s'] for name in names]

    # Sort by bandwidth
    sorted_indices = np.argsort(bandwidths)[::-1]
    names = [names[i] for i in sorted_indices]
    bandwidths = [bandwidths[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(12, 6))

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, bandwidths, color='skyblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel('Effective Bandwidth (GB/s)')
    ax.set_title(f'Memory Bandwidth Utilization\nN={data["config"]["N"]}, '
                 f'F={data["config"]["F"]}')
    ax.grid(axis='x', alpha=0.3)

    # Add reference line for peak bandwidth (A100 = ~2000 GB/s)
    peak_bw = 2000
    ax.axvline(peak_bw, color='red', linestyle='--', linewidth=2,
               label=f'Peak BW (A100): {peak_bw} GB/s')
    ax.legend()

    # Add values on bars
    for i, v in enumerate(bandwidths):
        pct = (v / peak_bw) * 100
        ax.text(v + 20, i, f'{v:.1f} GB/s ({pct:.1f}%)', va='center')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Saved bandwidth analysis to {output_file}")


def plot_comparison(data1: Dict, data2: Dict, output_file: str = "comparison.png"):
    """Compare two benchmark runs"""
    if not HAS_MATPLOTLIB:
        return

    results1 = data1['results']
    results2 = data2['results']

    # Find common kernels
    common_kernels = set(results1.keys()) & set(results2.keys())
    if not common_kernels:
        print("No common kernels to compare")
        return

    names = sorted(common_kernels)
    times1 = [results1[name]['mean_ms'] for name in names]
    times2 = [results2[name]['mean_ms'] for name in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Side-by-side bar chart
    x = np.arange(len(names))
    width = 0.35

    bars1 = ax1.bar(x - width/2, times1, width, label='Run 1', color='skyblue')
    bars2 = ax1.bar(x + width/2, times2, width, label='Run 2', color='lightcoral')

    ax1.set_xlabel('Kernel')
    ax1.set_ylabel('Execution Time (ms)')
    ax1.set_title('Execution Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Speedup chart
    speedups = [times1[i] / times2[i] for i in range(len(names))]
    colors = ['green' if s > 1 else 'red' for s in speedups]

    bars = ax2.barh(names, speedups, color=colors)
    ax2.axvline(1.0, color='black', linestyle='--', linewidth=2, label='No change')
    ax2.set_xlabel('Speedup (Run1 / Run2)')
    ax2.set_title('Speedup Analysis')
    ax2.legend()
    ax2.grid(axis='x', alpha=0.3)

    # Add speedup values
    for i, v in enumerate(speedups):
        label = f'{v:.2f}x'
        if v > 1:
            label += ' faster'
        else:
            label += ' slower'
        ax2.text(v + 0.02, i, label, va='center')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Saved comparison to {output_file}")


def print_comparison_table(data1: Dict, data2: Dict):
    """Print comparison table"""
    print("\n" + "="*80)
    print("BENCHMARK COMPARISON")
    print("="*80)

    results1 = data1['results']
    results2 = data2['results']

    common_kernels = sorted(set(results1.keys()) & set(results2.keys()))

    print(f"{'Kernel':20s} {'Run1 (ms)':>12s} {'Run2 (ms)':>12s} "
          f"{'Speedup':>10s} {'Change':>10s}")
    print("-"*80)

    for name in common_kernels:
        t1 = results1[name]['mean_ms']
        t2 = results2[name]['mean_ms']
        speedup = t1 / t2
        change = ((t2 - t1) / t1) * 100

        indicator = "↑" if speedup > 1.05 else "↓" if speedup < 0.95 else "≈"

        print(f"{name:20s} {t1:12.3f} {t2:12.3f} "
              f"{speedup:9.2f}x {change:+9.1f}% {indicator}")

    print("="*80)


def generate_report(data: Dict, output_dir: str = "."):
    """Generate full HTML report"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Generate plots
    if HAS_MATPLOTLIB:
        plot_timing_breakdown(data, str(output_path / "timing_breakdown.png"))
        plot_bandwidth_analysis(data, str(output_path / "bandwidth_analysis.png"))

    # Generate HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>PackBoost Kernel Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .config {{ background-color: #e7f3fe; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .highlight {{ background-color: #fff3cd; }}
        img {{ max-width: 100%; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>PackBoost CUDA Kernel Benchmark Report</h1>

    <div class="config">
        <h2>Configuration</h2>
        <ul>
            <li><b>Samples (N):</b> {data['config']['N']:,}</li>
            <li><b>Features (F):</b> {data['config']['F']:,}</li>
            <li><b>Tree Folds (K0):</b> {data['config']['K0']}</li>
            <li><b>Feature Sets (K1):</b> {data['config']['K1']}</li>
            <li><b>Max Depth:</b> {data['config']['max_depth']}</li>
            <li><b>Device:</b> {data['config']['device']}</li>
        </ul>
    </div>

    <h2>Execution Time Breakdown</h2>
    <img src="timing_breakdown.png" alt="Timing Breakdown">

    <h2>Memory Bandwidth Analysis</h2>
    <img src="bandwidth_analysis.png" alt="Bandwidth Analysis">

    <h2>Detailed Results</h2>
    <table>
        <tr>
            <th>Kernel</th>
            <th>Mean (ms)</th>
            <th>Std Dev (ms)</th>
            <th>Min (ms)</th>
            <th>Max (ms)</th>
            <th>Bandwidth (GB/s)</th>
            <th>% of Total</th>
        </tr>
"""

    results = data['results']
    total_time = sum(r['mean_ms'] for r in results.values())

    # Sort by time
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mean_ms'], reverse=True)

    for name, result in sorted_results:
        pct = (result['mean_ms'] / total_time * 100)
        bw = result.get('throughput_gb_s', None)
        bw_str = f"{bw:.2f}" if bw else "N/A"

        row_class = ' class="highlight"' if pct > 20 else ''

        html += f"""
        <tr{row_class}>
            <td>{name}</td>
            <td>{result['mean_ms']:.3f}</td>
            <td>{result['std_ms']:.3f}</td>
            <td>{result['min_ms']:.3f}</td>
            <td>{result['max_ms']:.3f}</td>
            <td>{bw_str}</td>
            <td>{pct:.1f}%</td>
        </tr>
"""

    html += f"""
        <tr style="font-weight: bold; background-color: #ddd;">
            <td>TOTAL</td>
            <td>{total_time:.3f}</td>
            <td colspan="4"></td>
            <td>100.0%</td>
        </tr>
    </table>

    <h2>Summary</h2>
    <ul>
        <li><b>Total Pipeline Time:</b> {total_time:.3f} ms</li>
        <li><b>Throughput:</b> {data['config']['N'] / (total_time / 1000):.0f} samples/sec</li>
    </ul>

</body>
</html>
"""

    report_path = output_path / "benchmark_report.html"
    with open(report_path, 'w') as f:
        f.write(html)

    print(f"\nGenerated HTML report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze kernel benchmark results")
    parser.add_argument("input", help="JSON file with benchmark results")
    parser.add_argument("--compare", help="Second JSON file to compare against")
    parser.add_argument("--report", action="store_true", help="Generate HTML report")
    parser.add_argument("--output-dir", default=".", help="Output directory for plots/report")

    args = parser.parse_args()

    # Load results
    if not Path(args.input).exists():
        print(f"Error: File not found: {args.input}")
        return 1

    data1 = load_results(args.input)
    print_summary(data1, "Benchmark Results")

    if args.compare:
        if not Path(args.compare).exists():
            print(f"Error: File not found: {args.compare}")
            return 1

        data2 = load_results(args.compare)
        print_summary(data2, "Comparison Results")
        print_comparison_table(data1, data2)

        if HAS_MATPLOTLIB:
            plot_comparison(data1, data2,
                          str(Path(args.output_dir) / "comparison.png"))
    else:
        if HAS_MATPLOTLIB:
            plot_timing_breakdown(data1,
                                str(Path(args.output_dir) / "timing_breakdown.png"))
            plot_bandwidth_analysis(data1,
                                  str(Path(args.output_dir) / "bandwidth_analysis.png"))

    if args.report:
        generate_report(data1, args.output_dir)

    return 0


if __name__ == "__main__":
    exit(main())
