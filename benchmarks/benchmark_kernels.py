#!/usr/bin/env python3
"""
PackBoost CUDA Kernel Benchmark Suite

Evaluates execution time of each CUDA kernel using a 10000x2000 feature matrix.
Provides detailed timing analysis and performance metrics.

Usage:
    python benchmark_kernels.py [--warmup N] [--iterations N] [--profile]
"""

import torch
import numpy as np
import argparse
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import csv
from pathlib import Path

# Import PackBoost modules (adjust path if needed)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from packboost import core
    from packboost.cuda import kernels
except ImportError as e:
    print(f"Warning: Could not import packboost modules: {e}")
    print("Some tests may be skipped.")


@dataclass
class KernelTiming:
    """Store timing results for a single kernel"""
    name: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    iterations: int
    warmup: int = 0
    throughput_gb_s: Optional[float] = None
    samples_per_sec: Optional[float] = None

    def __repr__(self):
        parts = [
            f"{self.name:20s}",
            f"Mean: {self.mean_ms:8.3f} ms",
            f"Std: {self.std_ms:6.3f} ms",
            f"Min: {self.min_ms:8.3f} ms",
            f"Max: {self.max_ms:8.3f} ms",
        ]
        if self.throughput_gb_s:
            parts.append(f"BW: {self.throughput_gb_s:6.2f} GB/s")
        if self.samples_per_sec:
            parts.append(f"Throughput: {self.samples_per_sec/1e6:6.2f} M samples/s")
        return " | ".join(parts)


class CUDATimer:
    """High-precision CUDA event-based timer"""

    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start_event.record()
        return self

    def __exit__(self, *args):
        self.end_event.record()
        torch.cuda.synchronize()
        self.elapsed_time = self.start_event.elapsed_time(self.end_event)

    def elapsed_ms(self) -> float:
        return self.elapsed_time


class KernelBenchmark:
    """Benchmark suite for PackBoost CUDA kernels"""

    def __init__(self, N: int = 10000, F: int = 2000, device: str = "cuda"):
        self.N = N  # Samples
        self.F = F  # Features
        self.device = device
        self.results: Dict[str, KernelTiming] = {}

        # Setup parameters
        self.K0 = 8  # Tree folds
        self.K1 = 64  # Feature sets (32 features each = 2048 total)
        self.max_depth = 5
        self.Dm = self.max_depth - 1
        self.nodes = (1 << self.max_depth) - 1
        self.M = (N + 31) // 32  # 32-bit words for bit packing



        self.pack = core.PackBoost(device=device)
        self.pack.nfeatsets = self.K1

        self._setup_data()

    def _setup_data(self):
        """Initialize all required tensors"""

        # Raw features (continuous)
        self.X_continuous = torch.randint(0, 128, (self.N, self.F),
                                         dtype=torch.int8, device=self.device)

        # Labels and predictions
        self.Y = torch.randint(-1000, 1000, (self.N,),
                               dtype=torch.int32, device=self.device)
        self.P = torch.randint(-1000, 1000, (self.N,),
                               dtype=torch.int32, device=self.device)

        # Gradients and leaf-encoding (will be computed by prep_vars)
        self.G = torch.zeros(self.N, dtype=torch.int16, device=self.device)
        self.LE = None

        # Branch bits (simulated tree paths)
        self.L_old = torch.randint(0, 2, (self.K0, self.Dm, self.N),
                                   dtype=torch.uint8, device=self.device)
        self.L_new = torch.zeros_like(self.L_old)

        # Feature set tree mapping
        self.FST = torch.randint(0, self.K0, (1, self.K1, self.max_depth),
                                 dtype=torch.uint8, device=self.device)

        # Feature schedule for sampling
        self.Fsch = torch.randint(0, self.F, (1, 32 * self.K1),
                                  dtype=torch.int32, device=self.device).to(torch.uint16)

        # Feature indices (for cut kernel)
        self.F_indices = torch.arange(32 * self.K1, dtype=torch.int32,
                                      device=self.device).to(torch.uint16).unsqueeze(0)

        # Encoded features (will be generated)
        self.XB = None
        self.XS = None
        self.LF = None

        # Histograms (will be generated)
        self.H0 = None
        self.H = None

        # Split results
        self.V = torch.zeros(1, self.K0, 2 * self.nodes,
                            dtype=torch.int32, device=self.device)
        self.I = torch.zeros(1, self.K0, self.nodes,
                            dtype=torch.uint16, device=self.device)

    def benchmark_kernel(self, name: str, func, warmup: int = 5,
                        iterations: int = 100, data_bytes: Optional[int] = None) -> KernelTiming:
        """Benchmark a single kernel with multiple iterations"""

        # Warmup
        for _ in range(warmup):
            func()
        torch.cuda.synchronize()

        # Timed runs
        timings = []
        for i in range(iterations):
            with CUDATimer() as timer:
                func()
            timings.append(timer.elapsed_ms())

        # Statistics
        mean_ms = np.mean(timings)
        std_ms = np.std(timings)
        min_ms = np.min(timings)
        max_ms = np.max(timings)

        # Calculate throughput if data size provided
        throughput_gb_s = None
        samples_per_sec = None
        if data_bytes:
            throughput_gb_s = (data_bytes / 1e9) / (mean_ms / 1000)
            samples_per_sec = self.N / (mean_ms / 1000)

        result = KernelTiming(
            name=name,
            mean_ms=mean_ms,
            std_ms=std_ms,
            min_ms=min_ms,
            max_ms=max_ms,
            iterations=iterations,
            warmup=warmup,
            throughput_gb_s=throughput_gb_s,
            samples_per_sec=samples_per_sec
        )

        self.results[name] = result

        return result

    def bench_encode_cuts(self, warmup: int = 5, iterations: int = 100):
        """Benchmark encode_cuts kernel"""
        try:
            from packboost.cuda import kernels
        except ImportError:
            return

        def run():
            self.XB = kernels.encode_cuts(self.X_continuous)

        # Data movement: Read X (N*F bytes), Write XB (4*F*M*4 bytes)
        data_bytes = self.N * self.F + 4 * self.F * self.M * 4

        self.benchmark_kernel("encode_cuts", run, warmup, iterations, data_bytes)

    def bench_et_sample(self, warmup: int = 5, iterations: int = 100):
        """Benchmark et_sample_1b torch path"""

        # Need XB first
        if self.XB is None:
            from packboost.cuda import kernels
            self.XB = kernels.encode_cuts(self.X_continuous)

        def run():
            self.XS = self.pack.et_sample_1b(self.XB, self.Fsch, 0)

        # Data: Read XB (4*F*M*4), Write XS (K1*32*M*4)
        data_bytes = 4 * self.F * self.M * 4 + self.K1 * 32 * self.M * 4

        self.benchmark_kernel("et_sample_1b", run, warmup, iterations, data_bytes)

    def bench_prep_vars(self, warmup: int = 5, iterations: int = 100):
        """Benchmark prep_vars kernel"""
        try:
            from packboost.cuda import kernels
        except ImportError:
            return

        def run():
            self.LE, self.G = kernels.prep_vars(self.L_old, self.Y, self.P)

        # Data: Read L, Y, P, Write LE, G
        data_bytes = self.K0 * self.Dm * self.N + self.N * 4 * 2 + self.N * 2

        self.benchmark_kernel("prep_vars", run, warmup, iterations, data_bytes)

    def bench_repack(self, warmup: int = 5, iterations: int = 100):
        """Benchmark repack_trees_for_features kernel"""
        try:
            from packboost.cuda import kernels
        except ImportError:
            return

        # LE must be computed by prep_vars first (produces uint16/32/64)
        if self.LE is None:
            self.LE, self.G = kernels.prep_vars(self.L_old, self.Y, self.P)

        self.LF = torch.empty(self.K1, self.LE.shape[1], dtype=self.LE.dtype,
                             device=self.device)

        def run():
            kernels.repack_trees_for_features(self.FST, self.LE, self.LF, 0)

        # Data: Read FST, LE, Write LF
        data_bytes = (self.K1 * self.max_depth +
                     self.LE.nelement() * self.LE.element_size() +
                     self.LF.nelement() * self.LF.element_size())

        self.benchmark_kernel("repack", run, warmup, iterations, data_bytes)

    def bench_h0(self, warmup: int = 5, iterations: int = 100):
        """Benchmark h0_sm_butterfly kernel"""
        try:
            from packboost.cuda import kernels
        except ImportError:
            return

        # Need G, LE first
        if self.G.sum() == 0:
            self.LE, self.G = kernels.prep_vars(self.L_old, self.Y, self.P)

        def run():
            self.H0 = kernels.h0_sm_butterfly(self.G, self.LE, self.max_depth)

        # Data: Read G, L_old, Write H0
        nodes = 1 << self.max_depth
        data_bytes = (self.N * 2 +
                     self.K0 * self.Dm * self.N +
                     self.K0 * nodes * 2 * 8)

        self.benchmark_kernel("h0_butterfly", run, warmup, iterations, data_bytes)

    def bench_h(self, warmup: int = 5, iterations: int = 100):
        """Benchmark h_sm kernel"""
        try:
            from packboost.cuda import kernels
        except ImportError:
            return

        # Need XS, G, LF
        if self.XS is None:
            self.XB = kernels.encode_cuts(self.X_continuous)
            self.XS = self.pack.et_sample_1b(self.XB, self.Fsch, 0)
            self.LE, self.G = kernels.prep_vars(self.L_old, self.Y, self.P)
            self.LF = torch.empty(self.K1, self.LE.shape[1], dtype=self.LE.dtype, device=self.device)
            kernels.repack_trees_for_features(self.FST, self.LE, self.LF, 0)

        def run():
            self.H = kernels.h_sm(self.XS, self.G, self.LF, self.max_depth)

        # Data: Read XS, G, LF, Write H
        data_bytes = (self.K1 * 32 * self.M * 4 +
                     self.N * 2 +
                     self.K1 * self.Dm * self.N +
                     self.K1 * self.nodes * 2 * 32 * 8)

        self.benchmark_kernel("h_sm", run, warmup, iterations, data_bytes)

    def bench_cut(self, warmup: int = 5, iterations: int = 100):
        """Benchmark cut_cuda kernel"""
        try:
            from packboost.cuda import kernels
        except ImportError:
            return

        # Need H, H0
        if self.H is None or self.H0 is None:
            self._prepare_histograms()

        def run():
            kernels.cut_cuda(
                self.F_indices, self.FST, self.H, self.H0,
                self.V, self.I, 0,
                1.0, 0.1, 15, self.max_depth,
                1.0, 0.0
            )

        # Data: Read H, H0, FST, F, Write V, I
        data_bytes = (self.K1 * self.nodes * 2 * 32 * 8 +
                     self.K0 * self.nodes * 2 * 8 +
                     self.K1 * self.max_depth +
                     32 * self.K1 * 2 +
                     self.K0 * 2 * self.nodes * 4 +
                     self.K0 * self.nodes * 2)

        self.benchmark_kernel("cut_cuda", run, warmup, iterations, data_bytes)

    def bench_adv(self, warmup: int = 5, iterations: int = 100):
        """Benchmark advance_and_predict kernel"""
        try:
            from packboost.cuda import kernels
        except ImportError:
            return

        # Need XB, V, I
        if self.XB is None:
            self.XB = kernels.encode_cuts(self.X_continuous)

        if self.V.sum() == 0:
            self._prepare_histograms()
            kernels.cut_cuda(
                self.F_indices, self.FST, self.H, self.H0,
                self.V, self.I, 0,
                1.0, 0.1, 15, self.max_depth,
                1.0, 0.0
            )

        P_test = torch.zeros(self.N, dtype=torch.int32, device=self.device)

        def run():
            kernels.advance_and_predict(
                P_test, self.XB, self.L_old, self.L_new,
                self.V, self.I.to(torch.uint16), 0
            )

        # Data: Read X, L_old, V, I, Write P, L_new
        data_bytes = (4 * self.F * self.M * 4 +
                     self.K0 * self.Dm * self.N +
                     self.K0 * 2 * self.nodes * 4 +
                     self.K0 * self.nodes * 2 +
                     self.N * 4 +
                     self.K0 * self.Dm * self.N)

        self.benchmark_kernel("advance_predict", run, warmup, iterations, data_bytes)

    def _prepare_histograms(self):
        """Helper to prepare H and H0 if needed"""
        from packboost.cuda import kernels

        if self.XB is None:
            self.XB = kernels.encode_cuts(self.X_continuous)
        if self.XS is None:
            self.XS = self.pack.et_sample_1b(self.XB, self.Fsch, 0)
        if self.G.sum() == 0:
            self.LE, self.G = kernels.prep_vars(self.L_old, self.Y, self.P)
        if self.LF is None:
            self.LF = torch.empty(self.K1, self.LE.shape[1], dtype=self.LE.dtype, device=self.device)
            kernels.repack_trees_for_features(self.FST, self.LE, self.LF, 0)

        self.H = kernels.h_sm(self.XS, self.G, self.LF, self.max_depth)
        self.H0 = kernels.h0_sm_butterfly(self.G, self.LE, self.max_depth)

    def run_all(self, warmup: int = 5, iterations: int = 100):
        """Run all benchmarks in pipeline order"""
        print("\n" + "="*80)
        print("RUNNING FULL BENCHMARK SUITE")
        print("="*80)

        self.bench_encode_cuts(warmup, iterations)
        self.bench_et_sample(warmup, iterations)
        self.bench_prep_vars(warmup, iterations)
        self.bench_repack(warmup, iterations)
        self.bench_h0(warmup, iterations)
        self.bench_h(warmup, iterations)
        self.bench_cut(warmup, iterations)
        self.bench_adv(warmup, iterations)

        print("\n" + "="*80)
        print("BENCHMARK COMPLETE")
        print("="*80)
        self.print_summary()

    def print_summary(self):
        """Print summary of all results"""
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        print(f"Configuration: N={self.N}, F={self.F}, K0={self.K0}, K1={self.K1}, max_depth={self.max_depth}")
        print(f"Device: {self.device}")
        print("-"*80)

        # Sort by mean time
        sorted_results = sorted(self.results.values(), key=lambda x: x.mean_ms)

        total_time = sum(r.mean_ms for r in sorted_results)

        for result in sorted_results:
            pct = (result.mean_ms / total_time * 100) if total_time > 0 else 0
            print(f"{result.name:20s} {result.mean_ms:8.3f} ms ({pct:5.1f}%)")

        print("-"*80)
        print(f"{'TOTAL':20s} {total_time:8.3f} ms (100.0%)")
        print("="*80)

    def save_csv(self, filename: str = "benchmark_results.csv"):
        """Save results to CSV file with parameters and timing data"""
        fieldnames = [
            "kernel", "N", "F", "K0", "K1", "max_depth", "device", "gpu",
            "warmup", "iterations",
            "mean_ms", "std_ms", "min_ms", "max_ms",
            "throughput_gb_s", "samples_per_sec",
        ]

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"

        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for name, r in self.results.items():
                writer.writerow({
                    "kernel": name,
                    "N": self.N,
                    "F": self.F,
                    "K0": self.K0,
                    "K1": self.K1,
                    "max_depth": self.max_depth,
                    "device": self.device,
                    "gpu": gpu_name,
                    "warmup": r.warmup,
                    "iterations": r.iterations,
                    "mean_ms": f"{r.mean_ms:.4f}",
                    "std_ms": f"{r.std_ms:.4f}",
                    "min_ms": f"{r.min_ms:.4f}",
                    "max_ms": f"{r.max_ms:.4f}",
                    "throughput_gb_s": f"{r.throughput_gb_s:.4f}" if r.throughput_gb_s else "",
                    "samples_per_sec": f"{r.samples_per_sec:.0f}" if r.samples_per_sec else "",
                })

        print(f"CSV results saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark PackBoost CUDA kernels")
    parser.add_argument("--N", type=int, default=10000, help="Number of samples")
    parser.add_argument("--F", type=int, default=2000, help="Number of features")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=100, help="Timed iterations")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--csv", type=str, default=str(Path(__file__).parent / "results" / "benchmark_results.csv"),
                       help="Output CSV file")
    parser.add_argument("--kernel", type=str, default=None,
                       help="Benchmark specific kernel only")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return 1

    # Print GPU info
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")

    bench = KernelBenchmark(N=args.N, F=args.F, device=args.device)

    if args.kernel:
        # Run specific kernel
        kernel_map = {
            "encode_cuts": bench.bench_encode_cuts,
            "et_sample": bench.bench_et_sample,
            "prep_vars": bench.bench_prep_vars,
            "repack": bench.bench_repack,
            "h0": bench.bench_h0,
            "h": bench.bench_h,
            "cut": bench.bench_cut,
            "adv": bench.bench_adv,
        }

        if args.kernel in kernel_map:
            kernel_map[args.kernel](args.warmup, args.iterations)
            bench.print_summary()
        else:
            print(f"Unknown kernel: {args.kernel}")
            print(f"Available: {', '.join(kernel_map.keys())}")
            return 1
    else:
        # Run all kernels
        bench.run_all(args.warmup, args.iterations)

    bench.save_csv(args.csv)

    return 0


if __name__ == "__main__":
    exit(main())
