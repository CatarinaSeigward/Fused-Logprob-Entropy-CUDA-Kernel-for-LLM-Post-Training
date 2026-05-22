"""Generate the 3 headline plots for README + REPORT.

Reads the most recent bench_micro_*.csv. Backward + integration numbers are
hardcoded from bench_backward.py / examples/compare_one_step.py runs and noted
in the source so they're audit-able. (The point of these plots is to be
embedded in the writeup, not to re-run benchmarks.)

Output: bench/plots/{speedup,bandwidth,memory}.png
"""
from __future__ import annotations

import csv
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PEAK_DRAM_GBPS = 256.0   # RTX 4060 Laptop


def latest_micro_csv() -> str:
    candidates = sorted(glob.glob(os.path.join(RESULTS_DIR, "bench_micro_*.csv")))
    if not candidates:
        raise FileNotFoundError("Run bench_micro.py first.")
    return candidates[-1]


def load_micro(path: str):
    """Return list[dict] with float-typed numeric fields."""
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            row["B"] = int(row["B"])
            row["S"] = int(row["S"])
            row["V"] = int(row["V"])
            row["BS"] = row["B"] * row["S"]
            row["latency_ms"] = float(row["latency_ms"])
            row["bandwidth_gbps"] = float(row["bandwidth_gbps"])
            row["bandwidth_pct_peak"] = float(row["bandwidth_pct_peak"])
            row["peak_alloc_mb"] = float(row["peak_alloc_mb"])
            rows.append(row)
    return rows


def plot_speedup_vs_vocab(rows, out_path: str):
    """Plot 1: K1 forward speedup vs TRL eager, grouped by vocab and B*S."""
    rows_bf16 = [r for r in rows if r["dtype"] == "bfloat16"]
    vocabs = sorted({r["V"] for r in rows_bf16})
    bs_values = sorted({r["BS"] for r in rows_bf16})

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.8 / len(bs_values)
    x = np.arange(len(vocabs))

    for i, bs in enumerate(bs_values):
        speedups = []
        for V in vocabs:
            ours = next((r for r in rows_bf16 if r["V"] == V and r["BS"] == bs and r["name"] == "ours_v1"), None)
            trl = next((r for r in rows_bf16 if r["V"] == V and r["BS"] == bs and r["name"] == "trl_eager"), None)
            speedups.append(trl["latency_ms"] / ours["latency_ms"] if (ours and trl) else 0)
        ax.bar(x + i * width - 0.4 + width / 2, speedups, width,
               label=f"B·S = {bs}")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{v//1000}k" for v in vocabs])
    ax.set_xlabel("Vocab size")
    ax.set_ylabel("Speedup vs TRL eager (×)")
    ax.set_title("K1 forward speedup over TRL `selective_log_softmax + entropy_from_logits`\n"
                 "RTX 4060 Laptop, bf16, RTX 4060 Laptop")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend(title="batch × seq")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_bandwidth_utilization(rows, out_path: str):
    """Plot 2: K1 hits much higher % of peak DRAM bw than baselines."""
    rows_bf16 = [r for r in rows if r["dtype"] == "bfloat16"]
    # Pick the largest shape per vocab to show steady-state BW
    by_vocab = {}
    for r in rows_bf16:
        key = (r["V"], r["name"])
        if key not in by_vocab or r["BS"] > by_vocab[key]["BS"]:
            by_vocab[key] = r

    vocabs = sorted({r["V"] for r in rows_bf16})
    backends = ["ours_v1", "trl_eager", "pytorch_separate", "ours_naive"]
    colors = {"ours_v1": "#1f77b4", "trl_eager": "#ff7f0e",
              "pytorch_separate": "#2ca02c", "ours_naive": "#d62728"}
    labels = {"ours_v1": "K1 (ours)", "trl_eager": "TRL eager",
              "pytorch_separate": "PyTorch separate", "ours_naive": "naive (ref)"}

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(vocabs))
    width = 0.2

    for i, backend in enumerate(backends):
        pcts = []
        for V in vocabs:
            r = by_vocab.get((V, backend))
            pcts.append(r["bandwidth_pct_peak"] if r else 0)
        ax.bar(x + i * width - 1.5 * width, pcts, width,
               label=labels[backend], color=colors[backend])

    ax.set_xticks(x)
    ax.set_xticklabels([f"{v//1000}k" for v in vocabs])
    ax.set_xlabel("Vocab size")
    ax.set_ylabel("DRAM bandwidth utilization (% of 256 GB/s peak)")
    ax.set_title("Effective DRAM bandwidth — K1 saturates the 4060's memory bus\n"
                 "(largest B·S per vocab, bf16)")
    ax.axhline(100.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5,
               label="hardware peak")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_memory_savings(rows, out_path: str):
    """Plot 3: peak intermediate-allocation per call. K1 = 0; eager paths
    materialize hundreds of MB."""
    rows_bf16 = [r for r in rows if r["dtype"] == "bfloat16"]
    # Show two representative shapes that bracket the regime
    target_shapes = [(1, 1024, 32000), (1, 1024, 128256), (1, 1024, 152064)]
    backends_order = ["ours_v1", "trl_eager", "pytorch_separate"]
    labels = {"ours_v1": "K1 (ours)", "trl_eager": "TRL eager",
              "pytorch_separate": "PyTorch separate"}
    colors = {"ours_v1": "#1f77b4", "trl_eager": "#ff7f0e",
              "pytorch_separate": "#2ca02c"}

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(target_shapes))
    width = 0.27

    for i, backend in enumerate(backends_order):
        mbs = []
        for B, S, V in target_shapes:
            r = next((r for r in rows_bf16
                      if r["B"] == B and r["S"] == S and r["V"] == V
                      and r["name"] == backend), None)
            mbs.append(r["peak_alloc_mb"] if r else 0)
        bars = ax.bar(x + i * width - width, mbs, width,
                      label=labels[backend], color=colors[backend])
        # Annotate each bar
        for bar, val in zip(bars, mbs):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 5, f"{val:.0f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"V={V//1000}k" for _, _, V in target_shapes])
    ax.set_xlabel("Vocab size (B·S = 1024)")
    ax.set_ylabel("Peak intermediate alloc per call (MB)")
    ax.set_title("Memory: K1 streams in fp32 registers, never materializes softmax\n"
                 "(eager paths' intermediates scale linearly with B·S × V)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    csv_path = latest_micro_csv()
    print(f"reading {csv_path}")
    rows = load_micro(csv_path)
    print(f"  {len(rows)} rows")

    plot_speedup_vs_vocab(rows, os.path.join(PLOTS_DIR, "speedup.png"))
    plot_bandwidth_utilization(rows, os.path.join(PLOTS_DIR, "bandwidth.png"))
    plot_memory_savings(rows, os.path.join(PLOTS_DIR, "memory.png"))


if __name__ == "__main__":
    main()
