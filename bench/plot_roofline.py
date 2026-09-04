"""Roofline + traffic-amplification plots from ncu-measured data.

Two figures:

  roofline.png              classic roofline; every kernel sits far left of the
                            ridge point, i.e. memory-bound, and the y-distance
                            to the roof is the bandwidth efficiency.
  traffic_amplification.png bytes actually moved vs the theoretical minimum.
                            This is the real reason the eager path is slow --
                            not slow kernels, but 8.4x redundant traffic.

Arithmetic intensity is derived ANALYTICALLY from the kernel source (counted
below), not measured: the ncu section set we captured exposes FLOP metrics as
peak-sustained rates rather than absolute instruction counts. Achieved
bandwidth IS ncu-measured. The caption on the figure says so.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ncu_parse import load_all, summarize, modeled_bytes, PEAK_DRAM_GBPS

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")

# ---------------------------------------------------------------- device spec
# RTX 4060 Laptop. Non-tensor FP32 FMA peak; K1 uses no Tensor Cores.
PEAK_FP32_GFLOPS = 15_000.0     # ~15 TFLOP/s fp32 FMA
RIDGE = PEAK_FP32_GFLOPS / PEAK_DRAM_GBPS   # FLOP/byte where the roof bends

# ------------------------------------------------- analytical FLOP per element
# Forward, per V element (csrc/fused_logprob.cu, combine() + loop body):
#   fmax 1, sub 2, __expf 2, Z: 2 mul + 1 add, T: 4 mul + 3 add  = 15 ops
# read 2 bytes (bf16)                          -> 15 / 2  = 7.5 FLOP/byte
#
# Backward, per V element (v1_backward_kernel loop body):
#   sub 1, __expf 1, fma 2, mul 1, convert 2   = 7 ops
# read 2 + write 2 = 4 bytes                   ->  7 / 4  = 1.75 FLOP/byte
AI_FORWARD = 7.5
AI_BACKWARD = 1.75
# NOTE: these are the *ideal* intensities, i.e. useful FLOPs per byte of
# strictly-necessary traffic. A backend that moves redundant bytes has a
# proportionally lower EFFECTIVE intensity, which we derive from the
# ncu-measured traffic amplification rather than guessing:
#     effective_AI = ideal_AI / (bytes_moved / bytes_required)

LABELS = {
    "ours_fwd": "K1 forward",
    "ours_bwd": "K1 backward",
    "naive": "naive (1 thread/row)",
    "trl": "TRL eager",
}
COLORS = {
    "ours_fwd": "#1f77b4",
    "ours_bwd": "#17becf",
    "naive": "#d62728",
    "trl": "#ff7f0e",
}
MARKERS = {"ours_fwd": "o", "ours_bwd": "s", "naive": "X", "trl": "^"}


def backend_of(tag: str) -> str:
    for b in ("ours_fwd", "ours_bwd", "naive", "trl"):
        if tag.startswith(b):
            return b
    return "?"


def ideal_ai_of(backend: str) -> float:
    if backend == "ours_bwd":
        return AI_BACKWARD
    return AI_FORWARD  # ours_fwd, naive, and TRL all compute the same quantity


def collect() -> list[dict]:
    out = []
    for tag, (units, rows) in load_all().items():
        s = summarize(tag, units, rows)
        if not s:
            continue
        b = backend_of(tag)
        s["backend"] = b
        s["shape"] = tag[len(b) + 1:]
        s["model_bytes"] = modeled_bytes(tag)
        amp = (s["dram_bytes"] / s["model_bytes"]) if s["model_bytes"] else 1.0
        s["amplification"] = amp
        # Effective intensity: same useful FLOPs, but spread over the bytes
        # actually moved. Redundant traffic pushes a kernel left on the chart.
        s["ai"] = ideal_ai_of(b) / max(amp, 1e-9)
        s["gflops"] = s["ai"] * s["dram_gbps"]   # FLOP/byte x byte/s
        out.append(s)
    return out


def plot_roofline(data: list[dict], path: str):
    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    ai = np.logspace(-1.2, 2.6, 400)
    roof = np.minimum(PEAK_FP32_GFLOPS, ai * PEAK_DRAM_GBPS)
    ax.loglog(ai, roof, "k-", lw=2.2, zorder=3,
              label=f"roofline: min({PEAK_FP32_GFLOPS/1000:.0f} TFLOP/s fp32, AI x 256 GB/s)")
    ax.axvline(RIDGE, color="gray", ls=":", lw=1.2, zorder=2,
               label=f"ridge = {RIDGE:.0f} FLOP/byte")

    # Label offsets tuned to keep the three overlapping ours_fwd points legible.
    offsets = {
        ("ours_fwd", "4096x32k"): (11, 7),
        ("ours_fwd", "1024x128k"): (11, -4),
        ("ours_fwd", "1024x152k"): (11, -15),
        ("ours_fwd", "256x32k"): (-64, -18),
        ("ours_bwd", "1024x128k"): (11, 4),
        ("ours_bwd", "4096x32k"): (11, -12),
    }

    seen = set()
    for d in data:
        b = d["backend"]
        lbl = LABELS[b] if b not in seen else None
        seen.add(b)
        ax.loglog(d["ai"], d["gflops"], MARKERS[b], color=COLORS[b],
                  ms=11, mec="black", mew=0.7, label=lbl, zorder=5)
        dx, dy = offsets.get((b, d["shape"]), (11, -4))
        ax.annotate(f"{d['shape']}  {d['dram_pct_peak']:.0f}% pk",
                    (d["ai"], d["gflops"]), textcoords="offset points",
                    xytext=(dx, dy), fontsize=7.5, color="#333", zorder=6)

    ax.set_xlabel("Effective arithmetic intensity (useful FLOP / byte actually moved)")
    ax.set_ylabel("Achieved performance (GFLOP/s)")
    ax.set_title(
        "Roofline — all kernels sit ~8x left of the ridge: strictly memory-bound\n"
        "Vertical gap to the roof = bandwidth left on the table; "
        "leftward shift = redundant traffic")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    fig.text(0.01, 0.012,
             "Useful-FLOP intensity counted from kernel source; DRAM traffic and achieved "
             "bandwidth measured by Nsight Compute.",
             fontsize=7, color="#555")
    fig.text(0.01, 0.001,
             "TRL sits left because of its 8.4x traffic amplification, not lower "
             "per-kernel efficiency. naive sits below the roof because of 8% occupancy.",
             fontsize=7, color="#555")
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")


def plot_traffic(data: list[dict], path: str):
    """The headline finding: TRL isn't slow because its kernels are slow."""
    canon = [d for d in data if d["shape"] == "1024x128k"]
    order = ["ours_fwd", "ours_bwd", "naive", "trl"]
    canon.sort(key=lambda d: order.index(d["backend"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- left: traffic amplification vs theoretical minimum
    names = [LABELS[d["backend"]] for d in canon]
    ratios = [d["dram_bytes"] / d["model_bytes"] for d in canon]
    colors = [COLORS[d["backend"]] for d in canon]
    bars = ax1.bar(names, ratios, color=colors)
    ax1.axhline(1.0, color="black", ls="--", lw=1, label="theoretical minimum")
    for bar, r, d in zip(bars, ratios, canon):
        ax1.text(bar.get_x() + bar.get_width() / 2, r + 0.15,
                 f"{r:.2f}x\n({d['dram_bytes']/1e6:.0f} MB)",
                 ha="center", va="bottom", fontsize=9)
    ax1.set_ylabel("DRAM bytes moved / theoretical minimum")
    ax1.set_title("Traffic amplification (B·S=1024, V=128k, bf16)")
    ax1.set_ylim(0, max(ratios) * 1.35)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)
    ax1.tick_params(axis="x", labelrotation=12)

    # --- right: bandwidth utilization + kernel launches
    pcts = [d["dram_pct_peak"] for d in canon]
    launches = [d["n_launches"] for d in canon]
    bars2 = ax2.bar(names, pcts, color=colors)
    for bar, p, n in zip(bars2, pcts, launches):
        ax2.text(bar.get_x() + bar.get_width() / 2, p + 1.5,
                 f"{p:.1f}%\n{n} launch{'es' if n > 1 else ''}",
                 ha="center", va="bottom", fontsize=9)
    ax2.axhline(100, color="black", ls=":", lw=1, label="hardware peak")
    ax2.set_ylabel("DRAM bandwidth utilization (% of 256 GB/s)")
    ax2.set_title("Per-kernel bandwidth efficiency\n"
                  "TRL's kernels are efficient — it just runs 43 of them")
    ax2.set_ylim(0, 118)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)
    ax2.tick_params(axis="x", labelrotation=12)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")


def print_table(data: list[dict]):
    print("\n=== ncu-measured summary (for REPORT tables) ===")
    print(f"{'backend':<12} {'shape':<11} {'dur(ms)':>8} {'GB/s':>7} {'%peak':>7} "
          f"{'occ%':>6} {'SM%':>6} {'traffic x':>10} {'launches':>9}")
    print("-" * 88)
    order = ["ours_fwd", "ours_bwd", "naive", "trl"]
    for d in sorted(data, key=lambda x: (order.index(x["backend"]), x["shape"])):
        amp = d["dram_bytes"] / d["model_bytes"] if d["model_bytes"] else float("nan")
        print(f"{d['backend']:<12} {d['shape']:<11} {d['duration_ms']:>8.4f} "
              f"{d['dram_gbps']:>7.1f} {d['dram_pct_peak']:>7.1f} "
              f"{d['occupancy_pct']:>6.1f} {d['sm_pct_peak']:>6.1f} "
              f"{amp:>10.2f} {d['n_launches']:>9}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    data = collect()
    if not data:
        print("No ncu data. Run bench/run_ncu.ps1 first.")
        return 1
    print_table(data)
    plot_roofline(data, os.path.join(PLOTS_DIR, "roofline.png"))
    plot_traffic(data, os.path.join(PLOTS_DIR, "traffic_amplification.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
