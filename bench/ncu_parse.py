"""Parse ncu --csv --page raw exports into a tidy table.

Schema notes (learned from ncu 2024.3.2 on this machine):
  - PowerShell '>' redirect writes UTF-16LE with BOM; ncu itself writes UTF-8.
  - Wide format: one column per metric, one row per kernel launch.
  - **The first data row is a UNITS row** ("ms", "%", "byte/cycle"), not data.
  - There is no plain `dram__bytes.sum` total; only `dram__bytes.sum.per_second`
    (Gbyte/s). Total bytes are derived as rate x duration.
  - There is no `dram__throughput.*.pct_of_peak_sustained_elapsed` in this
    section set; we compute % of peak against the device's spec bandwidth.

Standalone usage:
    python bench/ncu_parse.py                     # summary table
    python bench/ncu_parse.py --grep dram         # find metric columns
    python bench/ncu_parse.py --validate          # ncu vs bench_micro estimate
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import os
import re

NCU_DIR = os.path.join(os.path.dirname(__file__), "ncu")

# RTX 4060 Laptop spec sheet: 128-bit bus @ 8 Gbps effective GDDR6.
PEAK_DRAM_GBPS = 256.0

# Metric columns we consume (exact names for ncu 2024.3.2).
M_DURATION_MS = "gpu__time_duration.sum"
M_DRAM_GBPS = "dram__bytes.sum.per_second"
M_SM_PCT = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
M_OCCUPANCY = "sm__warps_active.avg.pct_of_peak_sustained_active"


# ncu picks the unit that best fits each report's magnitude, so the SAME
# metric can come back as "us" in one report and "ms" in another. Dropping
# the units row and assuming a fixed unit silently scales short-kernel
# durations by 1000x. Normalize everything against these tables instead.
_TIME_TO_MS = {"ns": 1e-6, "nsecond": 1e-6, "us": 1e-3, "usecond": 1e-3,
               "ms": 1.0, "msecond": 1.0, "s": 1e3, "second": 1e3}
_RATE_TO_GBPS = {"byte/s": 1e-9, "Kbyte/s": 1e-6, "Mbyte/s": 1e-3,
                 "Gbyte/s": 1.0, "Tbyte/s": 1e3}


def read_ncu_csv(path: str) -> tuple[dict, list[dict]]:
    """Read an ncu CSV export.

    Returns (units, rows). Handles UTF-16LE BOM from PowerShell redirect and
    separates the units row (row 0) from the data rows.
    """
    with open(path, "rb") as f:
        raw = f.read()

    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = raw.decode("utf-16")
    else:
        text = raw.decode("utf-8-sig")

    lines = text.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if line.startswith('"ID"') or line.startswith("ID,"):
            start = i
            break

    rows = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))

    units: dict = {}
    if rows and to_float(rows[0].get(M_DURATION_MS)) is None:
        units = rows[0]
        rows = rows[1:]
    return units, rows


def duration_ms(row: dict, units: dict) -> float | None:
    """Kernel duration normalized to milliseconds."""
    v = to_float(row.get(M_DURATION_MS))
    if v is None:
        return None
    unit = (units.get(M_DURATION_MS) or "ms").strip()
    return v * _TIME_TO_MS.get(unit, 1.0)


def dram_gbps(row: dict, units: dict) -> float | None:
    """Achieved DRAM throughput normalized to Gbyte/s."""
    v = to_float(row.get(M_DRAM_GBPS))
    if v is None:
        return None
    unit = (units.get(M_DRAM_GBPS) or "Gbyte/s").strip()
    return v * _RATE_TO_GBPS.get(unit, 1.0)


def to_float(v):
    if v is None:
        return None
    s = str(v).strip().replace(",", "")
    if s in ("", "N/A", "n/a", "-"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_all(pattern: str = "*_raw.csv") -> dict[str, tuple[dict, list[dict]]]:
    out = {}
    for path in sorted(glob.glob(os.path.join(NCU_DIR, pattern))):
        tag = os.path.basename(path).replace("_raw.csv", "")
        try:
            out[tag] = read_ncu_csv(path)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] failed to parse {path}: {e}")
    return out


def summarize(tag: str, units: dict, rows: list[dict]) -> dict | None:
    """Aggregate one report. Multiple rows = multiple kernel launches."""
    if not rows:
        return None

    total_ms = 0.0
    total_bytes = 0.0
    sm_num = occ_num = den = 0.0
    kernels = []

    for r in rows:
        dur = duration_ms(r, units)
        if dur is None:
            continue
        total_ms += dur

        gbps = dram_gbps(r, units)
        if gbps is not None:
            # Gbyte/s * ms  ->  bytes:  gbps*1e9 * dur*1e-3
            total_bytes += gbps * dur * 1e6

        sm = to_float(r.get(M_SM_PCT))
        occ = to_float(r.get(M_OCCUPANCY))
        if sm is not None:
            sm_num += dur * sm
        if occ is not None:
            occ_num += dur * occ
        den += dur

        name = (r.get("Kernel Name") or "?").strip()
        kernels.append(name)

    if den == 0:
        return None

    achieved_gbps = (total_bytes / 1e9) / (total_ms / 1e3) if total_ms else 0.0

    return {
        "tag": tag,
        "n_launches": len(kernels),
        "kernels": kernels,
        "duration_ms": total_ms,
        "dram_bytes": total_bytes,
        "dram_gbps": achieved_gbps,
        "dram_pct_peak": 100.0 * achieved_gbps / PEAK_DRAM_GBPS,
        "sm_pct_peak": sm_num / den,
        "occupancy_pct": occ_num / den,
    }


# ---------------------------------------------------------------- validation

# What bench_micro.py assumes it reads, per shape. logits are bf16 (2 bytes).
# Backward additionally writes d_logits (same size), so 2x traffic.
SHAPE_BS_V = {
    "256x32k": (256, 32000),
    "1024x32k": (1024, 32000),
    "4096x32k": (4096, 32000),
    "256x128k": (256, 128256),
    "1024x128k": (1024, 128256),
    "1024x152k": (1024, 152064),
}


def modeled_bytes(tag: str) -> float | None:
    """The byte count bench_micro.py / bench_backward.py assume."""
    for shape, (bs, v) in SHAPE_BS_V.items():
        if tag.endswith(shape):
            base = bs * v * 2  # bf16 logits, read once
            if "bwd" in tag:
                return base * 2  # read logits + write d_logits
            return base
    return None


def validate(reports: dict[str, tuple[dict, list[dict]]]):
    """Phase 3: does ncu's measured DRAM traffic match our modeled traffic?"""
    print("\n=== ncu measured vs bench-harness model ===")
    print(f"{'report':<22} {'model MB':>9} {'ncu MB':>9} {'ratio':>7} "
          f"{'ncu GB/s':>9} {'ncu %pk':>8}")
    print("-" * 72)
    for tag, (units, rows) in reports.items():
        s = summarize(tag, units, rows)
        if not s:
            continue
        model = modeled_bytes(tag)
        if model is None:
            continue
        ratio = s["dram_bytes"] / model if model else 0
        print(f"{tag:<22} {model/1e6:>9.1f} {s['dram_bytes']/1e6:>9.1f} "
              f"{ratio:>7.2f} {s['dram_gbps']:>9.1f} {s['dram_pct_peak']:>8.1f}")
    print("\nratio ~1.0  -> model is right, effective-bandwidth numbers validated")
    print("ratio <1.0  -> L2 absorbed traffic; DRAM read less than modeled")
    print("ratio >1.0  -> extra traffic (redundant passes / cache-line waste)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grep", type=str, default=None,
                    help="print metric columns matching this regex")
    ap.add_argument("--validate", action="store_true",
                    help="compare ncu-measured DRAM bytes vs the harness model")
    ap.add_argument("--kernels", action="store_true",
                    help="list kernel names per report")
    args = ap.parse_args()

    reports = load_all()
    if not reports:
        print(f"No CSVs found in {NCU_DIR}. Run bench/run_ncu.ps1 first.")
        return 1
    print(f"loaded {len(reports)} reports from {NCU_DIR}")

    if args.grep:
        first = next(iter(reports))
        units, _rows = reports[first]
        pat = re.compile(args.grep, re.IGNORECASE)
        cols = [c for c in units.keys() if pat.search(c)]
        print(f"\n{len(cols)} matching columns in {first}:")
        for c in cols:
            print(f"  {c}  =  {units.get(c, '')}")
        return 0

    print()
    print(f"{'report':<22} {'launches':>8} {'dur(ms)':>9} {'DRAM MB':>9} "
          f"{'GB/s':>8} {'%peak':>7} {'SM%':>6} {'occ%':>6}")
    print("-" * 82)
    for tag, (units, rows) in reports.items():
        s = summarize(tag, units, rows)
        if not s:
            continue
        print(f"{tag:<22} {s['n_launches']:>8} {s['duration_ms']:>9.4f} "
              f"{s['dram_bytes']/1e6:>9.1f} {s['dram_gbps']:>8.1f} "
              f"{s['dram_pct_peak']:>7.1f} {s['sm_pct_peak']:>6.1f} "
              f"{s['occupancy_pct']:>6.1f}")

    if args.kernels:
        print("\n=== kernel names ===")
        for tag, (units, rows) in reports.items():
            s = summarize(tag, units, rows)
            if not s:
                continue
            uniq = {}
            for k in s["kernels"]:
                uniq[k] = uniq.get(k, 0) + 1
            print(f"\n{tag}  ({s['n_launches']} launches):")
            for k, n in sorted(uniq.items(), key=lambda kv: -kv[1])[:12]:
                print(f"   {n:>4}x  {k[:90]}")

    if args.validate:
        validate(reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
