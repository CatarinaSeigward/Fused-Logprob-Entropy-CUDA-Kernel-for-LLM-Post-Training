<#
.SYNOPSIS
    Batch-run Nsight Compute over the K1 kernels and baselines.

.DESCRIPTION
    Profiles a focused matrix (not the full cross product) that answers the
    three questions REPORT section 4 needs:

      1. What is K1's REAL DRAM bandwidth utilization?  (vs our bytes/latency
         estimate from bench_micro.py)
      2. How does it scale with vocab and batch?
      3. Why are the baselines slow?  (kernel count + memory traffic)

    Writes .ncu-rep reports and raw-metric CSVs to bench/ncu/.

.NOTES
    ncu needs GPU performance-counter access. On Windows consumer GPUs this
    means EITHER running from an elevated (Administrator) PowerShell, OR a
    one-time registry change:

        New-Item -Path "HKLM:\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak" -Force
        New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak" `
            -Name "RmProfilingAdminOnly" -PropertyType DWord -Value 0 -Force
        # then REBOOT

    If you see ERR_NVGPUCTRPERM, that's what it is.

.PARAMETER Quick
    Use a targeted section list instead of --set full. Much faster
    (fewer replay passes) but omits some detail.

.PARAMETER All
    Run the full backend x shape cross product instead of the focused matrix.

.EXAMPLE
    # From an ELEVATED PowerShell, at the repo root:
    .\bench\run_ncu.ps1

.EXAMPLE
    .\bench\run_ncu.ps1 -Quick
#>
param(
    [switch]$Quick,
    [switch]$All
)

$ErrorActionPreference = "Continue"

# ---------------------------------------------------------------- locate ncu
$ncu = $null
$candidates = @(
    "C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.3.2\ncu.bat"
)
# Also glob for any other installed version.
$globbed = Get-ChildItem "C:\Program Files\NVIDIA Corporation\Nsight Compute*\ncu.bat" -ErrorAction SilentlyContinue
foreach ($g in $globbed) { $candidates += $g.FullName }

foreach ($c in $candidates) {
    if (Test-Path $c) { $ncu = $c; break }
}
if ($null -eq $ncu) {
    Write-Host "ERROR: could not find ncu.bat. Install Nsight Compute or edit this script." -ForegroundColor Red
    exit 1
}
Write-Host "[run_ncu] using: $ncu" -ForegroundColor Cyan

# ------------------------------------------------------------------- layout
$repoRoot = Split-Path -Parent $PSScriptRoot
$python   = Join-Path $repoRoot ".venv\Scripts\python.exe"
$targets  = Join-Path $repoRoot "bench\ncu_targets.py"
$outDir   = Join-Path $repoRoot "bench\ncu"

if (-not (Test-Path $python))  { Write-Host "ERROR: venv python not found at $python" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $targets)) { Write-Host "ERROR: ncu_targets.py not found at $targets" -ForegroundColor Red; exit 1 }
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$env:PYTHONUTF8 = "1"

# ------------------------------------------------------------- kernel filters
# TRL's eager path launches many aten kernels; deliberately unfiltered so the
# report shows the launch-count blowup.
$filters = @{
    "ours_fwd" = "fused_logprob_entropy_v1_kernel"
    "ours_bwd" = "fused_logprob_entropy_v1_backward_kernel"
    "naive"    = "fused_logprob_entropy_naive_kernel"
    "trl"      = $null
}

# ---------------------------------------------------------------- run matrix
if ($All) {
    $backends = @("ours_fwd", "ours_bwd", "naive", "trl")
    $shapes   = @("256x32k", "1024x32k", "4096x32k", "256x128k", "1024x128k", "1024x152k")
    $matrix = @()
    foreach ($b in $backends) { foreach ($s in $shapes) { $matrix += ,@($b, $s) } }
} else {
    # Focused matrix: all backends at the canonical shape, plus a K1 scaling
    # sweep, plus backward at the shape where it peaked in bench_backward.py.
    $matrix = @(
        @("ours_fwd", "1024x128k"),   # canonical: all four backends here
        @("naive",    "1024x128k"),
        @("trl",      "1024x128k"),
        @("ours_bwd", "1024x128k"),

        @("ours_fwd", "256x32k"),     # scaling sweep for K1 forward
        @("ours_fwd", "4096x32k"),
        @("ours_fwd", "1024x152k"),

        @("ours_bwd", "4096x32k")     # backward peaked at 87.8% here
    )
}

# ------------------------------------------------------------- section flags
if ($Quick) {
    $sectionArgs = @(
        "--section", "SpeedOfLight",
        "--section", "SpeedOfLight_RooflineChart",
        "--section", "MemoryWorkloadAnalysis",
        "--section", "Occupancy",
        "--section", "LaunchStats"
    )
    Write-Host "[run_ncu] mode: quick (targeted sections)" -ForegroundColor Yellow
} else {
    $sectionArgs = @("--set", "full")
    Write-Host "[run_ncu] mode: full (--set full; slower, more replay passes)" -ForegroundColor Yellow
}

# ------------------------------------------------------------------- run all
$results = @()
$i = 0
foreach ($entry in $matrix) {
    $i++
    $backend = $entry[0]
    $shape   = $entry[1]
    $tag     = "${backend}_${shape}"
    $repPath = Join-Path $outDir $tag

    Write-Host ""
    Write-Host "=== [$i/$($matrix.Count)] $backend @ $shape ===" -ForegroundColor Green

    $ncuArgs = @("--profile-from-start", "off")
    $ncuArgs += $sectionArgs
    if ($null -ne $filters[$backend]) {
        $ncuArgs += @("-k", "regex:$($filters[$backend])")
    }
    $ncuArgs += @("-o", $repPath, "--force-overwrite")
    $ncuArgs += @($python, $targets, "--backend", $backend, "--shape", $shape)

    & $ncu @ncuArgs
    $code = $LASTEXITCODE

    if ($code -ne 0) {
        Write-Host "  FAILED (exit $code)" -ForegroundColor Red
        $results += [PSCustomObject]@{ Backend=$backend; Shape=$shape; Status="FAIL($code)" }
        continue
    }

    # Export raw metrics to CSV so plot_roofline.py can read them without the GUI.
    $csvPath = "$repPath`_raw.csv"
    & $ncu --import "$repPath.ncu-rep" --csv --page raw > $csvPath
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  OK -> $tag.ncu-rep + $tag`_raw.csv" -ForegroundColor Green
        $results += [PSCustomObject]@{ Backend=$backend; Shape=$shape; Status="OK" }
    } else {
        Write-Host "  report OK but CSV export failed" -ForegroundColor Yellow
        $results += [PSCustomObject]@{ Backend=$backend; Shape=$shape; Status="OK(no csv)" }
    }
}

# -------------------------------------------------------------------- report
Write-Host ""
Write-Host "==================== SUMMARY ====================" -ForegroundColor Cyan
$results | Format-Table -AutoSize

$failed = @($results | Where-Object { $_.Status -like "FAIL*" })
if ($failed.Count -gt 0) {
    Write-Host ""
    Write-Host "Some runs failed. If the error was ERR_NVGPUCTRPERM:" -ForegroundColor Yellow
    Write-Host "  - re-run this script from an ELEVATED (Administrator) PowerShell, or" -ForegroundColor Yellow
    Write-Host "  - apply the RmProfilingAdminOnly registry fix in this script's header and reboot." -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "Reports in: $outDir" -ForegroundColor Cyan
Write-Host "Next: open one in the GUI for screenshots ->" -ForegroundColor Cyan
Write-Host "  ncu-ui `"$outDir\ours_fwd_1024x128k.ncu-rep`"" -ForegroundColor Cyan
Write-Host "Then: python bench\plot_roofline.py" -ForegroundColor Cyan
