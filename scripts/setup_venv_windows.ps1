<#
Create the project venv from scratch on Windows + RTX 5090 (Blackwell, sm_120).

Blackwell needs a CUDA >= 12.8 build of PyTorch and an NVIDIA driver >= 570.

Usage (from the repo root, in PowerShell):
    powershell -ExecutionPolicy Bypass -File scripts\setup_venv_windows.ps1
    powershell -ExecutionPolicy Bypass -File scripts\setup_venv_windows.ps1 -Recreate   # delete and rebuild .venv
#>
param(
    [string]$VenvDir = ".venv",
    [string]$TorchCuda = "cu128",  # cu128 / cu130 both support sm_120; cu126 does NOT
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"
$PythonVersion = "3.12"
$TorchVersion = "2.11.0"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Invoke-Checked([string]$Exe, [string[]]$Arguments) {
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "'$Exe $($Arguments -join ' ')' failed with exit code $LASTEXITCODE" }
}

Write-Host "==> [1/5] Checking NVIDIA driver"
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    throw "nvidia-smi not found. Install the NVIDIA driver (>= 570) first."
}
$gpu = (& nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader | Select-Object -First 1).Split(",").Trim()
Write-Host "    GPU: $($gpu[0]) | driver $($gpu[1]) | compute capability $($gpu[2])"
if ([int]($gpu[1].Split(".")[0]) -lt 570) {
    throw "Driver $($gpu[1]) is too old for CUDA 12.8 / Blackwell. Update the NVIDIA driver to >= 570."
}

Write-Host "==> [2/5] Checking Python $PythonVersion"
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python launcher 'py' not found. Install Python ${PythonVersion}: winget install Python.Python.$PythonVersion"
}
& py "-$PythonVersion" --version
if ($LASTEXITCODE -ne 0) {
    throw "Python $PythonVersion not installed. Run: winget install Python.Python.$PythonVersion"
}

Write-Host "==> [3/5] Creating venv at $VenvDir"
if (Test-Path $VenvDir) {
    if (-not $Recreate) { throw "$VenvDir already exists. Re-run with -Recreate to delete and rebuild it." }
    Remove-Item -Recurse -Force $VenvDir
}
Invoke-Checked "py" @("-$PythonVersion", "-m", "venv", $VenvDir)
$Python = Join-Path $VenvDir "Scripts\python.exe"
Invoke-Checked $Python @("-m", "pip", "install", "--upgrade", "pip")

Write-Host "==> [4/5] Installing torch $TorchVersion+$TorchCuda, then requirements.txt"
Invoke-Checked $Python @("-m", "pip", "install", "torch==$TorchVersion", "--index-url", "https://download.pytorch.org/whl/$TorchCuda")
Invoke-Checked $Python @("-m", "pip", "install", "-r", "requirements.txt")

Write-Host "==> [5/5] Verifying GPU"
Invoke-Checked $Python @("scripts\check_gpu.py")

Write-Host ""
Write-Host "Done. Activate with:  $VenvDir\Scripts\Activate.ps1"
Write-Host "Next: docker compose up -d ; python -m rag.ingest ; python -m uvicorn rag.api:app --port 8000"
