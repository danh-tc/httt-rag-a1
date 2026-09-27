<#
Start the whole system on Windows: Docker DB -> ingest (if needed) -> dataset figures (if needed) -> UI build (if needed) -> API + UI.

Usage (from the repo root, after scripts\setup_venv_windows.ps1):
    powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
    powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 -Port 8080 -NoBrowser
    powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 -Reingest     # force re-encoding the corpus
    powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 -RebuildUi    # force rebuilding the React UI (needs Node.js)

Stop with Ctrl+C (the DB keeps running; stop it with: docker compose stop).
#>
param(
    [int]$Port = 8000,
    [switch]$NoBrowser,
    [switch]$Reingest,
    [switch]$RebuildUi
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$DbContainer = "httt-rag-a1-db"
$env:PYTHONIOENCODING = "utf-8"

# Windows PowerShell 5.1 turns a native command's stderr into a terminating error under "Stop",
# so probes that are expected to fail run under "Continue" and report only their exit code.
function Test-Native([string]$Exe, [string[]]$Arguments) {
    $saved = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $Exe @Arguments *> $null; return $LASTEXITCODE -eq 0 } finally { $ErrorActionPreference = $saved }
}

function Invoke-Checked([string]$Exe, [string[]]$Arguments) {
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "'$Exe $($Arguments -join ' ')' failed with exit code $LASTEXITCODE" }
}

Write-Host "==> [1/7] Checking venv"
if (-not (Test-Path $Python)) {
    throw ".venv not found. Run first: powershell -ExecutionPolicy Bypass -File scripts\setup_venv_windows.ps1"
}

Write-Host "==> [2/7] Checking Docker"
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw "docker not found. Install Docker Desktop." }
if (-not (Test-Native "docker" @("info"))) {
    $desktop = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    if (-not (Test-Path $desktop)) { throw "Docker daemon is not running and Docker Desktop was not found. Start Docker manually." }
    Write-Host "    Docker daemon not running; starting Docker Desktop (can take ~1 minute)..."
    Start-Process $desktop
    $deadline = (Get-Date).AddSeconds(180)
    while (-not (Test-Native "docker" @("info"))) {
        if ((Get-Date) -gt $deadline) { throw "Docker did not become ready within 180s. Open Docker Desktop and check for errors." }
        Start-Sleep -Seconds 3
    }
}

Write-Host "==> [3/7] Starting Postgres (ParadeDB)"
Invoke-Checked "docker" @("compose", "up", "-d")
$deadline = (Get-Date).AddSeconds(60)
while (-not (Test-Native "docker" @("exec", $DbContainer, "pg_isready", "-U", "rag", "-d", "rag"))) {
    if ((Get-Date) -gt $deadline) { throw "Postgres not ready within 60s. Check: docker logs $DbContainer" }
    Start-Sleep -Seconds 2
}

Write-Host "==> [4/7] Checking indexed documents"
$saved = $ErrorActionPreference; $ErrorActionPreference = "Continue"
$inDb = (& docker exec $DbContainer psql -U rag -d rag -tAc "SELECT count(*) FROM documents" 2>$null | Select-Object -First 1)
$ErrorActionPreference = $saved
$inCorpus = (& $Python -c "from rag.dataset import load_corpus; print(len(load_corpus()))")
Write-Host "    documents in DB: $inDb / corpus: $inCorpus"
if ($Reingest -or "$inDb".Trim() -ne "$inCorpus".Trim()) {
    Write-Host "    Ingesting (first run downloads the embedding model, ~1 GB)..."
    Invoke-Checked $Python @("-m", "rag.ingest")
}

Write-Host "==> [5/7] Checking dataset figures"
if (-not (Test-Path "results\figures\doc_length.png")) {
    Invoke-Checked $Python @("-m", "rag.dataset_stats")
} else {
    Write-Host "    results\figures already present (regenerate with: python -m rag.dataset_stats)"
}

Write-Host "==> [6/7] Checking UI build"
if ($RebuildUi -or -not (Test-Path "web\dist\index.html")) {
    if (-not (Get-Command npm -ErrorAction SilentlyContinue)) { throw "npm not found. Install Node.js >= 20 to build the UI (web\)." }
    Invoke-Checked "npm" @("--prefix", "web", "ci")
    Invoke-Checked "npm" @("--prefix", "web", "run", "build")
} else {
    Write-Host "    web\dist already built (rebuild with: -RebuildUi)"
}

Write-Host "==> [7/7] Starting API + UI on http://localhost:$Port"
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "Port $Port is already in use. Stop that process or pass -Port <other>."
}
if (-not $NoBrowser) {
    # Open the browser once the server answers (model loading takes a while on the first run).
    Start-Job -ArgumentList $Port -ScriptBlock {
        param($p)
        for ($i = 0; $i -lt 300; $i++) {
            try { Invoke-WebRequest "http://localhost:$p/" -UseBasicParsing -TimeoutSec 2 | Out-Null; Start-Process "http://localhost:$p/"; return } catch { Start-Sleep -Seconds 2 }
        }
    } | Out-Null
}
Write-Host "    First start loads the models (~3 GB download on a fresh machine). Ctrl+C to stop."
& $Python -m uvicorn rag.api:app --host 127.0.0.1 --port $Port
