# Start DeepGuard — API + Frontend
# Usage: .\start.ps1

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
# Escape any single quotes in the path (e.g. repo's)
$escapedRoot = $root -replace "'", "''"
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
  Write-Host "Python executable not found at $python" -ForegroundColor Red
  exit 1
}

Write-Host "`n[DeepGuard] Starting servers...`n" -ForegroundColor Cyan

# Start FastAPI inference server in background
Start-Process powershell -ArgumentList "-NoExit", "-Command",
  "cd '$escapedRoot'; & '$($python -replace "'", "''")' -m uvicorn inference_api:app --host 0.0.0.0 --port 8000 --reload"

Start-Sleep 2

# Start React frontend
Start-Process powershell -ArgumentList "-NoExit", "-Command",
  "cd '$escapedRoot\frontend'; npm run dev"

Write-Host "[DeepGuard] API   → http://localhost:8000" -ForegroundColor Green
Write-Host "[DeepGuard] UI    → http://localhost:3000" -ForegroundColor Green
Write-Host "`nOpen http://localhost:3000 in your browser`n" -ForegroundColor Yellow
