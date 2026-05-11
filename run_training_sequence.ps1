$python = ".\.venv\Scripts\python.exe"
$models = @(
    @{ name = "EfficientNetB4"; config = "training/config/detector/efficientnetb4_dfd.yaml" },
    @{ name = "UCF";            config = "training/config/detector/ucf_dfd.yaml" },
    @{ name = "I3D";            config = "training/config/detector/i3d_dfd.yaml" }
)

if (-not (Test-Path $python)) {
    Write-Host "Python executable not found at $python"
    exit 1
}

Write-Host "Waiting for Xception to finish..."
while ($true) {
    $running = Get-Process python -ErrorAction SilentlyContinue | Where-Object {
        $_.Path -like "*DeepfakeBench-copy*" -and $_.Path -like "*.venv*"
    }

    if (-not $running -or $running.Count -le 1) {
        break
    }

    Write-Host "  Another Python training process is still running... $(Get-Date -Format 'HH:mm:ss')"
    Start-Sleep -Seconds 60
}

Write-Host "Starting sequence..."

foreach ($m in $models) {
    Write-Host ""
    Write-Host "============================="
    Write-Host "Starting $($m.name) - $(Get-Date -Format 'HH:mm:ss')"
    Write-Host "============================="

    & $python training/train.py --detector_path $m.config
    $code = $LASTEXITCODE

    if ($code -ne 0) {
        Write-Host "$($m.name) failed with exit code $code. Stopping sequence."
        exit $code
    }

    Write-Host "$($m.name) completed at $(Get-Date -Format 'HH:mm:ss')"
}

Write-Host ""
Write-Host "All models trained (EfficientNetB4 -> UCF -> I3D). Checkpoints are in logs/training/"
