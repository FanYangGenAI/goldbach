# Create project .venv and install CUDA-enabled PyTorch (cu121).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Pip = Join-Path $Root ".venv\Scripts\pip.exe"

& $Python -m pip install --upgrade pip
& $Pip install -r requirements-cuda.txt --index-url https://download.pytorch.org/whl/cu121

Write-Host ""
Write-Host "Verifying CUDA..."
& $Python -c "import torch; print('torch', torch.__version__); print('cuda built', torch.version.cuda); print('cuda available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"

Write-Host ""
Write-Host "Activate:   .\.venv\Scripts\Activate.ps1"
Write-Host "Train GPU:  .\.venv\Scripts\python.exe train\train.py --device cuda"
