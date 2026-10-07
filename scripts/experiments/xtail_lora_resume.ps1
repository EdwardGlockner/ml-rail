# Resume the LoRA-expert job where it stopped (low-load settings, hidden window).
$root = if ($env:XTAIL_ROOT) { $env:XTAIL_ROOT } else { Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'xtail_data' }
Remove-Item (Join-Path $root 'lora\PAUSE') -ErrorAction SilentlyContinue
$running = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -like '*xtail_lora.py*' }
if ($running) { Write-Output 'Already running.'; return }
Start-Process powershell -ArgumentList '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $PSScriptRoot 'xtail_lora_run.ps1') -WindowStyle Hidden
Write-Output 'Resumed. Progress: data\generated\xtail\lora_experts.log'
