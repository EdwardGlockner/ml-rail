# Gentle-mode launcher for scripts/experiments/xtail_lora.py. Idle CPU priority, one decode worker,
# GPU memory cap, CPU kernel solves. Appends to data/generated/xtail/lora_experts.log.
# Optional: -WaitFor <PID> starts only after that process has exited.
param([int]$WaitFor = 0)
(Get-Process -Id $PID).PriorityClass = 'Idle'
if ($WaitFor) { while (Get-Process -Id $WaitFor -ErrorAction SilentlyContinue) { Start-Sleep 20 } }
$env:XTAIL_WORKERS = '1'
$env:XTAIL_GPU_FRACTION = '0.6'
$env:XTAIL_SOLVE = 'cpu'
Set-Location (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent)
$py = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
& $py scripts/experiments/xtail_lora.py --stage all *>> data/generated/xtail/lora_experts.log
