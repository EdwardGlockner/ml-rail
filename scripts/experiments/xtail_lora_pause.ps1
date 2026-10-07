# Pause the LoRA-expert job: it saves its state (training checkpoints mid-epoch) and exits
# within about a minute. Nothing is lost; run scripts/experiments/xtail_lora_resume.ps1 to continue.
$root = if ($env:XTAIL_ROOT) { $env:XTAIL_ROOT } else { Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'xtail_data' }
New-Item -ItemType Directory -Force (Join-Path $root 'lora') | Out-Null
if (-not (Test-Path (Join-Path $root 'lora\PAUSE'))) { New-Item -ItemType File (Join-Path $root 'lora\PAUSE') | Out-Null }
Write-Output 'Pause requested. Watch data\generated\xtail\lora_experts.log for "PAUSED".'
