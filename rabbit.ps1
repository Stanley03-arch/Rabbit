# Rabbit PowerShell launcher
# Usage:  .\rabbit.ps1
#         .\rabbit.ps1 dashboard
#         .\rabbit.ps1 chat

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$RabbitBin = if (Test-Path ".\rabbit.exe") { ".\rabbit.exe" } elseif (Test-Path ".\rabbit") { ".\rabbit" } else { ".\rabbit.exe" }
$Model = if ($env:RABBIT_MODEL) { $env:RABBIT_MODEL } else { "models\tiny-test.rabbit" }
$Port = if ($env:RABBIT_PORT) { $env:RABBIT_PORT } else { "7860" }

function Ensure-Python {
  if (Get-Command python -ErrorAction SilentlyContinue) { return "python" }
  if (Get-Command py -ErrorAction SilentlyContinue) { return "py -3" }
  throw "Python not found"
}

function Ensure-Rabbit {
  if (-not (Test-Path $RabbitBin)) {
    Write-Host "Building rabbit.exe..."
    if (-not (Get-Command gcc -ErrorAction SilentlyContinue)) {
      throw "gcc not found. Install MinGW/MSYS2/WinLibs."
    }
    $srcs = @(
      "src/main.c","src/memory.c","src/model.c","src/moe.c","src/quant.c",
      "src/tokenizer.c","src/utils.c","src/attention.c","src/forward_v2.c"
    )
    & gcc -O2 -Wall -Wextra -std=c11 -I./src @srcs -o rabbit.exe -lm
    if ($LASTEXITCODE -ne 0) { throw "gcc build failed" }
    $script:RabbitBin = ".\rabbit.exe"
  }
}

function Ensure-Model {
  if (-not (Test-Path $Model)) {
    Write-Host "Creating demo model at $Model"
    New-Item -ItemType Directory -Force -Path models | Out-Null
    $py = Ensure-Python
    Invoke-Expression "$py tools/convert_hf_to_rabbit.py --demo --out $Model --hidden-dim 128 --n-experts 8 --quant 4"
  }
}

$cmd = if ($args.Count -gt 0) { $args[0].ToLower() } else { "menu" }

switch ($cmd) {
  "build" {
    Ensure-Rabbit
    Write-Host "OK: $RabbitBin"
  }
  "dashboard" {
    Ensure-Rabbit
    Ensure-Model
    $py = Ensure-Python
    try { & $py -c "import flask" 2>$null } catch { & $py -m pip install flask }
    Write-Host "Dashboard http://127.0.0.1:${Port}/"
    Invoke-Expression "$py tools/rabbit_dashboard.py --model $Model --host 127.0.0.1 --port $Port"
  }
  "chat" {
    Ensure-Rabbit
    Ensure-Model
    & $RabbitBin chat --model $Model --tokens 48 --temp 0.8
  }
  "stats" {
    Ensure-Rabbit
    Ensure-Model
    & $RabbitBin stats --model $Model
  }
  "gradio" {
    Ensure-Rabbit
    Ensure-Model
    $py = Ensure-Python
    & $py -m pip install gradio -q
    Invoke-Expression "$py tools/rabbit_ui.py --model $Model --repo ."
  }
  "version" {
    Ensure-Rabbit
    & $RabbitBin version
  }
  default {
    Write-Host @"

Rabbit PowerShell launcher
  .\rabbit.ps1 build
  .\rabbit.ps1 dashboard
  .\rabbit.ps1 chat
  .\rabbit.ps1 stats
  .\rabbit.ps1 gradio
  .\rabbit.ps1 version

Env: RABBIT_MODEL, RABBIT_PORT
Or use rabbit.bat for a full interactive menu.
"@
  }
}
