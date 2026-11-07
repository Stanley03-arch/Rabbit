@echo off
setlocal
cd /d "%~dp0"

set "MODEL=%RABBIT_MODEL%"
if "%MODEL%"=="" set "MODEL=models\tiny-test.rabbit"
set "PORT=%RABBIT_PORT%"
if "%PORT%"=="" set "PORT=7860"

if not exist "rabbit.exe" if not exist "rabbit" (
  echo Building rabbit.exe ...
  where gcc >nul 2>&1
  if errorlevel 1 (
    echo [error] gcc not found. Install MinGW/MSYS2 or run rabbit.bat option 1.
    pause
    exit /b 1
  )
  gcc -O2 -Wall -Wextra -std=c11 -I./src ^
    src/main.c src/memory.c src/model.c src/moe.c src/quant.c ^
    src/tokenizer.c src/utils.c src/attention.c src/forward_v2.c ^
    -o rabbit.exe -lm
  if errorlevel 1 (
    echo [error] build failed.
    pause
    exit /b 1
  )
)

if not exist "%MODEL%" (
  echo Model missing: %MODEL%
  echo Creating demo model...
  where python >nul 2>&1 && set PY=python || set PY=py -3
  if not exist models mkdir models
  %PY% tools\convert_hf_to_rabbit.py --demo --out "%MODEL%" --hidden-dim 128 --n-experts 8 --quant 4
)

where python >nul 2>&1 && set PY=python || set PY=py -3
%PY% -c "import flask" 2>nul || %PY% -m pip install flask

echo.
echo  Rabbit dashboard  http://127.0.0.1:%PORT%/
echo  Model: %MODEL%
echo  Ctrl+C to stop.
echo.
%PY% tools\rabbit_dashboard.py --model "%MODEL%" --host 127.0.0.1 --port %PORT%
pause
