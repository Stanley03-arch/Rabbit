@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Rabbit — Windows launcher
rem Run from repo root:  rabbit.bat
rem Or double-click this file.

cd /d "%~dp0"

set "RABBIT_EXE=rabbit.exe"
if exist "rabbit.exe" (
  set "RABBIT_BIN=rabbit.exe"
) else if exist "rabbit" (
  set "RABBIT_BIN=rabbit"
) else (
  set "RABBIT_BIN=rabbit.exe"
)

set "DEFAULT_MODEL=models\tiny-test.rabbit"
if not defined RABBIT_MODEL set "RABBIT_MODEL=%DEFAULT_MODEL%"
if not defined RABBIT_PORT set "RABBIT_PORT=7860"

:menu
cls
echo.
echo  ========================================
echo   Rabbit  —  local MoE engine (Windows)
echo  ========================================
echo   Model : %RABBIT_MODEL%
echo   Port  : %RABBIT_PORT%
echo  ----------------------------------------
echo   1^) Build engine          (gcc / make)
echo   2^) Create tiny test model
echo   3^) Chat ^(CLI^)
echo   4^) Stats
echo   5^) Dashboard ^(Flask + voice UI^)
echo   6^) Gradio UI
echo   7^) Version
echo   8^) Smoke probe
echo   9^) Set model path
echo   A^) Install Python deps
echo   0^) Exit
echo  ----------------------------------------
set /p "CHOICE=  Choice: "

if /i "%CHOICE%"=="1" goto build
if /i "%CHOICE%"=="2" goto testmodel
if /i "%CHOICE%"=="3" goto chat
if /i "%CHOICE%"=="4" goto stats
if /i "%CHOICE%"=="5" goto dashboard
if /i "%CHOICE%"=="6" goto gradio
if /i "%CHOICE%"=="7" goto version
if /i "%CHOICE%"=="8" goto smoke
if /i "%CHOICE%"=="9" goto setmodel
if /i "%CHOICE%"=="A" goto deps
if /i "%CHOICE%"=="a" goto deps
if /i "%CHOICE%"=="0" goto end
echo  Unknown choice.
pause
goto menu

:need_python
where python >nul 2>&1
if errorlevel 1 (
  where py >nul 2>&1
  if errorlevel 1 (
    echo  [error] Python not found. Install from https://www.python.org/ and retry.
    pause
    goto menu
  )
  set "PY=py -3"
) else (
  set "PY=python"
)
goto :eof

:need_rabbit
if not exist "%RABBIT_BIN%" (
  echo  [error] %RABBIT_BIN% not found. Choose option 1 to build first.
  pause
  goto menu
)
goto :eof

:build
echo.
echo  Building Rabbit...
where make >nul 2>&1
if not errorlevel 1 (
  make clean
  make
  if errorlevel 1 (
    echo  make failed — trying direct gcc...
    goto build_gcc
  )
  if exist rabbit set "RABBIT_BIN=rabbit"
  if exist rabbit.exe set "RABBIT_BIN=rabbit.exe"
  echo  Build OK: %RABBIT_BIN%
  pause
  goto menu
)

:build_gcc
where gcc >nul 2>&1
if errorlevel 1 (
  echo  [error] Neither make nor gcc found.
  echo  Install MinGW-w64 or MSYS2, then open a shell where gcc is on PATH.
  echo  Example: https://www.msys2.org/  or  https://winlibs.com/
  pause
  goto menu
)
echo  Compiling with gcc...
gcc -O2 -Wall -Wextra -std=c11 -I./src ^
  src/main.c src/memory.c src/model.c src/moe.c src/quant.c ^
  src/tokenizer.c src/utils.c src/attention.c src/forward_v2.c ^
  -o rabbit.exe -lm
if errorlevel 1 (
  echo  [error] gcc build failed.
  pause
  goto menu
)
set "RABBIT_BIN=rabbit.exe"
echo  Build OK: rabbit.exe
pause
goto menu

:testmodel
call :need_python
if not exist models mkdir models
echo  Creating tools\create_test_model.exe if needed...
if not exist tools\create_test_model.exe (
  where gcc >nul 2>&1
  if not errorlevel 1 (
    gcc -O2 -Wall -Wextra -o tools\create_test_model.exe tools\create_test_model.c
  )
)
if exist tools\create_test_model.exe (
  tools\create_test_model.exe models\tiny-test.rabbit
) else (
  echo  Falling back to Python demo converter...
  %PY% tools\convert_hf_to_rabbit.py --demo --out models\tiny-test.rabbit --hidden-dim 128 --n-experts 8 --quant 4
)
if errorlevel 1 (
  echo  [error] could not create test model.
) else (
  set "RABBIT_MODEL=models\tiny-test.rabbit"
  echo  Model ready: %RABBIT_MODEL%
)
pause
goto menu

:chat
call :need_rabbit
if not exist "%RABBIT_MODEL%" (
  echo  [error] Model not found: %RABBIT_MODEL%
  echo  Use option 2 to create a tiny test model, or option 9 to set path.
  pause
  goto menu
)
echo  Starting chat. Type prompts; type quit to exit.
echo.
"%RABBIT_BIN%" chat --model "%RABBIT_MODEL%" --tokens 48 --temp 0.8
pause
goto menu

:stats
call :need_rabbit
if not exist "%RABBIT_MODEL%" (
  echo  [error] Model not found: %RABBIT_MODEL%
  pause
  goto menu
)
"%RABBIT_BIN%" stats --model "%RABBIT_MODEL%"
pause
goto menu

:dashboard
call :need_python
call :need_rabbit
if not exist "%RABBIT_MODEL%" (
  echo  [error] Model not found: %RABBIT_MODEL%
  pause
  goto menu
)
echo  Installing flask if needed...
%PY% -c "import flask" 2>nul || %PY% -m pip install flask
echo.
echo  Dashboard: http://127.0.0.1:%RABBIT_PORT%/
echo  Voice UI works best in Chrome/Edge. Ctrl+C to stop.
echo.
%PY% tools\rabbit_dashboard.py --model "%RABBIT_MODEL%" --host 127.0.0.1 --port %RABBIT_PORT%
pause
goto menu

:gradio
call :need_python
call :need_rabbit
if not exist "%RABBIT_MODEL%" (
  echo  [error] Model not found: %RABBIT_MODEL%
  pause
  goto menu
)
%PY% -c "import gradio" 2>nul || %PY% -m pip install gradio
echo  Launching Gradio UI...
%PY% tools\rabbit_ui.py --model "%RABBIT_MODEL%" --repo .
pause
goto menu

:version
call :need_rabbit
"%RABBIT_BIN%" version
pause
goto menu

:smoke
call :need_rabbit
if not exist "%RABBIT_MODEL%" (
  echo  [warn] no model — creating tiny test model first...
  call :testmodel
)
echo  hello> "%TEMP%\rabbit_smoke_in.txt"
echo  quit>> "%TEMP%\rabbit_smoke_in.txt"
"%RABBIT_BIN%" chat --model "%RABBIT_MODEL%" --tokens 8 --temp 0.5 < "%TEMP%\rabbit_smoke_in.txt"
echo  smoke finished.
pause
goto menu

:setmodel
echo  Current: %RABBIT_MODEL%
set /p "RABBIT_MODEL=  New model path: "
if "%RABBIT_MODEL%"=="" set "RABBIT_MODEL=%DEFAULT_MODEL%"
echo  Model set to: %RABBIT_MODEL%
pause
goto menu

:deps
call :need_python
echo  Installing common Python packages...
%PY% -m pip install --upgrade pip
%PY% -m pip install flask gradio numpy
echo  Optional converter stack:
%PY% -m pip install safetensors huggingface_hub transformers
echo  Done.
pause
goto menu

:end
echo  Bye.
endlocal
exit /b 0
