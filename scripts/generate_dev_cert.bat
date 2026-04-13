@echo off
setlocal
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
cd /d %PROJECT_ROOT%
set PYTHONPATH=%PROJECT_ROOT%
python -m app.core.tls
