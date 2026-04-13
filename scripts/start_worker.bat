@echo off
setlocal
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
cd /d %PROJECT_ROOT%
set PYTHONPATH=%PROJECT_ROOT%
echo Starting LiveVoiceTranscriptor worker
python -m app.workers.worker
