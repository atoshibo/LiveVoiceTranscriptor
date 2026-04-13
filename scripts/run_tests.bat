@echo off
REM Run the test suite

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..

cd /d %PROJECT_ROOT%

set PYTHONPATH=%PROJECT_ROOT%

echo Running LiveVoiceTranscriptor tests...
python -m pytest app/tests/ -v --tb=short %*
