@echo off
REM Activates the build env for kernel-opt on Windows:
REM   - MSVC 14.29 (VS 2019 toolset; CUDA 12.6 compatible)
REM   - DISTUTILS_USE_SDK=1 so torch.utils.cpp_extension trusts the active VC env
REM   - Project venv on PATH
REM
REM Usage: from a fresh cmd.exe, run:
REM   call scripts\dev_env.bat
REM Then `pip install -e .`, `pytest`, `python ...` all work.
REM
REM From PowerShell, use:
REM   cmd /c "scripts\dev_env.bat && <command>"
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.29
if errorlevel 1 exit /b %errorlevel%
set DISTUTILS_USE_SDK=1
set PATH=%~dp0..\.venv\Scripts;%PATH%
echo [dev_env] MSVC 14.29 + venv ready.
