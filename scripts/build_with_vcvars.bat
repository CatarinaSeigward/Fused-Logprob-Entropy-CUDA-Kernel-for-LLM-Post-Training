@echo off
REM Activates VS 2019 MSVC toolset (14.29) which is compatible with CUDA 12.6,
REM then runs whatever command is passed in. Use this for any build/test that
REM needs cl.exe.
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.29
if errorlevel 1 exit /b %errorlevel%
%*
