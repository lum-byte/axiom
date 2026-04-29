@echo off
setlocal enabledelayedexpansion

pushd "%~dp0" >nul

set "CONFIG=Release"
set "PLATFORM=x64"
set "RELEASE_ROOT=Releases-x64"
set "WIN_BIN=%RELEASE_ROOT%\compiled\binaries\Winx64"
set "ROOT_DLL=%RELEASE_ROOT%\axi.dll"
set "ROOT_RUNTIME=%RELEASE_ROOT%\axirt.dll"
set "BIN_DLL=%WIN_BIN%\axi.dll"
set "BIN_RUNTIME=%WIN_BIN%\axirt.dll"

if /I "%~1"=="clean" (
  if exist "%RELEASE_ROOT%" rmdir /s /q "%RELEASE_ROOT%"
  popd >nul
  exit /b 0
)

if not exist "%WIN_BIN%" mkdir "%WIN_BIN%"

set "MSBUILD_EXE="
for %%M in (msbuild.exe) do (
  for /f "delims=" %%P in ('where %%M 2^>nul') do (
    if not defined MSBUILD_EXE set "MSBUILD_EXE=%%P"
  )
)

if defined MSBUILD_EXE (
  "%MSBUILD_EXE%" Axiom.sln /m /p:Configuration=%CONFIG% /p:Platform=%PLATFORM%
  if errorlevel 1 goto :fallback
) else (
  goto :fallback
)

goto :copy_aliases

:fallback
set "CC="
for %%C in (cl.exe gcc.exe) do (
  for /f "delims=" %%P in ('where %%C 2^>nul') do (
    if not defined CC set "CC=%%P"
  )
)

if not defined CC (
  echo No Visual Studio MSBuild, cl.exe, or gcc.exe compiler found.
  popd >nul
  exit /b 1
)

for %%F in ("%CC%") do set "CC_NAME=%%~nxF"
if /I "%CC_NAME%"=="cl.exe" (
  "%CC%" /nologo /O2 /LD axiom_runtime\axiom_runtime.c /Fe:"%BIN_RUNTIME%"
) else (
  "%CC%" -std=c11 -O2 -Wall -Wextra -shared axiom_runtime\axiom_runtime.c -o "%BIN_RUNTIME%"
)
if errorlevel 1 (
  popd >nul
  exit /b 1
)

:copy_aliases
copy /y "%BIN_RUNTIME%" "%BIN_DLL%" >nul
copy /y "%BIN_RUNTIME%" "%ROOT_RUNTIME%" >nul
copy /y "%BIN_RUNTIME%" "%ROOT_DLL%" >nul
echo %ROOT_DLL%
echo %ROOT_RUNTIME%
echo %BIN_DLL%
echo %BIN_RUNTIME%

popd >nul
