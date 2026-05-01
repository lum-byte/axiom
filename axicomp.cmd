@echo off
setlocal enabledelayedexpansion

pushd "%~dp0" >nul

set "SCRIPT_DIR=%~dp0"
set "AXIOM_UNC_BUILD="
if "%SCRIPT_DIR:~0,2%"=="\\" set "AXIOM_UNC_BUILD=1"

set "CONFIG=Release"
set "PLATFORM=x64"
set "RELEASE_ROOT=Releases-x64"
set "WIN_BIN=%RELEASE_ROOT%\compiled\binaries\Winx64"
set "ROOT_DLL=%RELEASE_ROOT%\axi.dll"
set "ROOT_DEP_RESOLVER=%RELEASE_ROOT%\axi-dep-resolver.exe"
set "ROOT_MCP=%RELEASE_ROOT%\tag-mcp.exe"
set "BIN_RUNTIME=%WIN_BIN%\axirt.dll"
set "BIN_DEP_RESOLVER=%WIN_BIN%\axi-dep-resolver.exe"
set "BIN_MCP=%WIN_BIN%\tag-mcp.exe"

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

goto :build_go

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

if exist tools\axi_dep_resolver\axi_dep_resolver.c (
  if /I "%CC_NAME%"=="cl.exe" (
    "%CC%" /nologo /O2 tools\axi_dep_resolver\axi_dep_resolver.c /Fe:"%BIN_DEP_RESOLVER%" shell32.lib advapi32.lib user32.lib
  ) else (
    "%CC%" -O2 -Wall -Wextra -municode -mwindows tools\axi_dep_resolver\axi_dep_resolver.c -o "%BIN_DEP_RESOLVER%" -lshell32 -ladvapi32 -luser32
  )
  if errorlevel 1 (
    popd >nul
    exit /b 1
  )
)

:build_go
set "GO_EXE="
for %%G in (go.exe) do (
  for /f "delims=" %%P in ('where %%G 2^>nul') do (
    if not defined GO_EXE set "GO_EXE=%%P"
  )
)
if /I not "%AXIOM_BUILD_GO_MCP%"=="1" (
  echo Go MCP build skipped by default; use "make build-go-windows" from WSL or set AXIOM_BUILD_GO_MCP=1 on a local Windows checkout.
) else if defined AXIOM_UNC_BUILD (
  echo Go MCP build skipped on UNC/WSL path; use "make build-go-windows" from WSL or run axicomp.cmd from a local Windows checkout.
) else if defined GO_EXE (
  "%GO_EXE%" build -o "%BIN_MCP%" .\cmd\tag-mcp
  if errorlevel 1 (
    echo Go MCP build failed; tag-mcp.exe skipped.
  )
) else (
  echo Go compiler not found; tag-mcp.exe skipped.
)

:copy_aliases
copy /y "%BIN_RUNTIME%" "%ROOT_DLL%" >nul
if exist "%BIN_DEP_RESOLVER%" copy /y "%BIN_DEP_RESOLVER%" "%ROOT_DEP_RESOLVER%" >nul
if exist "%BIN_MCP%" copy /y "%BIN_MCP%" "%ROOT_MCP%" >nul
if exist "%RELEASE_ROOT%\axirt.dll" del /q "%RELEASE_ROOT%\axirt.dll"
if exist "%RELEASE_ROOT%\axirt.so" del /q "%RELEASE_ROOT%\axirt.so"
if exist "%WIN_BIN%\axi.dll" del /q "%WIN_BIN%\axi.dll"
echo %ROOT_DLL%
echo %BIN_RUNTIME%
if exist "%ROOT_DEP_RESOLVER%" echo %ROOT_DEP_RESOLVER%
if exist "%BIN_DEP_RESOLVER%" echo %BIN_DEP_RESOLVER%
if exist "%ROOT_MCP%" echo %ROOT_MCP%
if exist "%BIN_MCP%" echo %BIN_MCP%

popd >nul
