param([string]$BuildDirectory = 'D:\opdistill\tmp')
$ErrorActionPreference = 'Stop'
$Vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$VisualStudio = & $Vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $VisualStudio) { throw 'MSVC C++ tools not found' }
New-Item -ItemType Directory -Force -Path $BuildDirectory | Out-Null
$Source = Join-Path $PSScriptRoot 'verify_cuda_toolkit.cu'
$Executable = Join-Path $BuildDirectory 'verify_cuda_toolkit.exe'
$Batch = Join-Path $BuildDirectory 'verify_cuda_toolkit.cmd'
@"
@echo off
call "$VisualStudio\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
if errorlevel 1 exit /b 1
nvcc --version
cl 2>&1
nvcc -arch=sm_89 -o "$Executable" "$Source"
if errorlevel 1 exit /b 1
"$Executable"
"@ | Set-Content -Encoding ascii -LiteralPath $Batch
& $env:ComSpec /c $Batch
if ($LASTEXITCODE -ne 0) { throw 'CUDA Toolkit compilation or kernel verification failed' }
