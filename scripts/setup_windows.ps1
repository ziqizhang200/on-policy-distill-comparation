param(
    [string]$Root = 'D:\opdistill'
)
$ErrorActionPreference = 'Stop'
$EnvDirectory = Join-Path $Root 'envs'
foreach ($relative in @('envs','conda-pkgs','tmp','pip-cache','hf-cache')) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Root $relative) | Out-Null
}
$env:CONDA_ENVS_PATH = $EnvDirectory
$env:CONDA_PKGS_DIRS = Join-Path $Root 'conda-pkgs'
$env:TEMP = Join-Path $Root 'tmp'
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = Join-Path $Root 'pip-cache'
$env:CONDA_REPODATA_USE_ZST = 'false'
conda env create --file configs/environment-windows.yml
if ($LASTEXITCODE -ne 0) { throw 'Conda environment creation failed' }
conda config --add envs_dirs $EnvDirectory
if ($LASTEXITCODE -ne 0) { throw 'Conda name registration failed' }
$Prefix = Join-Path $EnvDirectory 'opdistill'
conda env config vars set --prefix $Prefix "HF_HOME=$(Join-Path $Root 'hf-cache')" "PIP_CACHE_DIR=$env:PIP_CACHE_DIR" "TEMP=$env:TEMP" "TMP=$env:TMP" "CONDA_PKGS_DIRS=$env:CONDA_PKGS_DIRS" PYTHONUTF8=1
if ($LASTEXITCODE -ne 0) { throw 'Conda environment variables setup failed' }
$Python = Join-Path $Prefix 'python.exe'
& $Python -m pip install --index-url https://download.pytorch.org/whl/cu124 'torch==2.6.0+cu124'
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed; keep TLS and hash verification enabled' }
& $Python -m pip install --index-url https://pypi.org/simple -r configs/requirements-windows-lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Application dependencies installation failed' }
& $Python -m ipykernel install --sys-prefix --name opdistill --display-name 'Python (opdistill)'
if ($LASTEXITCODE -ne 0) { throw 'Jupyter kernel setup failed' }
& $Python scripts/verify_environment.py
if ($LASTEXITCODE -ne 0) { throw 'Environment verification failed' }
Write-Host 'opdistill ready: conda activate opdistill'
