# setup_safewatch.ps1
# Usage: Open PowerShell in repo root, then:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\setup_safewatch.ps1

$LogFile = "install-debug.log"
"`n=== SafeWatch install debug run at $(Get-Date) ===`n" | Out-File $LogFile -Encoding utf8 -Append

function LogRun($cmd) {
    "`n--- Running: $cmd ---`n" | Out-File $LogFile -Append
    try {
        iex $cmd 2>&1 | Out-File $LogFile -Append
    } catch {
        $_ | Out-File $LogFile -Append
    }
}

# 1) Detect python launcher for 3.11
"Detecting Python 3.11 availability..." | Out-File $LogFile -Append
$py311 = $null
try {
    $py311 = & py -3.11 -c "import sys; print(sys.executable)" 2>$null
} catch {}
if ($py311) {
    "Python 3.11 found: $py311" | Out-File $LogFile -Append
    $PythonCmd = "py -3.11"
} else {
    "Python 3.11 not found. Falling back to 'python' in PATH." | Out-File $LogFile -Append
    $PythonCmd = "python"
}

# 2) Create a fresh venv .venv-clean
$VenvPath = ".\.venv-clean"
if (Test-Path $VenvPath) {
    "Removing existing venv at $VenvPath" | Out-File $LogFile -Append
    Remove-Item -Recurse -Force $VenvPath
}
LogRun "$PythonCmd -m venv $VenvPath"

# 3) Activate venv for subsequent commands (PowerShell activation)
$Activate = Join-Path $VenvPath "Scripts\Activate.ps1"
if (-Not (Test-Path $Activate)) {
    "ERROR: venv activation script not found at $Activate" | Out-File $LogFile -Append
    Write-Host "ERROR: venv activation script not found. See $LogFile"
    exit 1
}
# Activate in current session
. $Activate
"Activated venv: $VenvPath" | Out-File $LogFile -Append

# 4) Upgrade pip and set setuptools compatible with torch
LogRun "python -m pip install --upgrade pip==23.3.1"
LogRun "python -m pip install 'setuptools<82,>=81.0.0' wheel"

# 5) If an old venv exists, attempt to uninstall problematic packages there (best-effort)
#    This only affects current active venv (we created a fresh one so usually nothing to uninstall)
LogRun "python -m pip uninstall -y torch torchvision PyYAML || echo 'uninstall attempted'"

# 6) Install core binary packages one-by-one and log output
#    Adjust versions if you know specific working versions. We prefer CPU-only torch index for Windows.
# Install TensorFlow (CPU Intel build if available)
LogRun "python -m pip install --no-cache-dir tensorflow==2.15.0"

# Install MediaPipe (use available newer release if exact pinned version missing)
# Try exact first, fallback to >=0.10.30
$mp_try = & python -c "import importlib,sys
try:
    importlib.import_module('mediapipe')
    print('mediapipe_already_installed')
except Exception:
    print('mediapipe_not_installed')" 2>$null
if ($mp_try -match "mediapipe_already_installed") {
    "mediapipe already present in this venv; skipping install." | Out-File $LogFile -Append
} else {
    LogRun "python -m pip install --no-cache-dir mediapipe==0.10.35" 
    # if previous fails, try a looser constraint
    LogRun "if ($LASTEXITCODE -ne 0) { python -m pip install --no-cache-dir 'mediapipe>=0.10.30' }"
}

# Install PyYAML compatible
LogRun "python -m pip install --no-cache-dir 'PyYAML==6.0.1'"

# Install PyTorch CPU wheels from official index (Windows CPU)
LogRun "python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu/ torch torchvision"

# 7) Install remaining requirements from requirements.txt but skip problematic pins if needed
if (Test-Path "requirements.txt") {
    # create a temp requirements copy replacing mediapipe pin if present
    $reqTemp = "requirements.temp.txt"
    (Get-Content requirements.txt) -replace 'mediapipe==0\.10\.5','mediapipe>=0.10.30' | Set-Content $reqTemp
    LogRun "python -m pip install --no-cache-dir -r $reqTemp"
    Remove-Item $reqTemp -ErrorAction SilentlyContinue
} else {
    "No requirements.txt found in repo root; skipping bulk install." | Out-File $LogFile -Append
}

# 8) Verify imports and versions; capture stdout/stderr to log
"`n--- Verification: import checks ---`n" | Out-File $LogFile -Append
LogRun "python - <<'PY'
import traceback
packages = ['tensorflow','mediapipe','torch','torchvision','yaml']
for p in packages:
    try:
        m = __import__(p if p!='yaml' else 'yaml')
        v = getattr(m,'__version__', 'unknown')
        print(p, 'imported, version:', v)
    except Exception as e:
        print(p, 'IMPORT FAILED:')
        traceback.print_exc()
PY"

# 9) Final pip check and freeze
LogRun "python -m pip check || echo 'pip check returned non-zero (see above)'"
LogRun "python -m pip freeze > requirements.lock"
"Install debug complete. See $LogFile and requirements.lock for locked versions." | Out-File $LogFile -Append

Write-Host "Setup script finished. See $LogFile in project root for details."
