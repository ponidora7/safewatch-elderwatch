# debug_and_try_tf.ps1
$log = "tf-install-debug.log"
"`n=== Debug run at $(Get-Date) ===`n" | Out-File $log -Encoding utf8 -Append

"--- Python executable and version ---" | Out-File $log -Append
python -c "import sys; print(sys.executable); print(sys.version)" 2>&1 | Out-File $log -Append

"--- Platform and bits ---" | Out-File $log -Appendpython -m pip check
pip 
python - <<'PY' 2>&1 | Out-File $log -Append
import platform, struct
print('platform:', platform.platform())
print('machine:', platform.machine())
print('bits:', struct.calcsize('P')*8)
PY

"--- pip version and debug ---" | Out-File $log -Append
python -m pip --version 2>&1 | Out-File $log -Append
pip debug --verbose 2>&1 | Out-File $log -Append

"--- Installed packages (top 200 chars per line) ---" | Out-File $log -Append
python -m pip list --format=columns 2>&1 | Out-File $log -Append

"--- Try installing TensorFlow variants ---" | Out-File $log -Append
$tf_versions = @("2.12.0","2.11.0","2.10.0")
foreach ($v in $tf_versions) {
    "Attempting pip install tensorflow==$v" | Out-File $log -Append
    python -m pip install --no-cache-dir "tensorflow==$v" 2>&1 | Out-File $log -Append
    "Exit code: $LASTEXITCODE" | Out-File $log -Append
}

"--- Try installing tensorflow-cpu (if available) ---" | Out-File $log -Append
python -m pip install --no-cache-dir tensorflow-cpu 2>&1 | Out-File $log -Append
"Exit code: $LASTEXITCODE" | Out-File $log -Append

"`n--- End of debug run ---`n" | Out-File $log -Append
Write-Host "Debug complete. See $log"
