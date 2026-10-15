$ErrorActionPreference = 'Stop'
$pythonPath = 'D:\anaconda3\envs\PY10\python.exe'
$scriptPath = 'E:\LJQ_code\CJ_Imbalance_0716\experiments\run_route_a_c1_baseline_full_worker.py'
$outputRoot = 'E:\LJQ_code\CJ_Imbalance_0716\outputs\route_a_c1_baseline_full_20260826'
$logPath = Join-Path $outputRoot 'train_stdout.log'
$errorPath = Join-Path $outputRoot 'train_stderr.log'
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
Remove-Item -LiteralPath $logPath, $errorPath -Force -ErrorAction SilentlyContinue
$argumentList = @(
  '-u', $scriptPath
)
$process = Start-Process -FilePath $pythonPath -ArgumentList $argumentList -WorkingDirectory 'E:\LJQ_code\CJ_Imbalance_0716' -RedirectStandardOutput $logPath -RedirectStandardError $errorPath -WindowStyle Hidden -PassThru
Write-Output ("PID=" + $process.Id)
