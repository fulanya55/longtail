$ErrorActionPreference = 'Stop'
$project = 'C:\LJQ_code\CJ_Imbalance_0716'
$python = 'D:\anaconda3\envs\PY10\python.exe'
$queueRoot = Join-Path $project 'outputs\tapc\route_a_h2_multipulse_queue'
$queueReceipt = Join-Path $queueRoot 'queue_receipt.json'
$deadline = (Get-Date).AddHours(4)

while (-not (Test-Path -LiteralPath $queueReceipt)) {
    $active = @(Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe'" | Where-Object { $_.CommandLine -like '*run_route_a_h2_remaining_queue.ps1*' })
    if ($active.Count -eq 0) {
        Start-Sleep -Seconds 5
        if (-not (Test-Path -LiteralPath $queueReceipt)) { throw 'H2 formal queue stopped without a completion receipt' }
        break
    }
    if ((Get-Date) -gt $deadline) { throw 'H2 formal queue did not complete before postqueue audit deadline' }
    Start-Sleep -Seconds 15
}

$queue = Get-Content -Raw $queueReceipt | ConvertFrom-Json
if ($queue.status -ne 'completed' -or $queue.external_used -or $queue.locked_test_used -or $queue.test_v2_used -or $queue.test_v3_used) { throw 'H2 queue receipt is invalid for postqueue audit' }
$runRoots = @(Get-ChildItem (Join-Path $project 'outputs\tapc') -Directory -Filter 'route_a_h2_multipulse_c*_seed*' | Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'run_receipt.json') })
if ($runRoots.Count -ne 25) { throw "expected 25 formal run receipts, found $($runRoots.Count)" }

$externalSeal = Get-Content -Raw (Join-Path $project 'frozen\20260803_route_a_h2_external_mp_v1\external_mp_sealed_manifest.json') | ConvertFrom-Json
$testV3Seal = Get-Content -Raw (Join-Path $project 'frozen\20260731_route_a_test_v3_sealed\test_v3_sealed_manifest.json') | ConvertFrom-Json
if ($externalSeal.external_access_authorized -or $externalSeal.external_analysis_allowed -or $externalSeal.external_used) { throw 'external H2 seal invalid' }
if ($testV3Seal.test_v3.access_authorized -or $testV3Seal.test_v3.analysis_allowed -or $testV3Seal.test_v3.used) { throw 'test-v3 seal invalid' }

$artifactAudit = Join-Path $project 'outputs\tapc\route_a_h2_multipulse_artifact_audit.json'
$summary = Join-Path $project 'outputs\tapc\route_a_h2_multipulse_development_summary.json'
& $python (Join-Path $project 'experiments\audit_route_a_h2_multipulse_development_artifacts.py') --output $artifactAudit
if ($LASTEXITCODE -ne 0) { throw "artifact audit failed with exit=$LASTEXITCODE" }
& $python (Join-Path $project 'experiments\summarize_route_a_h2_multipulse_development.py') --output $summary
if ($LASTEXITCODE -ne 0) { throw "development summary failed with exit=$LASTEXITCODE" }

$auditValue = Get-Content -Raw $artifactAudit | ConvertFrom-Json
$summaryValue = Get-Content -Raw $summary | ConvertFrom-Json
$receipt = [ordered]@{
    status = 'completed_postqueue_audit'
    experiment = 'RouteA_H2_MultiPulseSixLabel'
    formal_run_count = $runRoots.Count
    artifact_audit = $artifactAudit
    development_summary = $summary
    development_gate_passed = $summaryValue.gate.passed
    c3_c4_equal_parameter_count = $auditValue.c3_c4_equal_parameter_count
    external_used = $false
    locked_test_used = $false
    test_v2_used = $false
    test_v3_used = $false
}
$receipt | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 (Join-Path $queueRoot 'postqueue_audit_receipt.json')
Write-Output 'H2_POSTQUEUE_AUDIT_COMPLETE'
