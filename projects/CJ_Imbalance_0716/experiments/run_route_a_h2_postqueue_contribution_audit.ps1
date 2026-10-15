$ErrorActionPreference = 'Stop'
$project = 'C:\LJQ_code\CJ_Imbalance_0716'
$python = 'D:\anaconda3\envs\PY10\python.exe'
$queueRoot = Join-Path $project 'outputs\tapc\route_a_h2_multipulse_queue'
$postqueueReceipt = Join-Path $queueRoot 'postqueue_audit_receipt.json'
$deadline = (Get-Date).AddHours(5)

while (-not (Test-Path -LiteralPath $postqueueReceipt)) {
    $active = @(Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe'" | Where-Object { $_.CommandLine -like '*run_route_a_h2_postqueue_audit.ps1*' })
    if ($active.Count -eq 0) {
        Start-Sleep -Seconds 5
        if (-not (Test-Path -LiteralPath $postqueueReceipt)) { throw 'postqueue H2 audit stopped without a completion receipt' }
        break
    }
    if ((Get-Date) -gt $deadline) { throw 'postqueue H2 audit did not complete before contribution-audit deadline' }
    Start-Sleep -Seconds 15
}

$postqueue = Get-Content -Raw $postqueueReceipt | ConvertFrom-Json
if ($postqueue.status -ne 'completed_postqueue_audit' -or $postqueue.external_used -or $postqueue.locked_test_used -or $postqueue.test_v2_used -or $postqueue.test_v3_used) { throw 'invalid H2 postqueue receipt' }
$ljqRoot = Split-Path -Parent $project
$h2DataRoot = Get-ChildItem -LiteralPath $ljqRoot -Directory |
    Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'h2_multipulse_sixlabel_development-mp-v1_20260803_r1\closedset_6class.mat') } |
    Select-Object -First 1
if ($null -eq $h2DataRoot) { throw 'cannot locate H2 development source under C:\LJQ_code' }
$development = Join-Path $h2DataRoot.FullName 'h2_multipulse_sixlabel_development-mp-v1_20260803_r1\closedset_6class.mat'
$output = Join-Path $project 'outputs\tapc\route_a_h2_multipulse_expert_contributions.json'
& $python (Join-Path $project 'experiments\audit_route_a_h2_expert_contributions.py') --development-path $development --output $output
if ($LASTEXITCODE -ne 0) { throw "expert contribution audit failed with exit=$LASTEXITCODE" }
$value = Get-Content -Raw $output | ConvertFrom-Json
if ($value.status -ne 'completed_development_expert_contribution_audit' -or $value.external_used -or $value.locked_test_used -or $value.test_v2_used -or $value.test_v3_used) { throw 'invalid contribution audit result' }
$receipt = [ordered]@{ status = 'completed_postqueue_contribution_audit'; experiment = 'RouteA_H2_MultiPulseSixLabel'; contribution_audit = $output; record_count = @($value.records).Count; external_used = $false; locked_test_used = $false; test_v2_used = $false; test_v3_used = $false }
$receipt | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 (Join-Path $queueRoot 'postqueue_contribution_audit_receipt.json')
Write-Output 'H2_POSTQUEUE_CONTRIBUTION_AUDIT_COMPLETE'
