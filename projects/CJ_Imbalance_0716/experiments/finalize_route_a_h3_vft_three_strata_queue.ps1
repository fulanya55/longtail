param(
    [string]$Python = 'D:\anaconda3\envs\PY10\python.exe'
)

# This post-queue runner intentionally performs no model selection.  It only
# waits for the already-frozen H3 queue, then invokes its deterministic audit,
# preregistered five-seed summary, and regression tests.  It never names or
# opens an external, locked, v2, or v3 test source.
$ErrorActionPreference = 'Stop'
$queueRoot = 'outputs\tapc\route_a_h3_vft_three_strata_queue'
$modes = 'T0-h3','T1-h3','T2-h3'
$seeds = 20261003,20261011,20261019,20261027,20261104
$expected = foreach ($seed in $seeds) {
    foreach ($mode in $modes) {
        $safeMode = $mode.Replace('-', '_')
        [pscustomobject]@{ mode=$mode; seed=$seed; root="outputs\tapc\route_a_h3_vft_$safeMode`_seed$seed" }
    }
}
New-Item -ItemType Directory -Force -Path $queueRoot | Out-Null
$receiptPath = Join-Path $queueRoot 'postqueue_receipt.json'
try {
    while ($true) {
        $workers = @(Get-CimInstance Win32_Process | Where-Object {
            $_.Name -eq 'python.exe' -and $_.CommandLine -like '*run_route_a_h3_vft_three_strata_models.py*'
        })
        if ($workers.Count -eq 0) { break }
        Start-Sleep -Seconds 30
    }
    foreach ($run in $expected) {
        $path = Join-Path $run.root 'run_receipt.json'
        if (-not (Test-Path -LiteralPath $path)) { throw "missing completed H3 receipt: $path" }
        $value = Get-Content -Raw $path | ConvertFrom-Json
        if ($value.status -ne 'completed_development_unaggregated' -or $value.smoke -or $value.modes.Count -ne 1 -or $value.modes[0] -ne $run.mode -or [int]$value.seeds[0] -ne $run.seed) {
            throw "invalid formal H3 receipt: $path"
        }
        if ($value.external_used -or $value.locked_test_used -or $value.test_v2_used -or $value.test_v3_used) { throw "test-boundary violation: $path" }
    }
    & $Python experiments\audit_route_a_h3_vft_three_strata_artifacts.py
    if ($LASTEXITCODE -ne 0) { throw 'H3 artifact audit failed' }
    & $Python experiments\summarize_route_a_h3_vft_three_strata_models.py
    if ($LASTEXITCODE -ne 0) { throw 'H3 development summary failed' }
    & $Python experiments\report_route_a_h3_vft_three_strata_results.py
    if ($LASTEXITCODE -ne 0) { throw 'H3 development report failed' }
    & $Python -m pytest -q tests\test_summarize_route_a_h3_vft_three_strata_models.py tests\test_audit_route_a_h3_vft_three_strata_artifacts.py tests\test_route_a_h3_vft_three_strata_models.py
    if ($LASTEXITCODE -ne 0) { throw 'H3 regression test suite failed' }
    $summary = Get-Content -Raw 'outputs\tapc\route_a_h3_vft_three_strata_development_summary.json' | ConvertFrom-Json
    $record = @{ status='completed_postqueue_audit'; experiment='RouteA_H3_VFTThreeStrataIdentifiability'; formal_run_count=$expected.Count; development_gate_passed=[bool]$summary.gate.passed; artifact_audit='outputs/tapc/route_a_h3_vft_three_strata_artifact_audit.json'; development_summary='outputs/tapc/route_a_h3_vft_three_strata_development_summary.json'; development_report='docs/results/route_a_h3_vft_three_strata_results_20260803.md'; external_used=$false; locked_test_used=$false; test_v2_used=$false; test_v3_used=$false; completed_at=(Get-Date).ToString('o') }
    $record | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 $receiptPath
} catch {
    $record = @{ status='failed_postqueue_audit'; experiment='RouteA_H3_VFTThreeStrataIdentifiability'; error=$_.Exception.Message; external_used=$false; locked_test_used=$false; test_v2_used=$false; test_v3_used=$false; completed_at=(Get-Date).ToString('o') }
    $record | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 $receiptPath
    throw
}
