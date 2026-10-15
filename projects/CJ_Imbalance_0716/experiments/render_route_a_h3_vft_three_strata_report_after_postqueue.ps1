param([string]$Python = 'D:\anaconda3\envs\PY10\python.exe')
$ErrorActionPreference = 'Stop'
$queueRoot = 'outputs\tapc\route_a_h3_vft_three_strata_queue'
$postqueue = Join-Path $queueRoot 'postqueue_receipt.json'
$receipt = Join-Path $queueRoot 'report_render_receipt.json'
try {
    while (-not (Test-Path -LiteralPath $postqueue)) { Start-Sleep -Seconds 30 }
    $post = Get-Content -Raw $postqueue | ConvertFrom-Json
    if ($post.status -ne 'completed_postqueue_audit') { throw "H3 postqueue did not complete: $($post.status)" }
    & $Python experiments\report_route_a_h3_vft_three_strata_results.py
    if ($LASTEXITCODE -ne 0) { throw 'H3 report rendering failed' }
    @{ status='completed_report_render'; postqueue_receipt=$postqueue; report='docs/results/route_a_h3_vft_three_strata_results_20260803.md'; external_used=$false; locked_test_used=$false; test_v2_used=$false; test_v3_used=$false; completed_at=(Get-Date).ToString('o') } | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 $receipt
} catch {
    @{ status='failed_report_render'; error=$_.Exception.Message; external_used=$false; locked_test_used=$false; test_v2_used=$false; test_v3_used=$false; completed_at=(Get-Date).ToString('o') } | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 $receipt
    throw
}
