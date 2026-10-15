param(
    [string]$Python = 'D:\anaconda3\envs\PY10\python.exe'
)
$ErrorActionPreference = 'Stop'
# Keep this Windows PowerShell 5-compatible: the script is consumed without
# relying on a UTF-8 BOM, so discover the user-provided data root rather than
# embedding a non-ASCII directory literal in a queue launcher.
$dataRoot = Get-ChildItem -LiteralPath 'C:\LJQ_code' -Directory | Where-Object {
    $_.Name -notlike 'CJ_*' -and (Test-Path -LiteralPath (Join-Path $_.FullName 'h3_vft_three_strata_train-3strata-v1_20260803_r1'))
} | Select-Object -First 1 -ExpandProperty FullName
if ([string]::IsNullOrWhiteSpace($dataRoot)) { throw 'H3 data root could not be discovered under C:\LJQ_code' }
$train = Join-Path $dataRoot 'h3_vft_three_strata_train-3strata-v1_20260803_r1\closedset_6class.mat'
$dev = Join-Path $dataRoot 'h3_vft_three_strata_development-3strata-v1_20260803_r1\closedset_6class.mat'
$auditRoot = 'outputs\tapc\route_a_h3_vft_three_strata_generation'
$queueRoot = 'outputs\tapc\route_a_h3_vft_three_strata_queue'
New-Item -ItemType Directory -Force -Path $queueRoot | Out-Null
$seeds = 20261003,20261011,20261019,20261027,20261104
$modes = 'T0-h3','T1-h3','T2-h3'
$records = @()
foreach ($seed in $seeds) {
    foreach ($mode in $modes) {
        $safeMode = $mode.Replace('-', '_')
        $root = "outputs\tapc\route_a_h3_vft_$safeMode`_seed$seed"
        $receipt = Join-Path $root 'run_receipt.json'
        if (Test-Path -LiteralPath $receipt) {
            $records += @{ mode=$mode; seed=$seed; root=$root; action='skipped_existing_receipt' }
            continue
        }
        $log = Join-Path $queueRoot "$safeMode`_seed$seed.log"
        & $Python experiments\run_route_a_h3_vft_three_strata_models.py --train-path $train --development-path $dev --train-audit (Join-Path $auditRoot 'train_3strata_v1_r1_source_audit.json') --development-audit (Join-Path $auditRoot 'development_3strata_v1_r1_source_audit.json') --output-root $root --modes $mode --seeds $seed --epochs 24 --learning-rate 0.0003 --batch-size 16 --width 32 --expert-dim 16 *>&1 | Tee-Object -FilePath $log
        if ($LASTEXITCODE -ne 0) { throw "H3 formal fit failed: $mode / $seed" }
        $records += @{ mode=$mode; seed=$seed; root=$root; action='completed'; log=$log }
    }
}
$value = @{ status='completed_development_queue'; experiment='RouteA_H3_VFTThreeStrataIdentifiability'; modes=$modes; seeds=$seeds; records=$records; external_used=$false; locked_test_used=$false; test_v2_used=$false; test_v3_used=$false; calibration='none'; threshold_selection='none' }
$value | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 (Join-Path $queueRoot 'queue_receipt.json')
