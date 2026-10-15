$ErrorActionPreference = 'Stop'
$project = 'C:\LJQ_code\CJ_Imbalance_0716'
$python = 'D:\anaconda3\envs\PY10\python.exe'
$train = 'C:\LJQ_code\干扰数据\h2_multipulse_sixlabel_train-mp-v1_20260803_r1\closedset_6class.mat'
$development = 'C:\LJQ_code\干扰数据\h2_multipulse_sixlabel_development-mp-v1_20260803_r1\closedset_6class.mat'
$ljqRoot = Split-Path -Parent $project
$h2DataRoot = Get-ChildItem -LiteralPath $ljqRoot -Directory |
    Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'h2_multipulse_sixlabel_train-mp-v1_20260803_r1\closedset_6class.mat') } |
    Select-Object -First 1
if ($null -eq $h2DataRoot) { throw 'cannot locate H2 train source under C:\LJQ_code' }
$train = Join-Path $h2DataRoot.FullName 'h2_multipulse_sixlabel_train-mp-v1_20260803_r1\closedset_6class.mat'
$development = Join-Path $h2DataRoot.FullName 'h2_multipulse_sixlabel_development-mp-v1_20260803_r1\closedset_6class.mat'
if (-not (Test-Path -LiteralPath $train) -or -not (Test-Path -LiteralPath $development)) { throw 'H2 train/development source resolution failed' }
$runner = Join-Path $project 'experiments\run_route_a_h2_multipulse_sixlabel.py'
$queueRoot = Join-Path $project 'outputs\tapc\route_a_h2_multipulse_queue'
New-Item -ItemType Directory -Force -Path $queueRoot | Out-Null

$seal = Get-Content -Raw (Join-Path $project 'frozen\20260803_route_a_h2_external_mp_v1\external_mp_sealed_manifest.json') | ConvertFrom-Json
$testV3 = Get-Content -Raw (Join-Path $project 'frozen\20260731_route_a_test_v3_sealed\test_v3_sealed_manifest.json') | ConvertFrom-Json
if ($seal.external_access_authorized -or $seal.external_analysis_allowed -or $seal.external_used) { throw 'external-mp seal invalid' }
if ($testV3.test_v3.access_authorized -or $testV3.test_v3.analysis_allowed -or $testV3.test_v3.used) { throw 'test-v3 seal invalid' }
$active = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -like '*run_route_a_h2_multipulse_sixlabel.py*' })
if ($active.Count -gt 0) { throw "H2 runner already active: $($active.ProcessId -join ',')" }

$records = @()
foreach ($seed in @(20260819, 20260827, 20260904)) {
    foreach ($pair in @(@('c0','C0-mp'), @('c1','C1-mp'), @('c2','C2-mp'), @('c3','C3-mp'), @('c4','C4-mp'))) {
        $short = $pair[0]; $mode = $pair[1]
        $output = Join-Path $project "outputs\tapc\route_a_h2_multipulse_${short}_seed${seed}"
        $receipt = Join-Path $output 'run_receipt.json'
        if (Test-Path -LiteralPath $receipt) {
            $records += [ordered]@{ seed = $seed; mode = $mode; action = 'skipped_existing_receipt'; receipt = $receipt }
            continue
        }
        Write-Output "H2_QUEUE_START seed=$seed mode=$mode $(Get-Date -Format o)"
        & $python $runner --train-path $train --development-path $development --output-root $output --modes $mode --seeds $seed --epochs 24 --batch-size 16 --width 32 --expert-dim 16
        if ($LASTEXITCODE -ne 0) { throw "H2 runner failed for seed=$seed mode=$mode exit=$LASTEXITCODE" }
        if (-not (Test-Path -LiteralPath $receipt)) { throw "missing receipt after seed=$seed mode=$mode" }
        $value = Get-Content -Raw $receipt | ConvertFrom-Json
        if ($value.test_v3_used -or $value.test_v2_used -or $value.locked_test_used) { throw "test boundary violation seed=$seed mode=$mode" }
        $records += [ordered]@{ seed = $seed; mode = $mode; action = 'completed'; receipt = $receipt }
        $records | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 (Join-Path $queueRoot 'progress.json')
        Write-Output "H2_QUEUE_DONE seed=$seed mode=$mode $(Get-Date -Format o)"
    }
}
$result = [ordered]@{ status = 'completed'; experiment = 'RouteA_H2_MultiPulseSixLabel_remaining_formal_queue'; records = $records; external_used = $false; locked_test_used = $false; test_v2_used = $false; test_v3_used = $false }
$result | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 (Join-Path $queueRoot 'queue_receipt.json')
Write-Output 'H2_QUEUE_COMPLETE'
