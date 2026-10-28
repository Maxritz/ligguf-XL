# Run the engine's -Z quant self-test across every loadable model and aggregate
# per ggml type, so a broken codec shows up regardless of which model carries it.
$ErrorActionPreference = 'Continue'
$exe = Join-Path $PSScriptRoot 'ligguf-fast.exe'

$models = @(
    'smollm-135m-instruct-add-basics-q8_0.gguf',
    'Llama-3.2-1B-Instruct-Q8_0.gguf',
    'MiniCPM5-1B-Q8_0.gguf',
    'MiniCPM5-1B.i1-Q6_K.gguf',
    'qwen2.5-0.5b-instruct-q2_k.gguf',
    'tinyllama-1.1b-chat-v1.0-q2_k.gguf',
    'tinyllama-15M-stories-Q2_K.gguf',
    'tinyllama-15M-stories-Q3_K_M.gguf',
    'tiny-llama3-test-Q3_K_M.gguf',
    'tinystories-gpt-0.1-3m.Q2_K.gguf',
    'Tiny-LLM.Q4_K_M.gguf',
    'Tiny-Moe.Q4_K_M.gguf',
    'TinyLLama-v0.Q8_0.gguf',
    'Qwen2.5-1.5B-MoE-A0.9B-Q4_K_M.gguf',
    'Llama-3.2-1B-4B-Quad-MoE.Q4_K_M.gguf',
    'Phi-3-mini-4k-instruct-q4.gguf'
)

$byType = @{}
$modelRows = @()
foreach ($m in $models) {
    $path = Join-Path 'C:\models' $m
    if (!(Test-Path $path)) { $modelRows += [pscustomobject]@{ Model=$m; Verdict='missing'; Types='' }; continue }
    $out = & $exe -m $path -Z 2>&1 | Out-String
    $verdict = if ($out -match 'Self-test PASSED') { 'PASS' } elseif ($out -match 'Self-test FAILED') { 'FAIL' } else { 'no-run' }
    $types = @()
    foreach ($mm in [regex]::Matches($out, 'type=(\d+)\s+n=(\d+)\s+maxrel=([\d.e+\-]+)\s+(ok|FAIL)\s+\((\w+)\)')) {
        $t = [int]$mm.Groups[1].Value
        $v = $mm.Groups[4].Value
        $rel = [double]$mm.Groups[3].Value
        $where = $mm.Groups[5].Value
        $types += ("{0}:{1}" -f $t, $(if ($v -eq 'ok') { 'ok' } else { 'FAIL' }))
        if (!$byType.ContainsKey($t)) { $byType[$t] = @() }
        $byType[$t] += [pscustomobject]@{ Model=$m; Where=$where; Rel=$rel; Verdict=$v }
    }
    $modelRows += [pscustomobject]@{ Model=$m; Verdict=$verdict; Types=($types -join ' ') }
}

Write-Host "=== per model ==="
$modelRows | Format-Table -AutoSize | Out-String -Width 220 | Write-Host

Write-Host "=== per quant type (worst case across all models) ==="
$names = @{0='F32';1='F16';2='Q4_0';3='Q4_1';6='Q5_0';7='Q5_1';8='Q8_0';10='Q2_K';11='Q3_K';
           12='Q4_K';13='Q5_K';14='Q6_K';20='IQ4_NL';30='BF16';34='TQ1_0';35='TQ2_0';39='MXFP4';143='PTQ1_0'}
$rows = @()
foreach ($t in ($byType.Keys | Sort-Object)) {
    $rs = $byType[$t]
    $worst = $rs | Sort-Object Rel -Descending | Select-Object -First 1
    $nbad = ($rs | Where-Object { $_.Verdict -ne 'ok' }).Count
    $rows += [pscustomobject]@{
        Type   = ("{0} {1}" -f $t, $(if ($names.ContainsKey($t)) { $names[$t] } else { '?' }))
        Seen   = $rs.Count
        Worst  = ('{0:e2}' -f $worst.Rel)
        Broken = $nbad
        Verdict= $(if ($nbad -gt 0) { 'BROKEN' } else { 'ok' })
        Example= ("{0} ({1})" -f $worst.Model, $worst.Where)
    }
}
$rows | Format-Table -AutoSize | Out-String -Width 220 | Write-Host
