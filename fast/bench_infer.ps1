# CPU inference benchmark: prefill (prompt processing) and token generation.
# Runs each model with a ~512-token prompt and greedy sampling, and scrapes the
# two timings the engine now reports separately.
$ErrorActionPreference = 'Continue'
$exe    = Join-Path $PSScriptRoot 'ligguf-fast.exe'
$prompt = Join-Path $PSScriptRoot 'bench_prompt.txt'
$models = Join-Path 'C:\models' '*'

$list = @(
    'smollm-135m-instruct-add-basics-q8_0.gguf',
    'tinyllama-15M-stories-Q2_K.gguf',
    'Tiny-LLM.Q4_K_M.gguf',
    'TinyLLama-v0.Q8_0.gguf',
    'Tiny-Moe.Q4_K_M.gguf',
    'qwen2.5-0.5b-instruct-q2_k.gguf',
    'tinyllama-1.1b-chat-v1.0-q2_k.gguf',
    'MiniCPM5-1B-Q8_0.gguf',
    'MiniCPM5-1B.i1-Q6_K.gguf',
    'Llama-3.2-1B-Instruct-Q8_0.gguf',
    'Qwen2.5-1.5B-MoE-A0.9B-Q4_K_M.gguf',
    'Firefly-v5.Q4_K_M.gguf',
    'Uraion-Agent-Steer-Q4_K_M.gguf'
)

$rows = @()
foreach ($m in $list) {
    $path = Join-Path 'C:\models' $m
    if (!(Test-Path $path)) { $rows += [pscustomobject]@{ Model=$m; Prefill='missing'; Decode=''; Note='' }; continue }
    $sw = [diagnostics.stopwatch]::StartNew()
    $out = & $exe -m $path -f $prompt -n 128 -c 1024 -G 2>&1 | Out-String
    $sw.Stop()
    $p = [regex]::Match($out, 'Prefill:\s+(\d+) tokens in\s+([\d.]+) ms\s+->\s+([\d.]+) tok/s')
    $g = [regex]::Match($out, 'Token generation:\s+(\d+) tokens in\s+([\d.]+) ms\s+->\s+([\d.]+) tok/s')
    if ($p.Success -and $g.Success) {
        $rows += [pscustomobject]@{
            Model   = $m
            Prefill = ('{0} tok/s' -f [math]::Round([double]$p.Groups[3].Value,1))
            Decode  = ('{0} tok/s' -f [math]::Round([double]$g.Groups[3].Value,1))
            Note    = ('prefill {0} tok / {1} ms, decode {2} tok / {3} ms, wall {4:N1}s' -f `
                       $p.Groups[1].Value, $p.Groups[2].Value, $g.Groups[1].Value, $g.Groups[2].Value, $sw.Elapsed.TotalSeconds)
        }
    } else {
        $last = (($out -split "`r?`n" | Where-Object { $_.Trim() } | Select-Object -Last 2) -join ' | ')
        $rows += [pscustomobject]@{ Model=$m; Prefill='FAILED'; Decode=''; Note=$last }
    }
}
$rows | Format-Table -AutoSize | Out-String -Width 400 | Write-Host
Write-Host "--- raw ---"
$rows | ForEach-Object { '{0}: {1} | {2}' -f $_.Model, $_.Prefill, $_.Note }
