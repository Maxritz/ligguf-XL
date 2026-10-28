# CPU inference benchmark: prefill (prompt processing) and token generation.
#
# Runs each model with a fixed prompt and greedy sampling, then scrapes the two
# timings the engine reports separately. Small-context toy models get a retry
# with a shorter prompt and window, since a 1k-token prompt does not fit in a
# 256-token model and would otherwise record nothing.
#
# Read-only: it does not touch the power scheme. The clock is probed and recorded
# with the results, because absolute tok/s is meaningless without it.
$ErrorActionPreference = 'Continue'
$exe     = Join-Path $PSScriptRoot 'ligguf-fast.exe'
$prompt  = Join-Path $PSScriptRoot 'bench_prompt.txt'
$promptS = Join-Path $PSScriptRoot 'bench_prompt_small.txt'
$md      = Join-Path $PSScriptRoot 'BENCH_INFER.md'

# name, tokens to generate, long prompt ok?
$list = @(
    @('smollm-135m-instruct-add-basics-q8_0.gguf', 128),
    @('tinyllama-15M-stories-Q2_K.gguf',            64),
    @('TinyLLama-v0.Q8_0.gguf',                    128),
    @('Tiny-LLM.Q4_K_M.gguf',                      128),
    @('Tiny-Moe.Q4_K_M.gguf',                      128),
    @('tinystories-gpt-0.1-3m.Q2_K.gguf',          128),
    @('qwen2.5-0.5b-instruct-q2_k.gguf',           128),
    @('tinyllama-1.1b-chat-v1.0-q2_k.gguf',        128),
    @('MiniCPM5-1B-Q8_0.gguf',                      64),
    @('MiniCPM5-1B.i1-Q6_K.gguf',                   64),
    @('Llama-3.2-1B-Instruct-Q8_0.gguf',            64),
    @('Qwen2.5-1.5B-MoE-A0.9B-Q4_K_M.gguf',         64),
    @('Llama-3.2-1B-4B-Quad-MoE.Q4_K_M.gguf',       64),
    @('Phi-3-mini-4k-instruct-q4.gguf',             32)
)

# Reported, not timed here: these are big enough that a single-thread sweep
# costs many minutes each, or they have no loader yet.
$skipped = @(
    'Uraion-Agent-Steer-Q4_K_M.gguf (4.4 GB, supported but slow)',
    'Phi-4-reasoning-plus-Q3_K_M.gguf (6.9 GB, supported but slow)',
    'Ternary-Bonsai / Qwen3.5-9B / Qwen3.8-4B / MiniCPM-V (qwen35, GDN not implemented)',
    'Firefly-v5 / gemma-4-E4B (gemma4, no loader)',
    'Mellum2.1-12B (mellum, no loader)'
)

function Run-One($path, $pf, $n, $c) {
    $sw = [diagnostics.stopwatch]::StartNew()
    $out = & $exe -m $path -f $pf -n $n -c $c -G 2>&1 | Out-String
    $sw.Stop()
    return @{ out = $out; sec = $sw.Elapsed.TotalSeconds }
}

$rows = @()
foreach ($e in $list) {
    $name = $e[0]; $n = [int]$e[1]
    $path = Join-Path 'C:\models' $name
    if (!(Test-Path $path)) { $rows += [pscustomobject]@{ Model=$name; Prefill='missing'; Decode=''; Note='' }; continue }

    $r = Run-One $path $prompt $n 2048
    $cfg = "c=2048 n=$n"
    if ($r.out -notmatch 'Prefill:') {
        # context too small for the long prompt: retry with the short one
        $r = Run-One $path $promptS 48 256
        $cfg = 'c=256 n=48 short'
    }

    $p = [regex]::Match($r.out, 'Prefill:\s+(\d+) tokens in\s+([\d.]+) ms\s+->\s+([\d.]+) tok/s')
    $g = [regex]::Match($r.out, 'Token generation:\s+(\d+) tokens in\s+([\d.]+) ms\s+->\s+([\d.]+) tok/s')

    if ($p.Success -and $g.Success) {
        $rows += [pscustomobject]@{
            Model   = $name
            Prefill = ('{0}' -f [math]::Round([double]$p.Groups[3].Value,1))
            Decode  = ('{0}' -f [math]::Round([double]$g.Groups[3].Value,1))
            Note    = ('{0} tok prefill in {1} ms, {2} tok decode in {3} ms [{4}]' -f `
                       $p.Groups[1].Value, $p.Groups[2].Value, $g.Groups[1].Value, $g.Groups[2].Value, $cfg)
        }
    } else {
        $last = (($r.out -split "`r?`n" | Where-Object { $_.Trim() } | Select-Object -Last 1) -join ' ')
        $rows += [pscustomobject]@{ Model=$name; Prefill='FAILED'; Decode=''; Note=$last }
    }
}

$clock = (& powershell -NoProfile -ExecutionPolicy Bypass -File `
          (Join-Path $PSScriptRoot '..\..\npuxl-improved\tools\laptop\probe_clock.ps1') 2>&1 | Out-String)
if (!$clock -or $clock -notmatch 'GHz') {
    $clock = (& powershell -NoProfile -ExecutionPolicy Bypass -File `
              (Join-Path $PSScriptRoot 'probe_clock.ps1') 2>&1 | Out-String)
}

Write-Host $clock
$rows | Format-Table -AutoSize | Out-String -Width 400 | Write-Host

$lines = @()
$lines += '# CPU inference: prefill and token generation'
$lines += ''
$lines += 'Single-threaded decode (no thread pool yet), greedy sampling, `-G`.'
$lines += 'Prefill and decode are timed separately by the engine.'
$lines += ''
$lines += 'Important caveat: **prefill is not batched**. The prompt is fed one token'
$lines += 'per forward pass, exactly like decode, so prefill tok/s tracks decode tok/s'
$lines += 'instead of running an order of magnitude faster. Batching the prompt into'
$lines += 'a real GEMM is the single biggest prefill win available and is not done yet.'
$lines += ''
$lines += '```'
$lines += ($clock.Trim())
$lines += '```'
$lines += ''
$lines += '| model | prefill tok/s | decode tok/s | detail |'
$lines += '|---|---|---|---|'
foreach ($r in $rows) { $lines += ('| {0} | {1} | {2} | {3} |' -f $r.Model, $r.Prefill, $r.Decode, $r.Note) }
$lines += ''
$lines += 'Not swept here:'
foreach ($s in $skipped) { $lines += ('- ' + $s) }
$lines += ''
Set-Content -Path $md -Value $lines -Encoding UTF8
Write-Host "wrote $md"
