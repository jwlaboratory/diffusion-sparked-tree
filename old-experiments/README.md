# old-experiments (historical archive)

> **⚠️ ERRATUM (2026-08-05): numbers in this directory are NOT citable.**
>
> The final benchmark here (`final_benchmark/`, source of the BLOG.md "+17.8% over
> DDTree") deviated from DDTree benchmarking practices in one way and inherited a
> second distortion:
>
> 1. **C++ KV compaction was disabled** (`final_benchmark/run_final.py:105` passes
>    `--disable-cpp-compact-cache`; upstream defaults it ON). Every tree-method round
>    paid a 72-op Python cache-compaction instead of one C++ call.
> 2. **Headline TPS included per-stage sync barriers** (upstream `dflash.py:157`
>    `cuda_time` = `torch.cuda.synchronize()` per stage, ~11-14/round). Upstream
>    itself times this way; the error was quoting headline TPS from it with no
>    barrier-free pass.
>
> Both taxes are per-round, so they fall hardest on low-acceptance methods (DDTree)
> and relatively flatter high-acceptance ones (SparklingTree) — the +17.8% aggregate
> does not survive their removal (re-measured 2026-08-05: acceptance reproduces to
> ~1-2%; wall-clock becomes mixed — ST wins gsm8k/chat, loses code/math).
>
> Citable numbers come from `experiment5-final-results/` (single GPU, compaction on,
> clean + instrumented passes — see its README "Benchmarking practices").

These files are kept verbatim as the historical record of what was run; do not
"fix" the launchers here.
