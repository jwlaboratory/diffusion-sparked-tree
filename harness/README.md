# Shared harness

The single copy of the DDTree decode core + a merged experiment driver. Built for
experiment 3; exp1/exp2 still run from their own frozen `DDTree/` copies until the
migration in [`PLAN.md`](PLAN.md) completes (steps 4+ pending).

```
ddtree/    decode core (was experiment*/DDTree, byte-identical copies unified)
           + timing.py (toggleable stage timers: clean vs instrumented passes)
           + the exp3 instrumentation: commit split into walk_accept / kv_update /
             state_carry, dflash argmax split out as candidate_build, round 0
             excluded from stages and reported as cold_round_time
runner/    backbones.py  loaders for dflash+dspark, RoPE + block-size guards
           methods.py    {backbone, corrector, verify} -> callable; verify="ddtree" added
           metrics.py    native stages -> canonical exp3 phases, aggregation, rollup
           driver.py     two-pass run loop, fingerprinted unit cache, summary assembly
```

Modules use flat imports (`from model import ...`); put both `harness/ddtree` and
`harness/runner` on `sys.path` (the Modal launchers do). Consumer:
`experiment3-timings/modal_benchmark.py`.

Differences from the frozen exp1/exp2 generators (timing semantics only — the
accepted-token math is untouched):
- `stage_times` keys changed: `commit` → `walk_accept`/`kv_update`/`state_carry`;
  dflash gained `candidate_build`.
- Round 0 is excluded from **all** stage buckets (was: only `draft`) and returned
  as `cold_round_time`; the decode window includes it.
- Results gained `total_decode_time` and `cold_round_time`.
- TTFT / decode totals use `sync_time()` (always-barrier) so they stay honest when
  stage timers are off.
