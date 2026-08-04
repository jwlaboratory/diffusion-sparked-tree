"""Rebuild summary.json from checkpointed raw units -- no GPU, no re-run.

The benchmark writes one cache file per (tree_budget, dataset) unit containing the
raw per-round acceptance lengths. This turns any set of those files back into the
summary the charts read, so you can:

  * chart a PARTIAL run while the rest is still on the GPU,
  * recover a summary if a run died after doing the work but before writing one,
  * re-derive every number without trusting a stale summary.json.

Filenames encode the unit: `b{budget}__{dataset}__n{samples}.json`.

Usage:
    # pull the units down from the Modal volume first
    modal volume get --force ddtree-results /block16/cache results/cache

    python build_summary.py                          # ./cache -> ./summary.json
    python build_summary.py --cache-dir DIR --out F
    python build_summary.py --depth-limit 16

Then: python make_charts.py summary.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "DDTree"))
import aggregate  # noqa: E402  the single implementation of the rollup

UNIT_RE = re.compile(r"^b(?P<budget>\d+)__(?P<dataset>.+)__n(?P<n>\d+)$")


def load_units(cache_dir: Path) -> tuple[dict, dict]:
    """Return ({(budget, dataset): {method: [lengths]}}, {dataset: n_samples})."""
    units, samples = {}, {}
    for path in sorted(cache_dir.rglob("*.json")):
        m = UNIT_RE.match(path.stem)
        if not m:
            continue  # e.g. a stray summary.json living in the same tree
        budget = int(m.group("budget"))
        dataset = m.group("dataset")
        units[(budget, dataset)] = json.load(open(path))
        samples[dataset] = int(m.group("n"))
    return units, samples


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=Path, default=here / "cache")
    p.add_argument("--out", type=Path, default=here / "summary.json")
    p.add_argument("--depth-limit", type=int, default=16)
    a = p.parse_args()

    if not a.cache_dir.exists():
        raise SystemExit(
            f"no cache at {a.cache_dir}\n"
            "pull it first:  modal volume get --force ddtree-results /block16/cache results/cache"
        )

    units, samples = load_units(a.cache_dir)
    if not units:
        raise SystemExit(f"no unit files (b*__*__n*.json) under {a.cache_dir}")

    budgets = sorted({b for b, _ in units})
    datasets = sorted({d for _, d in units})
    print(f"loaded {len(units)} unit(s): budgets {budgets}, datasets {datasets}")
    # Say plainly when the matrix is partial -- a summary built from 4 of 6 units
    # is legitimate, but it should never be mistaken for the finished run.
    missing = [(b, d) for b in budgets for d in datasets if (b, d) not in units]
    if missing:
        print(f"PARTIAL: missing {len(missing)} unit(s): "
              + ", ".join(f"b{b}/{d}" for b, d in missing))

    summary = aggregate.build_summary(
        units,
        depth_limit=a.depth_limit,
        config_extra={
            "tasks": [[d, samples[d]] for d in datasets],
            "partial": bool(missing),
            "missing_units": [f"b{b}__{d}" for b, d in missing],
        },
    )
    a.out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {a.out}")
    aggregate.print_rollups(summary)


if __name__ == "__main__":
    main()
