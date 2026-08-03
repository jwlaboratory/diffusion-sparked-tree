"""Turn the concurrency benchmark JSON into REPORT.md.

    python3 concurrency/report.py CONCURRENCY_H100_<ts>.json --out concurrency/REPORT.md

Two tables carry the result. The first is the raw ladder. The second is the one
that matters: the overlap ratio per method, which prices what a host-resident
tree builder forfeits by being unable to run `supports_overlap=True`.

Blocked arms are printed, not omitted. A four-way table showing two arms reads
as a completed comparison; it is not one.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg  # noqa: E402


def ladder(results, arms, concurrency, metric, fmt="{:.1f}"):
    lines = ["| arm | " + " | ".join(f"c={c}" for c in concurrency) + " |",
             "|---" * (len(concurrency) + 1) + "|"]
    for arm in arms:
        rows = results.get(arm)
        if not rows:
            lines.append(f"| {arm} | " + " | ".join("—" for _ in concurrency) + " |")
            continue
        cells = []
        for c in concurrency:
            value = (rows.get(str(c)) or rows.get(c) or {}).get(metric)
            cells.append(fmt.format(value) if value is not None else "—")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    return lines


def deltas(results, concurrency):
    lines = ["| ratio | " + " | ".join(f"c={c}" for c in concurrency) + " |",
             "|---" * (len(concurrency) + 1) + "|"]
    for label, numerator, denominator in cfg.DELTAS:
        num_rows, den_rows = results.get(numerator), results.get(denominator)
        if not num_rows or not den_rows:
            lines.append(f"| {label} | " + " | ".join("—" for _ in concurrency) + " |")
            continue
        cells = []
        for c in concurrency:
            a = (num_rows.get(str(c)) or num_rows.get(c) or {}).get("output_throughput")
            b = (den_rows.get(str(c)) or den_rows.get(c) or {}).get("output_throughput")
            cells.append(f"{a / b:.3f}x" if a and b else "—")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path")
    parser.add_argument("--out", default="concurrency/REPORT.md")
    args = parser.parse_args()

    with open(args.json_path) as handle:
        payload = json.load(handle)

    results = payload["results"]
    concurrency = payload["concurrency"]
    arms = list(payload["arms"])

    out = [
        "# Concurrency benchmark",
        "",
        f"`{payload['gpu']}` · `{payload['sglang_image']}` · target `{payload['target']}` "
        f"· dataset `{payload['dataset']}`",
        "",
        "## Output throughput (tok/s)",
        "",
        *ladder(results, arms, concurrency, "output_throughput"),
        "",
        "## Mean TPOT (ms)",
        "",
        *ladder(results, arms, concurrency, "mean_tpot_ms", "{:.2f}"),
        "",
        "## Acceptance length",
        "",
        *ladder(results, arms, concurrency, "accept_length", "{:.3f}"),
        "",
        "## What overlap is worth",
        "",
        "Ratio of output throughput. The chain rows are the price a host-resident",
        "tree builder pays for being unable to register `supports_overlap=True`.",
        "",
        *deltas(results, concurrency),
        "",
        "## Not measured",
        "",
        "| arm | blocked by |",
        "|---|---|",
    ]
    for arm, reason in payload.get("blocked_arms", {}).items():
        out.append(f"| {arm} | {reason} |")
    out += [
        "",
        "Both tree arms are absent, so this run does **not** compare trees against",
        "chains at concurrency. What it establishes is the handicap those arms will",
        "carry when they land, and the chain baseline they must beat.",
        "",
    ]

    Path(args.out).write_text("\n".join(out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
