#!/usr/bin/env python3
"""Synchronize the README headline benchmark from the canonical 7B summary.json.

The benchmark workflow owns results/openllama-7b-v2-ab/summary.json.  README is a
rendered view of that evidence, not a second place where benchmark numbers are
maintained by hand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
SUMMARY = ROOT / "results" / "openllama-7b-v2-ab" / "summary.json"
START = "<!-- BEGIN_CANONICAL_7B_BENCHMARK -->"
END = "<!-- END_CANONICAL_7B_BENCHMARK -->"


def gib_from_kib(value: int) -> float:
    return value / (1024.0 * 1024.0)


def render(summary: dict) -> str:
    mv_pp = summary["memvanta_pp"]
    lc_pp = summary["llama_pp"]
    mv_tg = summary["memvanta_tg"]
    lc_tg = summary["llama_tg"]
    mv_rss = gib_from_kib(summary["memvanta_peak_rss_kib"])
    lc_rss = gib_from_kib(summary["llama_peak_rss_kib"])
    reduction = summary["memvanta_rss_reduction_pct"]

    return f"""{START}
| Metric | MemVanta | pinned `llama.cpp` |
|---|---:|---:|
| OpenLLaMA 7B v2 Q4_0 peak RSS | **{mv_rss:.2f} GiB** | {lc_rss:.2f} GiB |
| Prompt processing | {mv_pp['mean']:.2f} ± {mv_pp['sd']:.2f} tok/s | **{lc_pp['mean']:.2f} ± {lc_pp['sd']:.2f} tok/s** |
| Token generation | {mv_tg['mean']:.2f} ± {mv_tg['sd']:.2f} tok/s | **{lc_tg['mean']:.2f} ± {lc_tg['sd']:.2f} tok/s** |
| Peak-RSS reduction | **{reduction:.2f}%** | baseline |

Source of truth: [`results/openllama-7b-v2-ab/summary.json`](results/openllama-7b-v2-ab/summary.json). The README table is generated from that file; do not edit its numbers by hand.
{END}"""


def synchronized_text(readme: str, block: str) -> str:
    start = readme.find(START)
    end = readme.find(END)
    if start < 0 or end < 0 or end < start:
        raise RuntimeError(
            f"README must contain exactly one {START} ... {END} generated block"
        )
    end += len(END)
    if readme.find(START, start + len(START)) >= 0 or readme.find(END, end) >= 0:
        raise RuntimeError("README contains duplicate canonical benchmark markers")
    return readme[:start] + block + readme[end:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if README is not synchronized; do not modify files",
    )
    args = parser.parse_args()

    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    current = README.read_text(encoding="utf-8")
    expected = synchronized_text(current, render(summary))

    if current == expected:
        print("README benchmark is synchronized with", SUMMARY.relative_to(ROOT))
        return 0

    if args.check:
        print(
            "README benchmark is stale. Run: python3 scripts/sync_headline_benchmark.py",
            file=sys.stderr,
        )
        return 1

    README.write_text(expected, encoding="utf-8")
    print("Updated", README.relative_to(ROOT), "from", SUMMARY.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
