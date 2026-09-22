#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/openllama-7b-v2-ab/summary.json"
TEMPLATE = ROOT / "site/index.template.html"
SITEMAP = ROOT / "site/sitemap.xml"
OUT = ROOT / "_site"


def gib_from_kib(value: float) -> str:
    return f"{value / 1024 / 1024:.2f}"


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    html = TEMPLATE.read_text(encoding="utf-8")
    sitemap = SITEMAP.read_text(encoding="utf-8")

    values = {
        "{{MEMVANTA_RSS_GIB}}": gib_from_kib(data["memvanta_peak_rss_kib"]),
        "{{LLAMA_RSS_GIB}}": gib_from_kib(data["llama_peak_rss_kib"]),
        "{{RSS_REDUCTION}}": f"{data['memvanta_rss_reduction_pct']:.2f}",
        "{{MEMVANTA_PP}}": f"{data['memvanta_pp']['mean']:.2f}",
        "{{LLAMA_PP}}": f"{data['llama_pp']['mean']:.2f}",
        "{{MEMVANTA_TG}}": f"{data['memvanta_tg']['mean']:.2f}",
        "{{LLAMA_TG}}": f"{data['llama_tg']['mean']:.2f}",
        "{{LASTMOD}}": date.today().isoformat(),
    }

    for key, value in values.items():
        html = html.replace(key, value)
        sitemap = sitemap.replace(key, value)

    unresolved_html = [token for token in values if token in html]
    if unresolved_html:
        raise RuntimeError(f"Unresolved HTML template values: {unresolved_html}")
    if "{{LASTMOD}}" in sitemap:
        raise RuntimeError("Unresolved sitemap LASTMOD value")

    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir()
    (OUT / "index.html").write_text(html, encoding="utf-8")
    (OUT / "sitemap.xml").write_text(sitemap, encoding="utf-8")
    shutil.copy2(ROOT / "site" / "robots.txt", OUT / "robots.txt")
    (OUT / ".nojekyll").write_text("", encoding="utf-8")


if __name__ == "__main__":
    main()
