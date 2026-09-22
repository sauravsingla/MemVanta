#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
SUMMARY = ROOT / "results/openllama-7b-v2-ab/summary.json"
SITEMAP = SITE / "sitemap.xml"
OUT = ROOT / "_site"


def gib_from_kib(value: float) -> str:
    return f"{value / 1024 / 1024:.2f}"


def render(text: str, values: dict[str, str], source: Path) -> str:
    for key, value in values.items():
        text = text.replace(key, value)
    unresolved = [token for token in values if token in text]
    if unresolved:
        raise RuntimeError(f"Unresolved template values in {source}: {unresolved}")
    return text


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
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

    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir()

    templates = sorted(SITE.rglob("*.template.html"))
    if not templates:
        raise RuntimeError("No GitHub Pages templates found")

    for template in templates:
        relative = template.relative_to(SITE)
        output_relative = relative.with_name(relative.name.replace(".template.html", ".html"))
        output_path = OUT / output_relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        html = render(template.read_text(encoding="utf-8"), values, template)
        output_path.write_text(html, encoding="utf-8")

    sitemap = render(SITEMAP.read_text(encoding="utf-8"), values, SITEMAP)
    (OUT / "sitemap.xml").write_text(sitemap, encoding="utf-8")

    for asset in ("robots.txt", "styles.css"):
        shutil.copy2(SITE / asset, OUT / asset)

    (OUT / ".nojekyll").write_text("", encoding="utf-8")


if __name__ == "__main__":
    main()
