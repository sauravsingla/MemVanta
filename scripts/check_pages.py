#!/usr/bin/env python3
from __future__ import annotations

import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree

BASE = "https://sauravsingla.github.io/MemVanta/"


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_title = False
        self.in_h1 = False
        self.title_parts: list[str] = []
        self.h1_count = 0
        self.description: str | None = None
        self.canonical: str | None = None
        self.robots: str | None = None
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "title":
            self.in_title = True
        elif tag == "h1":
            self.in_h1 = True
            self.h1_count += 1
        elif tag == "meta" and values.get("name") == "description":
            self.description = values.get("content")
        elif tag == "meta" and values.get("name") == "robots":
            self.robots = values.get("content")
        elif tag == "link" and values.get("rel") == "canonical":
            self.canonical = values.get("href")
        elif tag == "a" and values.get("href"):
            self.hrefs.append(values["href"] or "")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self.in_title = False
        elif tag == "h1":
            self.in_h1 = False

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title_parts.append(data)


def local_path_for_url(out: Path, url: str) -> Path | None:
    parsed = urlparse(url)
    base = urlparse(BASE)
    if parsed.netloc and parsed.netloc != base.netloc:
        return None
    path = parsed.path
    base_path = base.path
    if not path.startswith(base_path):
        return None
    relative = path[len(base_path):].lstrip("/")
    if not relative:
        return out / "index.html"
    candidate = out / relative
    if path.endswith("/"):
        candidate = candidate / "index.html"
    return candidate


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "_site")
    errors: list[str] = []

    required = ["index.html", "sitemap.xml", "robots.txt", "styles.css", "favicon.svg", "llms.txt"]
    for name in required:
        if not (out / name).exists():
            errors.append(f"missing required site asset: {name}")

    sitemap_path = out / "sitemap.xml"
    if not sitemap_path.exists():
        raise SystemExit("sitemap.xml is required before page validation")

    root = ElementTree.parse(sitemap_path).getroot()
    ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = [node.text.strip() for node in root.findall("s:url/s:loc", ns) if node.text]
    if len(urls) != len(set(urls)):
        errors.append("sitemap contains duplicate URLs")
    if not urls:
        errors.append("sitemap contains no URLs")

    canonical_seen: set[str] = set()
    for url in urls:
        page_path = local_path_for_url(out, url)
        if page_path is None or not page_path.exists():
            errors.append(f"sitemap URL has no generated page: {url}")
            continue

        text = page_path.read_text(encoding="utf-8")
        if "{{" in text or "}}" in text:
            errors.append(f"unresolved template token in {page_path}")

        parser = PageParser()
        parser.feed(text)
        title = "".join(parser.title_parts).strip()
        if not title:
            errors.append(f"missing title: {page_path}")
        if not parser.description:
            errors.append(f"missing meta description: {page_path}")
        if parser.h1_count != 1:
            errors.append(f"expected exactly one H1 in {page_path}, found {parser.h1_count}")
        if parser.canonical != url:
            errors.append(f"canonical mismatch in {page_path}: {parser.canonical!r} != {url!r}")
        if parser.canonical in canonical_seen:
            errors.append(f"duplicate canonical URL: {parser.canonical}")
        if parser.canonical:
            canonical_seen.add(parser.canonical)
        if not parser.robots or "noindex" in parser.robots.lower():
            errors.append(f"indexable sitemap page has invalid robots directive: {page_path}")

        for href in parser.hrefs:
            if href.startswith(("#", "mailto:", "javascript:")):
                continue
            resolved = urljoin(url, href)
            target = local_path_for_url(out, resolved)
            if target is not None and not target.exists():
                errors.append(f"broken internal link in {page_path}: {href} -> {target}")

    llms = out / "llms.txt"
    if llms.exists() and "{{" in llms.read_text(encoding="utf-8"):
        errors.append("unresolved template token in llms.txt")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    print(f"Validated {len(urls)} indexed pages and required discovery assets.")


if __name__ == "__main__":
    main()
