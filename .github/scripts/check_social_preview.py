#!/usr/bin/env python3
from __future__ import annotations

import struct
import sys
from html.parser import HTMLParser
from pathlib import Path

EXPECTED_IMAGE = "https://sauravsingla.github.io/MemVanta/social-card.png"
EXPECTED_WIDTH = "1200"
EXPECTED_HEIGHT = "630"


class SocialParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.meta: dict[tuple[str, str], str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "meta":
            return
        values = dict(attrs)
        content = values.get("content")
        if content is None:
            return
        if values.get("property"):
            self.meta[("property", values["property"] or "")] = content
        if values.get("name"):
            self.meta[("name", values["name"] or "")] = content


def check_png(path: Path, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"missing social preview image: {path}")
        return
    data = path.read_bytes()
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        errors.append(f"social preview is not a valid PNG: {path}")
        return
    width, height = struct.unpack(">II", data[16:24])
    if (width, height) != (1200, 630):
        errors.append(f"social preview must be 1200x630, found {width}x{height}")


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "_site")
    errors: list[str] = []
    check_png(out / "social-card.png", errors)

    pages = sorted(out.rglob("index.html"))
    if not pages:
        errors.append("no generated HTML pages found")

    for page in pages:
        parser = SocialParser()
        parser.feed(page.read_text(encoding="utf-8"))
        meta = parser.meta

        required = {
            ("property", "og:image"): EXPECTED_IMAGE,
            ("property", "og:image:type"): "image/png",
            ("property", "og:image:width"): EXPECTED_WIDTH,
            ("property", "og:image:height"): EXPECTED_HEIGHT,
            ("name", "twitter:card"): "summary_large_image",
            ("name", "twitter:image"): EXPECTED_IMAGE,
        }
        for key, expected in required.items():
            actual = meta.get(key)
            if actual != expected:
                errors.append(f"{page}: {key[1]} is {actual!r}, expected {expected!r}")

        for key in (("property", "og:image:alt"), ("name", "twitter:image:alt")):
            if not meta.get(key, "").strip():
                errors.append(f"{page}: missing {key[1]}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    print(f"Validated large social previews on {len(pages)} generated pages.")


if __name__ == "__main__":
    main()
