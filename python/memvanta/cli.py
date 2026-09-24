"""Console entry points for the native MemVanta binaries bundled in the wheel."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _exec(binary: str) -> None:
    executable = Path(__file__).resolve().parent / "bin" / binary
    if not executable.is_file():
        raise SystemExit(
            f"MemVanta native executable is missing from this installation: {executable}"
        )
    os.execv(str(executable), [str(executable), *sys.argv[1:]])


def main() -> None:
    _exec("memvanta")


def real() -> None:
    _exec("memvanta_real")


def tokenize() -> None:
    _exec("memvanta_tokenize")


def gguf_inspect() -> None:
    _exec("memvanta_gguf_inspect")
