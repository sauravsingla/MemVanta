"""Python packaging shim for the MemVanta native CLI runtime."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("memvanta")
except PackageNotFoundError:  # source tree / editable tooling
    __version__ = "0+unknown"

__all__ = ["__version__"]
