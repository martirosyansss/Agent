"""Package version snapshot — persisted with each model checkpoint.

Captured at save time and checked at load time so a model trained on
sklearn 1.2 can warn operators before it silently behaves differently
under sklearn 1.5 (tree-split ordering changes, isotonic implementation
changes, etc.). Missing packages are reported as "missing" rather than
raising — the model may not need them.
"""
from __future__ import annotations


def capture_package_versions() -> dict[str, str]:
    """Return ``{pkg: version}`` for the ML stack we persist alongside models."""
    versions: dict[str, str] = {}
    for pkg_name in ("sklearn", "lightgbm", "xgboost", "numpy"):
        try:
            pkg = __import__(pkg_name)
            versions[pkg_name] = getattr(pkg, "__version__", "unknown")
        except ImportError:
            versions[pkg_name] = "missing"
    return versions
