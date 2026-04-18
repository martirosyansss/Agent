"""Sentinel ML package.

Subpackages:
* ``domain``      — pure data (configs, metrics, constants, scoring helpers).
                    Zero sklearn / numpy-beyond-ndarray / business-rule coupling.

Older code imports ML types from ``analyzer.ml_predictor``; that module
still re-exports everything for backwards compatibility, so no existing
caller needs to change when we move a class into here.
"""
