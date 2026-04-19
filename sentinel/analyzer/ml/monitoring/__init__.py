"""Runtime ML monitoring: feature drift, prediction drift, calibration drift.

These detectors complement ``LivePerformanceTracker`` (which watches model
*outputs* — precision/win-rate) by watching model *inputs* (feature
distributions). Together they catch both kinds of failure: the model
seeing the same world but acting differently (output drift), and the
world changing while the model still acts the same (input drift).
"""
from analyzer.ml.monitoring.feature_drift import (
    FeatureDriftMonitor,
    FeatureDriftReport,
    population_stability_index,
)

__all__ = [
    "FeatureDriftMonitor",
    "FeatureDriftReport",
    "population_stability_index",
]
