"""Persistence layer — pickle save/load + version registry.

``codec.py`` owns the restricted unpickler, the v3_signed envelope,
checksum validation, and the atomic-write save path. ``registry.py``
maintains a rolling JSON audit log of saved model versions. Both are
framework-free: no sklearn / lightgbm / xgboost imports here.
"""
