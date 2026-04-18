"""Feature extraction — StrategyTrade → numeric feature vector.

Depends on the ``domain`` layer (constants, config) and on
``core.models.StrategyTrade``. Never imports sklearn / xgboost / lightgbm.
The resulting ``X`` matrix has shape ``(n_trades, N_FEATURES)`` — exact
column order is defined by ``FEATURE_NAMES`` in ``domain.constants``.
"""
