"""Domain layer — data types, constants, pure scoring helpers.

Modules here may depend only on the stdlib and numpy. They must not
import sklearn, lightgbm, xgboost, or anything from ``features/``,
``models/``, ``training/``, or ``persistence/``. The dependency rule is
enforced by convention — any violation means the wrong thing is being
imported and the test suite will flag the cycle.
"""
