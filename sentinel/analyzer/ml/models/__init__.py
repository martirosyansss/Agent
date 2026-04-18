"""Model builders — sklearn / lightgbm / xgboost estimator factories.

The only module in the ML package that imports sklearn-family libraries
directly. Everything else receives already-constructed estimators via
these factories, which makes the training path trivially testable with
``DummyClassifier`` substitutes.
"""
