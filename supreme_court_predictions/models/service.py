"""
This file works as the central point of interaction for the models package.
"""

from supreme_court_predictions.models.logistic_regression import (
    LogisticRegression,
)
from supreme_court_predictions.models.random_forest import RandomForest
from supreme_court_predictions.models.xg_boost import XGBoost


def run_linear_regression():
    LogisticRegression().run()


def run_random_forest():
    RandomForest().run()


def run_xg_boost():
    XGBoost.run()
