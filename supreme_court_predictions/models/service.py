"""
This file works as the central point of interaction for the models package.
"""

from supreme_court_predictions.models.logistic_regression import (
    LogisticRegression,
)

from supreme_court_predictions.models.random_forest import (
    RandomForest,
)


def run_linear_regression():
    LogisticRegression().run()


def run_random_forest():
    RandomForest().run()
