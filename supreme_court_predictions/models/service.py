"""
This file works as the central point of interaction for the models package.
"""

from supreme_court_predictions.models.logistic_regression import (
    LogisticRegression,
)


def run_linear_regression():
    LogisticRegression().run()
