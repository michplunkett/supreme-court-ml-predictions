"""
This file works as the central point of interaction for the models package.
"""

from supreme_court_predictions.models.logistic_regression import (
    LogisticRegression,
)
from supreme_court_predictions.models.random_forest import RandomForest
from supreme_court_predictions.models.xg_boost import XGBoost


def run_linear_regression(debug_mode=False, simulation=False):
    """
    Runner function for the Linear Regression model.

    :param bool debug_mode: Indicates if the application requires debug print
        statements.
    """
    LogisticRegression(debug_mode=debug_mode).run()
    if simulation:
        pass


def run_random_forest(debug_mode=False):
    """
    Runner function for the Random Forest model.

    :param bool debug_mode: Indicates if the application requires debug print
        statements.
    """
    RandomForest(debug_mode=debug_mode).run()


def run_xg_boost(debug_mode=False):
    """
    Runner function for the XG Boost model.

    :param bool debug_mode: Indicates if the application requires debug print
        statements.
    """
    XGBoost(debug_mode=debug_mode).run()
