"""
This file works as the central point of interaction for the statistics package.
"""

from supreme_court_predictions.processing.datacleaner import DataCleaner
from supreme_court_predictions.statistics.descriptives import Descriptives


def clean_data(debug_mode=False):
    """
    Runner function for cleaning and formatting the Convokit corpus.

    :param bool debug_mode: Indicates if the application requires debug print
        statements.
    """
    DataCleaner(debug_mode).parse_all_data()


def describe_data(debug_mode=False):
    """
    Runner function for calculating the descriptive statistics.

    :param bool debug_mode: Indicates if the application requires debug print
        statements.
    """
    Descriptives(debug_mode).parse_all_data()
