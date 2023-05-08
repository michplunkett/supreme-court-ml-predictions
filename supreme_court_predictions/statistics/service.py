"""
This file works as the central point of interaction for the statistics package.
"""

from supreme_court_predictions.processing.datacleaner import DataCleaner
from supreme_court_predictions.statistics.descriptives import Descriptives


def clean_data():
    DataCleaner().parse_all_data()


def describe_data():
    Descriptives().parse_all_data()
