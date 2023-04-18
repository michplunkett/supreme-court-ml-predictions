"""
This file works as the central point of interaction for the statistics package.
"""

from supreme_court_predictions.statistics.datacleaner import DataCleaner
from supreme_court_predictions.statistics.descriptives import Descriptives


def main():
    DataCleaner(downloaded_corpus=True, save_data=True).parse_all_data()
    Descriptives(downloaded_clean_corpus=True, save_data=True).parse_all_data()
