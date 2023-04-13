"""
This file works as the central point of interaction for the statistics package.
"""

from supreme_court_predictions.statistics.datacleaner import DataCleaner


def main():
    DataCleaner(downloaded_corpus=True, save_data=True).parse_all_data()
