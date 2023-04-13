"""
This file works as the central point of interaction for the statistics package.
"""

from supreme_court_predictions.statistics.DataCleaner import DataCleaner


def main():
    DataCleaner(downloaded_corpus=True, save_data=True)
