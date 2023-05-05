"""
This file works as the central point of interaction for the statistics package.
"""

from supreme_court_predictions.models.logistic import LogisiticRegression
from supreme_court_predictions.processing.datacleaner import DataCleaner
from supreme_court_predictions.processing.token_aggregation import (
    TokenAggregations,
)
from supreme_court_predictions.processing.tokenizer import Tokenizer
from supreme_court_predictions.statistics.descriptives import Descriptives


def clean_data():
    DataCleaner(downloaded_corpus=True, save_data=True).parse_all_data()


def describe_data():
    Descriptives(downloaded_clean_corpus=True, save_data=True).parse_all_data()


def tokenize_data():
    Tokenizer()


def process_data():
    TokenAggregations(save_data=True).parse_all_data()


def logistic_regression():
    LogisiticRegression()
