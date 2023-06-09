"""
This file works as the central point of interaction for the processing package.
"""

from supreme_court_predictions.processing.datacleaner import DataCleaner
from supreme_court_predictions.processing.token_aggregation import (
    TokenAggregations,
)
from supreme_court_predictions.processing.tokenizer import Tokenizer


def clean_data(debug_mode=False):
    """
    Runner function for cleaning and formatting the Convokit corpus.

    :param bool debug_mode: Indicates if the application requires debug print
        statements.
    """
    DataCleaner(debug_mode).parse_all_data()


def tokenize_data(debug_mode=False):
    """
    Runner function for tokenizing data.

    :param bool debug_mode: Indicates if the application requires debug print
        statements.
    """
    Tokenizer(debug_mode)


def process_data(debug_mode=False):
    """
    Runner function for aggregating tokens.

    :param bool debug_mode: Indicates if the application requires debug print
        statements.
    """
    TokenAggregations(debug_mode).parse_all_data()
