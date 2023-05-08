"""
This file works as the central point of interaction for the processing package.
"""

from supreme_court_predictions.processing.token_aggregation import (
    TokenAggregations,
)
from supreme_court_predictions.processing.tokenizer import Tokenizer


def tokenize_data():
    Tokenizer()


def process_data():
    TokenAggregations().parse_all_data()
