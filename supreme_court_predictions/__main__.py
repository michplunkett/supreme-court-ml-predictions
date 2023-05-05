"""
This file serves as the entry point for the supreme_court_predictions module.
"""

import argparse

from supreme_court_predictions.api.convokit import get_data
from supreme_court_predictions.statistics.service import (
    clean_data,
    describe_data,
    tokenize_data,
)

if __name__ == "__main__":
    print("oh, what up?")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--get-data",
        help="Get data from Convokit",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "--clean-data",
        help="Clean the data from Convokit",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "--describe-data",
        help="Generate descriptive statistics for Convokit",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )

    args = parser.parse_args()

    if args.get_data:
        get_data()

    if args.clean_data:
        clean_data()
        tokenize_data()

    if args.describe_data:
        describe_data()
