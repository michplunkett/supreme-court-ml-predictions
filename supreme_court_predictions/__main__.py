"""
This file serves as the entry point for the supreme_court_predictions module.
"""

import argparse

from supreme_court_predictions.api.convokit import get_data
from supreme_court_predictions.models.service import (
    run_linear_regression,
    run_random_forest,
)
from supreme_court_predictions.processing.service import (
    process_data,
    tokenize_data,
)
from supreme_court_predictions.statistics.service import (
    clean_data,
    describe_data,
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
        "--tokenize-data",
        help="Tokenize the data from clean_convokit",
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

    parser.add_argument(
        "--process-data",
        help="Generate aggregate tokenizations for Convokit",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "--logistic-regression",
        help="Logistic Regression",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "--random-forest",
        help="Random Forest",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )

    args = parser.parse_args()

    if args.get_data:
        get_data()

    if args.clean_data:
        clean_data()

    if args.tokenize_data:
        tokenize_data()

    if args.describe_data:
        describe_data()

    if args.process_data:
        process_data()

    if args.logistic_regression:
        run_linear_regression()

    if args.random_forest:
        run_random_forest()
