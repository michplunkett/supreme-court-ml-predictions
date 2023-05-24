"""
This file serves as the entry point for the supreme_court_predictions module.
"""

import argparse

from supreme_court_predictions.api.convokit import get_data
from supreme_court_predictions.models.service import (
    run_linear_regression,
    run_random_forest,
    run_simulation,
    run_xg_boost,
)
from supreme_court_predictions.processing.service import (
    clean_data,
    process_data,
    tokenize_data,
)
from supreme_court_predictions.summary_analysis.service import describe_data

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
        help="Run the Logistic Regression model",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "--random-forest",
        help="Run the Random Forest model",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "--xg-boost",
        help="Run the XG Boost model",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "--simulate",
        help="Simulate all three models for \
            each judge and reach majority decision.",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )

    args = parser.parse_args()

    # Debug mode will only trigger when we call functionality from the command
    # line, which means we shouldn't get an obscene amount of printouts
    # in our Jupyter notebooks.
    debug_mode = any(
        [
            args.get_data,
            args.clean_data,
            args.tokenize_data,
            args.describe_data,
            args.process_data,
            args.logistic_regression,
            args.random_forest,
            args.xg_boost,
        ]
    )

    if args.get_data:
        get_data(debug_mode)

    if args.clean_data:
        clean_data(debug_mode)

    if args.tokenize_data:
        tokenize_data(debug_mode)

    if args.describe_data:
        describe_data(debug_mode)

    if args.process_data:
        process_data(debug_mode)

    if args.logistic_regression:
        run_linear_regression(debug_mode)

    if args.random_forest:
        run_random_forest(debug_mode)

    if args.xg_boost:
        run_xg_boost(debug_mode)

    if args.simulate:
        run_simulation(debug_mode)
