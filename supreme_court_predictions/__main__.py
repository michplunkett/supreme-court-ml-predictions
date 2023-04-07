"""
This file serves as the entry point for the supreme_court_predictions module.
"""

import argparse

from supreme_court_predictions.api.convokit.client import get_data

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

    args = parser.parse_args()

    if args.get_data:
        get_data()
