"""
This file works as the central point of interaction for the statistics package.
"""

from supreme_court_predictions.summary_analysis.descriptive_statistics import (
    DescriptiveStatistics,
)


def describe_data(debug_mode=False):
    """
    Runner function for calculating the descriptive statistics.

    :param bool debug_mode: Indicates if the application requires debug print
        statements.
    """
    DescriptiveStatistics(debug_mode).parse_all_data()
