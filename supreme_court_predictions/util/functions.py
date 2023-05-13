"""
This file holds general utility functions.
"""

import os


def debug_print(message, debug_mode=True):
    """
    TODO: Add to `processing` and `statistics` packages.
    TODO: Eventually turn into a Singleton.

    This function uses `print` if the application is in debug mode.

    :param str message: The desired message to print.
    :param bool debug_mode: If the application is in debug mode.
    """
    if debug_mode:
        print(message)


def get_full_data_pathway(desired_folder):
    """
    This function gives you the full pathway your desired data sub-folder.

    :param str desired_folder: The folder you are accessing.
    :return: The full pathway for the desired_folder value.
    :rtype: str
    """

    return (
        os.getcwd().replace("\\", "/")
        + "/supreme_court_predictions/data/"
        + desired_folder
    )
