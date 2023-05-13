"""
This file holds general utility functions.
"""

import os


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
