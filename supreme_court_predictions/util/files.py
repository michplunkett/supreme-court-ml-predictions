"""
This file holds functions used in multiple places that are relevant to files in
a general sense.
"""

import os


def get_full_pathway(desired_folder):
    """
    This function gives you the full pathway for the desired_folder parameter.

    :param str desired_folder: The folder you are accessing.
    :return: The full pathway for the desired_folder value.
    :rtype: str
    """

    return os.getcwd().replace("\\", "/") + desired_folder
