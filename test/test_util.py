"""
This file contains tests for the util folder.
"""
import os

from supreme_court_predictions.util.files import get_full_data_pathway


def test_get_full_pathway():
    """
    Tests the get_full_pathway function.
    """
    pathway = (
        os.getcwd().replace("\\", "/") + "/supreme_court_predictions/data/"
    )
    test_dir = "/lol/test"
    result = get_full_data_pathway(test_dir)

    assert pathway + test_dir == result
