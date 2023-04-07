"""
This file serves as the client for convokit.
"""
from convokit import Corpus, download


def get_data():
    """
    Loads and outputs the Supreme Court Corpus data
    """

    print("Loading Supreme Court Corpus Data...")
    corpus = Corpus(filename=download("supreme-corpus"))
    corpus.dump(
        "supreme_corpus", base_path="./supreme_court_predictions/data/convokit/"
    )
