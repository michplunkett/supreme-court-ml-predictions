"""
This file serves as the client for convokit.
"""
import requests
from convokit import Corpus, download

from supreme_court_predictions.util.contants import (
    ENCODING_UTF_8,
    FILE_MODE_WRITE,
)
from supreme_court_predictions.util.functions import get_full_data_pathway


def get_data(debug=True):
    """
    Loads and outputs the Supreme Court Corpus and case verdict data
    """

    convokit_path = get_full_data_pathway("convokit/")

    if debug:
        print("Loading Supreme Court Corpus Data...")
    corpus = Corpus(filename=download("supreme-corpus"))
    corpus.dump("supreme_corpus", base_path=convokit_path)

    r = requests.get(
        url="https://zissou.infosci.cornell.edu/convokit"
        "/datasets/supreme-corpus/cases.jsonl",
        allow_redirects=True,
    )

    with open(
        f"{convokit_path}supreme_corpus/cases.jsonl", mode=FILE_MODE_WRITE
    ) as outfile:
        outfile.write(r.content.decode(ENCODING_UTF_8))
