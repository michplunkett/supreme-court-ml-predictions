"""
This file serves as the client for convokit.
"""
import requests
from convokit import Corpus, download

from supreme_court_predictions.util.contants import (
    ENCODING_UTF_8,
    FILE_MODE_WRITE,
)
from supreme_court_predictions.util.functions import (
    debug_print,
    get_full_data_pathway,
)


def get_data(debug_mode=False):
    """
    Loads and outputs the Supreme Court Corpus and case verdict data

    :param bool debug_mode: Indicates if the application requires debug print
        statements.
    """

    convokit_path = get_full_data_pathway("convokit/")

    debug_print("Loading Supreme Court Corpus Data...", debug_mode)
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
