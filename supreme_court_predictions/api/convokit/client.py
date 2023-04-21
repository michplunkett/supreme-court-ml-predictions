"""
This file serves as the client for convokit.
"""
import requests
from convokit import Corpus, download

from supreme_court_predictions.util.contants import ENCODING_UTF_8

# Constants
DOWNLOAD_BASE_PATH = "./supreme_court_predictions/data/convokit/"


def get_data():
    """
    Loads and outputs the Supreme Court Corpus and case verdict data
    """

    print("Loading Supreme Court Corpus Data...")
    corpus = Corpus(filename=download("supreme-corpus"))
    corpus.dump("supreme_corpus", base_path=DOWNLOAD_BASE_PATH)

    r = requests.get(
        url="https://zissou.infosci.cornell.edu/convokit"
        "/datasets/supreme-corpus/cases.jsonl",
        allow_redirects=True,
    )

    with open(f"{DOWNLOAD_BASE_PATH}cases.jsonl", "w") as outfile:
        outfile.write(r.content.decode(ENCODING_UTF_8))
