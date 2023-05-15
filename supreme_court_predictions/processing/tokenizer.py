"""
A module implementing a Tokenizer class that tokenizes text.
The Tokenizer class provides methods for initializing the required
components and processing text data. It is designed to handle tokenization
tasks efficiently and effectively in a user-friendly manner.
"""

import pandas as pd
import spacy

from supreme_court_predictions.util.functions import (
    debug_print,
    get_full_data_pathway,
)


class Tokenizer:
    """
    Tokenizer class that uses the spaCy library for
    tokenizing / lemmatizing text. This class initializes and provides
    methods for tokenizing text.
    """

    SPACY_PACKAGE = "en_core_web_sm"

    def __init__(self, debug_mode=False):
        """
        Initializes the Tokenizer class by setting up the local path
        and loading the spaCy model.
        """
        self.debug_mode = debug_mode
        self.local_path = get_full_data_pathway("clean_convokit/")
        debug_print(
            f"Data will be saved to: \n{self.local_path}", self.debug_mode
        )

        try:
            self.nlp = spacy.load(self.SPACY_PACKAGE, disable=["parser", "ner"])
        except OSError:
            debug_print(
                "spaCy not present. Downloading package.", self.debug_mode
            )
            spacy.cli.download(self.SPACY_PACKAGE)
            self.nlp = spacy.load(self.SPACY_PACKAGE, disable=["parser", "ner"])
        debug_print("spaCy module successfully loaded.", self.debug_mode)

        self.tokenize()

    def spacy_apply(self, text):
        """
        Applies the spaCy tokenizer on the input text
        and returns a list of tokens.

        :param text: Input text to tokenize.
        :return: List of tokenized words.
        """
        doc = self.nlp(text)
        return [
            token.lemma_
            for token in doc
            if token.is_alpha and not token.is_stop
        ]

    def tokenize(self):
        """
        Tokenizes the text in the utterances DataFrame
        and saves the result as a new CSV file.
        """
        utterances_df = pd.read_csv(self.local_path + "utterances_df.csv")

        utterances_df["tokens"] = (
            utterances_df.loc[:, "text"].astype(str).apply(self.spacy_apply)
        )
        utterances_df.to_csv(self.local_path + "utterances_df.csv", index=False)
        utterances_df.to_pickle(self.local_path + "utterances_df.p")
        debug_print("Spacy tokenization complete.", self.debug_mode)
