"""
A module implementing a Tokenizer class that tokenizes text. 
The Tokenizer class provides methods for initializing the required 
components and processing text data. It is designed to handle tokenization
tasks efficiently and effectively in a user-friendly manner.
"""

import os

import pandas as pd
import spacy


class Tokenizer:
    """
    Tokenizer class that uses the spaCy library for
    tokenizing / lemmatizing text. This class initializes and provides
    methods for tokenizing text.
    """

    def __init__(self):
        """
        Initializes the Tokenizer class by setting up the local path
        and loading the spaCy model.
        """
        # Get local directory
        cwd = os.getcwd()
        self.local_path = (
            cwd.replace("\\", "/")
            + "/supreme_court_predictions/data/clean_convokit/"
        )
        print(f"Data will be saved to: \n{self.local_path}")

        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except OSError:
            print("Spacy not present. Downloading files.")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        print("Spacy module successfully loaded.")

        self.tokenize()

    def spacy_apply(self, text):
        """
        Applies the spaCy tokenizer on the input text
        and returns a list of tokens.

        :param text: Input text to tokenize.
        :return: List of tokenized words.
        """
        doc = self.nlp(text)
        return [token.text for token in doc]

    def tokenize(self):
        """
        Tokenizes the text in the utterances DataFrame
        and saves the result as a new CSV file.
        """
        utterances_df = pd.read_csv(self.local_path + "utterances_df.csv")

        utterances_df["tokens"] = utterances_df.loc[:, "text"].apply(
            self.spacy_apply
        )
        utterances_df.to_csv(self.local_path + "utterances_df.csv", index=False)
        print("Spacy Tokenization Complete")
