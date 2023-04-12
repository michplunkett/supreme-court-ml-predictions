### Imports
import os
import json
import pandas as pd

# Convokit import
from convokit import Corpus, download


class DataCleaner:
    def __init__(self, download_corpus):
        self.download_corpus = download_corpus

        # Get local directory
        cwd = os.getcwd()
        self.LOCAL_PATH = cwd.replace("\\", "/")
        self.LOCAL_PATH = self.LOCAL_PATH.replace(
            "data/statistics", "data/convokit"
        )
        print(f"Working in {self.LOCAL_PATH}")

    def get_data(self):
        """
        Loads and outputs the Supreme Court Corpus data
        """

        print("Loading Supreme Court Corpus Data...")
        corpus = Corpus(filename=download("supreme-corpus"))
        corpus.dump("supreme_corpus", base_path=self.LOCAL_PATH)

    ### Begin reading data
    def load_data(self, file_name):
        """
        Opens the data and returns it as a dictionary

        :param file_name: The name of the file to open
        :return: The data as a dictionary
        """

        path = self.LOCAL_PATH + f"/supreme_corpus/{file_name}"
        if "jsonl" in file_name:
            data = []
            with open(path) as json_file:
                json_list = list(json_file)

            for json_str in json_list:
                clean_json = json.loads(json_str)
                data.append(clean_json)
        else:
            with open(path) as file:
                data = json.load(file)
        return data
