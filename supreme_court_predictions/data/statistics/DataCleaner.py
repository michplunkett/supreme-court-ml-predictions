"""
This class is used to clean the data and convert it to a usable format
"""

# Global Imports
import os
import json
import pandas as pd

# Convokit import
from convokit import Corpus, download


class DataCleaner:
    def __init__(self, downloaded_corpus):
        self.downloaded_corpus = downloaded_corpus

        # Get local directory
        cwd = os.getcwd()
        self.LOCAL_PATH = cwd.replace("\\", "/")
        self.LOCAL_PATH = self.LOCAL_PATH.replace(
            "data/statistics", "data/convokit"
        )
        print(f"Working in {self.LOCAL_PATH}")

        if not downloaded_corpus:
            print("Corpus not present, downloading...")
            self.get_data()

    def get_data(self):
        """
        Loads and outputs the Supreme Court Corpus data
        """

        print("Loading Supreme Court Corpus Data...")
        corpus = Corpus(filename=download("supreme-corpus"))
        corpus.dump("supreme_corpus", base_path=self.LOCAL_PATH)

    # Begin reading data

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

    def speakers_to_df(self, speakers_dict):
        """
        Converts the speakers dictionary to a pandas dataframe

        :param speakers_dict: The speakers dictionary
        :return: The speakers dataframe
        """

        dict_list = []
        for speaker_key in list(speakers_dict.keys()):
            speaker_data = speakers_dict[speaker_key]["meta"]
            speaker_data["speaker_key"] = speaker_key
            dict_list.append(speaker_data)

        df = pd.DataFrame(dict_list)
        df.rename(
            columns={
                "name": "speaker_name",
                "type": "speaker_type",
                "role": "speaker_role",
            },
            inplace=True,
        )
        return df

    def get_conversation_dfs(self, conversations_dict):
        """
        Converts the conversations dictionary to several
        pandas dataframes

        :param conversations_dict: The conversations dictionary
        :return: The conversations dataframe, advocates dataframe,
                and voters dataframe
        """
        metadata_list = []
        advocates_list = []
        voters_list = []

        for conversation_id in list(conversations_dict.keys()):
            clean_dict = {}
            conversation_data = conversations_dict[conversation_id]["meta"]
            clean_dict["id"] = conversation_id
            clean_dict["case_id"] = conversation_data["case_id"]
            clean_dict["winning_side"] = conversation_data["win_side"]

            advocates = conversation_data["advocates"]
            voters = conversation_data["votes_side"]

            for advocate in advocates:
                advocate_dict = {}
                advocate_dict["id"] = conversation_id
                advocate_dict["case_id"] = conversation_data["case_id"]
                advocate_dict["advocate"] = advocate
                advocate_dict["side"] = advocates[advocate]["side"]
                advocate_dict["role"] = advocates[advocate]["role"]
                advocates_list.append(advocate_dict)

            if voters:
                for voter, vote in voters.items():
                    vote_dict = {}
                    vote_dict["id"] = conversation_id
                    vote_dict["case_id"] = conversation_data["case_id"]
                    vote_dict["voter"] = voter
                    vote_dict["vote"] = vote
                    voters_list.append(vote_dict)
            else:
                vote_dict = {}
                vote_dict["id"] = conversation_id
                vote_dict["case_id"] = conversation_data["case_id"]
                voters_list.append(vote_dict)

            metadata_list.append(clean_dict)

        conversation_metadata_df = pd.DataFrame(metadata_list)
        advocates_df = pd.DataFrame(advocates_list)
        voters_df = pd.DataFrame(voters_list)

        return conversation_metadata_df, advocates_df, voters_df

    def clean_utterances(self, utterances_list):
        """
        Cleans the utterances list

        :param utterances_list: The utterances list
        :return: The cleaned utterances list
        """

        clean_utterances_list = []
        for utterance in utterances_list:
            clean_dict = {}
            clean_dict["case_id"] = utterance["meta"]["case_id"]
            clean_dict["speaker"] = utterance["speaker"]
            clean_dict["speaker_type"] = utterance["meta"]["speaker_type"]
            clean_dict["conversation_id"] = utterance["conversation_id"]
            clean_dict["id"] = utterance["id"]
            utterance_text = utterance["text"]
            clean_utterance = utterance_text.replace("\n", " ").strip()
            clean_dict["text"] = clean_utterance

            clean_utterances_list.append(clean_dict)

        utterances_df = pd.DataFrame(clean_utterances_list)

        return clean_utterances_list, utterances_df
