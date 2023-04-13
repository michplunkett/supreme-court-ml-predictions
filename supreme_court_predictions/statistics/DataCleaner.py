"""
This class is used to clean the data and convert it to a usable format.
"""

# Global Imports
import os
import json
import pandas as pd

from supreme_court_predictions.util.contants import ENCODING_UTF_8_SIG

# Convokit import
from convokit import Corpus, download


class DataCleaner:
    def __init__(self, downloaded_corpus, save_data=True):
        self.downloaded_corpus = downloaded_corpus
        self.save_data = save_data

        # Get local directory
        cwd = os.getcwd()
        self.LOCAL_PATH = cwd.replace("\\", "/")
        self.LOCAL_PATH = self.LOCAL_PATH.replace(
            "supreme_court_predictions/data/statistics",
            "supreme_court_predictions/data/convokit ",
        )
        print(f"Working in {self.LOCAL_PATH}")

        # Set output path
        self.OUTPUT_PATH = self.LOCAL_PATH.replace("convokit", "clean_convokit")
        print(f"Data will be saved to {self.OUTPUT_PATH}")

        if not downloaded_corpus:
            print("Corpus not present, downloading...")
            self.get_data()
        else:
            print("Corpus already downloaded")

    def get_data(self):
        """
        Loads and outputs the Supreme Court Corpus data.
        """

        print("Loading Supreme Court Corpus Data...")
        corpus = Corpus(filename=download("supreme-corpus"))
        corpus.dump("supreme_corpus", base_path=self.LOCAL_PATH)

    # Begin reading data
    def load_data(self, file_name):
        """
        Opens the data and returns it as a dictionary.

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

    @staticmethod
    def speakers_to_df(speakers_dict):
        """
        Converts the speakers dictionary to a pandas dataframe.

        :param speakers_dict: The speaker's dictionary
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

    @staticmethod
    def get_conversation_dfs(conversations_dict):
        """
        Converts the conversations dictionary to several pandas dataframes.

        :param conversations_dict: The conversations' dictionary
        :return: The conversations dataframe, advocates dataframe,
                and voters dataframe
        """
        metadata_list = []
        advocates_list = []
        voters_list = []

        for conversation_id in list(conversations_dict.keys()):
            conversation_data = conversations_dict[conversation_id]["meta"]
            clean_dict = {
                "id": conversation_id,
                "case_id": conversation_data["case_id"],
                "winning_side": conversation_data["win_side"],
            }
            advocates = conversation_data["advocates"]
            voters = conversation_data["votes_side"]

            for advocate in advocates:
                advocate_dict = {
                    "id": conversation_id,
                    "case_id": conversation_data["case_id"],
                    "advocate": advocate,
                    "side": advocates[advocate]["side"],
                    "role": advocates[advocate]["role"],
                }
                advocates_list.append(advocate_dict)

            if voters:
                for voter, vote in voters.items():
                    vote_dict = {
                        "id": conversation_id,
                        "case_id": conversation_data["case_id"],
                        "voter": voter,
                        "vote": vote,
                    }
                    voters_list.append(vote_dict)
            else:
                vote_dict = {
                    "id": conversation_id,
                    "case_id": conversation_data["case_id"],
                }
                voters_list.append(vote_dict)

            metadata_list.append(clean_dict)

        conversation_metadata_df = pd.DataFrame(metadata_list)
        advocates_df = pd.DataFrame(advocates_list)
        voters_df = pd.DataFrame(voters_list)

        return conversation_metadata_df, advocates_df, voters_df

    @staticmethod
    def clean_utterances(utterances_list):
        """
        Cleans the utterances list.

        :param utterances_list: The utterances list
        :return: The cleaned utterances list
        """

        clean_utterances_list = []
        for utterance in utterances_list:
            clean_dict = {
                "case_id": utterance["meta"]["case_id"],
                "speaker": utterance["speaker"],
                "speaker_type": utterance["meta"]["speaker_type"],
                "conversation_id": utterance["conversation_id"],
                "id": utterance["id"],
            }
            utterance_text = utterance["text"]
            # TODO: More robust cleaning
            clean_utterance = utterance_text.replace("\n", " ").strip()
            clean_dict["text"] = clean_utterance

            clean_utterances_list.append(clean_dict)

        utterances_df = pd.DataFrame(clean_utterances_list)

        return clean_utterances_list, utterances_df

    def parse_all_data(self):
        """
        Cleans and parses all the data.
        """
        print("Parsing speakers...")
        speakers_dict = self.load_data("speakers.json")
        self.speakers_df = self.speakers_to_df(speakers_dict)

        print("Parsing conversations metadata...")
        conversations_dict = self.load_data("conversations.json")
        (
            self.conversations_df,
            self.advocates_df,
            self.voters_df,
        ) = self.get_conversation_dfs(conversations_dict)

        print("Parsing utterances...")
        utterances_list = self.load_data("utterances.jsonl")
        self.clean_utterances_list, self.utterances_df = self.clean_utterances(
            utterances_list
        )

        if self.save_data:
            self.speakers_df.to_csv(
                self.OUTPUT_PATH + "/speakers_df.csv",
                index=False,
                encoding=ENCODING_UTF_8_SIG,
            )
            self.conversations_df.to_csv(
                self.OUTPUT_PATH + "/conversations_df.csv",
                index=False,
                encoding=ENCODING_UTF_8_SIG,
            )
            self.advocates_df.to_csv(
                self.OUTPUT_PATH + "/advocates_df.csv",
                index=False,
                encoding=ENCODING_UTF_8_SIG,
            )
            self.voters_df.to_csv(
                self.OUTPUT_PATH + "/voters_df.csv",
                index=False,
                encoding=ENCODING_UTF_8_SIG,
            )
            self.utterances_df.to_csv(
                self.OUTPUT_PATH + "/utterances_df.csv",
                index=False,
                encoding=ENCODING_UTF_8_SIG,
            )

            print("Data saved to " + self.OUTPUT_PATH)
