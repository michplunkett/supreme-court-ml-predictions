"""
This file houses the class that is used to clean the convokit data and convert
it to a usable format.
"""

import json
import re

import pandas as pd
from convokit import Corpus, download

from supreme_court_predictions.util.constants import (
    ENCODING_UTF_8,
    FILE_MODE_READ,
    LATEST_YEAR,
)
from supreme_court_predictions.util.functions import (
    debug_print,
    get_full_data_pathway,
)


class DataCleaner:
    """
    This class houses the functions needed to clean the convokit data and turn
    it into a usable format.
    """

    def __init__(self, debug_mode=False):
        self.cases_df = None
        self.clean_case_ids = None  # stores the case IDs to use
        self.clean_utterances_list = None
        self.debug_mode = debug_mode
        self.local_path = get_full_data_pathway("convokit/")
        self.output_path = get_full_data_pathway("clean_convokit/")
        self.speakers_df = None
        self.utterances_df = None

        debug_print(f"Working in {self.local_path}", self.debug_mode)
        debug_print(
            f"Data will be saved to {self.output_path}", self.debug_mode
        )

    def get_data(self):
        """
        Loads and outputs the Supreme Court Corpus data.
        """

        debug_print("Loading Supreme Court Corpus Data...", self.debug_mode)
        corpus = Corpus(filename=download("supreme-corpus"))
        corpus.dump("supreme_corpus", base_path=self.local_path)

    # Begin reading data
    def load_data(self, file_name):
        """
        Opens the data and returns it as a dictionary.

        :param file_name: The name of the file to open
        :return: The data as a dictionary
        """

        path = self.local_path + f"supreme_corpus/{file_name}"
        if "jsonl" in file_name:
            data = []
            with open(
                path, encoding=ENCODING_UTF_8, mode=FILE_MODE_READ
            ) as json_file:
                json_list = list(json_file)

            for json_str in json_list:
                clean_json = json.loads(json_str)
                data.append(clean_json)
        else:
            with open(
                path, encoding=ENCODING_UTF_8, mode=FILE_MODE_READ
            ) as file:
                data = json.load(file)
        return data

    def get_cases_df(self, cases_lst):
        """
        Converts the cases list to a metadata dataframe. Also provides list of
        cleaned and filtered cases to use.

        :param cases_lst: The cases' list containing dictionaries of cases.
        :return: The cases dataframe of case metadata.
        """

        # Convert to dataframe
        cases_df = self.load_cases_df(cases_lst)

        # Clean, filter, and return dataframe data
        self.clean_case_ids = self.get_clean_cases(cases_df)
        clean_cases_df = cases_df.loc[
            (cases_df.loc[:, "id"].isin(self.clean_case_ids)), :
        ]

        clean_cases_df = clean_cases_df.astype({"win_side": "int32"})
        return clean_cases_df

    @staticmethod
    def load_cases_df(cases_lst):
        """
        Generates and unclean and unfiltered dataframe of court cases.

        :param cases_lst: The cases' list containing dictionaries of cases.
        :return: The uncleaned/unfiltered cases dataframe of case metadata.
        """

        # Convert to dataframe
        metadata = {
            "id": [],
            "year": [],
            "citation": [],
            "title": [],
            "petitioner": [],
            "respondent": [],
            "docket_no": [],
            "court": [],
            "decided_date": [],
            "win_side": [],
            "is_eq_divided": [],
        }

        for case in cases_lst:
            # Get metadata
            for attr, observations in metadata.items():
                observations.append(case[attr])

        return pd.DataFrame(metadata)

    @staticmethod
    def get_clean_cases(cases):
        """
        Generates a list of cleaned case IDs.

        :param cases: An uncleaned dataframe of cases.
        : return: A list of clean case IDs
        """

        # Clean cases to 0/1 win side and cases from the last 5 years
        case_ids = cases.loc[
            (
                (cases.loc[:, "win_side"] == 0.0)
                | (cases.loc[:, "win_side"] == 1.0)
            )
            & (cases.loc[:, "year"] >= LATEST_YEAR - 5),
            "id",
        ].unique()

        return case_ids

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
            speaker_data["speaker_key"] = re.sub(r"^j__", "", speaker_key)
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

        # Remove low-quality data - unknown speaker types
        df_cleaned = df.loc[(df.loc[:, "speaker_type"] != "U"), :]

        return df_cleaned

    def get_conversation_dfs(self, conversations_dict):
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

            # Filter dataset based on cleaned case ids and 0/1 side
            if conversation_data["case_id"] in self.clean_case_ids:
                clean_dict = {
                    "id": conversation_id,
                    "case_id": conversation_data["case_id"],
                    "winning_side": conversation_data["win_side"],
                }

                advocates = conversation_data["advocates"]
                voters = conversation_data["votes_side"]

                for advocate in advocates:
                    if advocates[advocate]["side"] in [0, 1]:
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
                        if vote in [0, 1]:
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

        # Clean voters df - one vote per voter per case
        voters_df = voters_df.drop_duplicates(
            subset=["case_id", "voter"], keep="last"
        ).reset_index(drop=True)

        return conversation_metadata_df, advocates_df, voters_df

    def clean_utterances(self, utterances_list):
        """
        Cleans the utterances list.

        :param utterances_list: The utterances list
        :return: The cleaned utterances list
        """

        # Filter dataset based on cleaned case ids
        utterances_list_filtered = [
            u
            for u in utterances_list
            if u["meta"]["case_id"] in self.clean_case_ids
        ]

        clean_utterances_list = []
        for utterance in utterances_list_filtered:
            clean_dict = {
                "case_id": utterance["meta"]["case_id"],
                "speaker": utterance["speaker"],
                "speaker_type": utterance["meta"]["speaker_type"],
                "conversation_id": utterance["conversation_id"],
                "id": utterance["id"],
            }
            utterance_text = utterance["text"]
            clean_utterance = utterance_text.lower()
            no_newline = re.sub(r"[\r\n\t]", " ", clean_utterance)
            no_bracket = re.sub(r"[\[\]\(\)]", "", no_newline)

            clean_dict["text"] = no_bracket

            clean_utterances_list.append(clean_dict)

        utterances_df = pd.DataFrame(clean_utterances_list)

        return clean_utterances_list, utterances_df

    def parse_all_data(self):
        """
        Cleans and parses all the data.
        """
        debug_print("Parsing cases...", self.debug_mode)
        cases_list = self.load_data("cases.jsonl")
        self.cases_df = self.get_cases_df(cases_list)

        debug_print("Parsing speakers...", self.debug_mode)
        speakers_dict = self.load_data("speakers.json")
        self.speakers_df = self.speakers_to_df(speakers_dict)

        debug_print("Parsing conversations metadata...", self.debug_mode)
        conversations_dict = self.load_data("conversations.json")
        (
            self.conversations_df,
            self.advocates_df,
            self.voters_df,
        ) = self.get_conversation_dfs(conversations_dict)

        debug_print("Parsing utterances...", self.debug_mode)
        utterances_list = self.load_data("utterances.jsonl")
        self.clean_utterances_list, self.utterances_df = self.clean_utterances(
            utterances_list
        )

        self.speakers_df.to_csv(
            self.output_path + "/speakers_df.csv",
            index=False,
            encoding=ENCODING_UTF_8,
        )
        self.conversations_df.to_csv(
            self.output_path + "/conversations_df.csv",
            index=False,
            encoding=ENCODING_UTF_8,
        )
        self.advocates_df.to_csv(
            self.output_path + "/advocates_df.csv",
            index=False,
            encoding=ENCODING_UTF_8,
        )
        self.voters_df.to_csv(
            self.output_path + "/voters_df.csv",
            index=False,
            encoding=ENCODING_UTF_8,
        )
        self.utterances_df.to_csv(
            self.output_path + "/utterances_df.csv",
            index=False,
            encoding=ENCODING_UTF_8,
        )
        self.cases_df.to_csv(
            self.output_path + "/cases_df.csv",
            index=False,
            encoding=ENCODING_UTF_8,
        )

        debug_print(f"Data saved to {self.output_path}", self.debug_mode)
