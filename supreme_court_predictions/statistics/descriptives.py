"""
This file houses the class that is used to provide descriptive summary 
statistics for the convokit datasets.
"""
import json
import os
import re

import pandas as pd
import numpy as np
from convokit import Corpus, download

from .datacleaner import DataCleaner
from supreme_court_predictions.util.contants import (
    ENCODING_UTF_8,
    ENCODING_UTF_8_SIG,
)


class Descriptives:
    """
    This class houses the functions needed to provide and export descriptive 
    statistics of the convokit data.
    """

    def __init__(self, downloaded_clean_corpus, save_data=True):
        self.advocates_stats = None
        self.conversations_stats = None
        self.speakers_stats = None
        self.utterances_stats = None
        self.voters_stats = None

        self.downloaded_clean_corpus = downloaded_clean_corpus
        self.save_data = save_data

        # Get local directory
        cwd = os.getcwd()
        self.local_path = (
            cwd.replace("\\", "/") +
            "/supreme_court_predictions/data/clean_convokit/"
        )
        print(f"Working in {self.local_path}")

        # Set output path
        self.output_path = self.local_path.replace(
            "clean_convokit", "statistics")
        print(f"Data will be saved to {self.output_path}")

        # Download necessary data
        if not downloaded_clean_corpus:
            print("Cleaned corpus not present, downloading...")
            cleaner = DataCleaner(True)
            cleaner.parse_all_data()
        else:
            print("Cleaned corpus already downloaded")

    def get_count_desc(self, df, cols):
        """
        [Insert]
        """

        df_stats = df.loc[:, cols].apply(lambda x: x.value_counts())
        df_stats.index.name = None
        df_stats = df_stats.T.stack()
        df_stats.name = "counts"
        df_stats = df_stats.to_frame()

        return df_stats

    def get_advocate_desc(self):
        """
        [Insert]
        """

        # Load data
        advocates = pd.read_csv(self.local_path + "advocates_df.csv")

        # Get advocate side descriptives
        advocate_stats = self.get_count_desc(advocates, ["side"])

        # Add additional descriptives
        for col in advocates.columns:
            if col in ["side", "id", "case_id"]:
                continue
            advocate_stats.loc[(col, ""), :] = len(
                advocates.loc[:, col].unique())

        return advocate_stats

    # TODO- add to DataCleaner
    def clean_roles(self, df):
        """
        [Insert]
        """

        roles = list(np.char.lower(
            np.array(df.loc[:, "role"]).astype(str)))

        clean_roles = {
            "inferred": 0, "for respondent": 0, "for petitioner": 0, "amicus curiae": 0, "for appellant": 0}

        for role in roles:
            if "inferred" in role:
                clean_roles["inferred"] = clean_roles.get("inferred") + 1
            elif "respondent" in role or "appelle" in role:
                clean_roles["for respondent"] = clean_roles.get(
                    "for respondent") + 1
            elif "petitioner" in role:
                clean_roles["for petitioner"] = clean_roles.get(
                    "for petitioner") + 1
            elif "amicus curiae" in role:
                clean_roles["amicus curiae"] = clean_roles.get(
                    "amicus curiae") + 1
            elif "appellant" in role:
                clean_roles["for appellant"] = clean_roles.get("appellant") + 1
            else:
                clean_roles[role] = clean_roles.get(role, 0) + 1

        return clean_roles

    def get_conversation_desc(self):
        """
        [Insert]
        """

        # Load data
        conversations = pd.read_csv(self.local_path + "conversations_df.csv")

        # For winning_side:
        # 1 = for petitioner, 0 = for respondent, 2 = unclear,
        # -1 = unavailable

        # Get dataframe of winning side descriptive stats
        conversation_stats = self.get_count_desc(
            conversations, ["winning_side"])

        # Add case count descriptives
        conversation_stats.loc[("cases", ""), :] = len(
            conversations.loc[:, "case_id"].unique())

        return conversation_stats

    def get_speaker_desc(self):
        """
        [Insert]
        """
        pass

        # Load data
        speakers = pd.read_csv(self.local_path + "speakers_df.csv")

        # Get descriptive stats
        speakers_stats = self.get_count_desc(
            speakers, ["speaker_type", "speaker_role"])

        # Add count of unique speaker names and keys
        speakers_stats.loc[("speaker_name", ""), :] = len(
            speakers.loc[:, "speaker_name"].unique())
        speakers_stats.loc[("speaker_key", ""), :] = len(
            speakers.loc[:, "speaker_key"].unique())

        return speakers_stats

    # TODO- may not need
    def get_utterances_desc(self):
        """
        [Insert]
        """
        filepath = self.local_path + "utterances_df.csv"
        utterances = pd.read_csv(filepath)

        pass

    def get_voters_desc(self):
        """
        [Insert]
        """
        # Load data
        voters = pd.read_csv(self.local_path + "voters_df.csv")

        # Get descriptive stats of votes
        voter_stats = self.get_count_desc(voters, ["vote"])
        voter_stats.loc[("justices", ""), :] = len(
            voters.loc[:, "voter"].unique())

        return voter_stats

    def parse_all_data(self):
        """
        [Insert]
        """

        print("Grabbing advocates descriptive statistics...")
        self.advocates_stats = self.get_advocate_desc()

        print("Grabbing conversations descriptive statistics...")
        self.conversations_stats = self.get_conversation_desc()

        print("Grabbing speakers descriptive statistics...")
        self.speakers_stats = self.get_speaker_desc()

        print("Grabbing voters descriptive statistics...")
        self.voters_stats = self.get_voters_desc()

        if self.save_data:
            desc_out = self.output_path + "/descriptive_statistics.xlsx"
            self.advocates_stats.to_csv(
                self.output_path + "/advocates_stats.csv",
                index=True,
                encoding=ENCODING_UTF_8_SIG,
            )

            self.conversations_stats.to_csv(
                self.output_path + "/conversations_stats.csv",
                index=True,
                encoding=ENCODING_UTF_8_SIG,
            )

            self.speakers_stats.to_csv(
                self.output_path + "/speakers_stats.csv",
                index=True,
                encoding=ENCODING_UTF_8_SIG,
            )

            self.voters_stats.to_csv(
                self.output_path + "/voters_stats.csv",
                index=True,
                encoding=ENCODING_UTF_8_SIG,
            )

            with pd.ExcelWriter(desc_out) as writer:
                self.advocates_stats.to_excel(writer, sheet_name='advocates')
                self.conversations_stats.to_excel(
                    writer, sheet_name='conversations')
                self.speakers_stats.to_excel(writer, sheet_name='speakers')
                self.voters_stats.to_excel(writer, sheet_name='voters')

            print("Data saved to " + self.output_path)
