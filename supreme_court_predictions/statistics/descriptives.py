"""
This file houses the class that is used to provide descriptive summary
statistics for the convokit datasets.
"""
import os

import numpy as np
import pandas as pd

from supreme_court_predictions.util.contants import ENCODING_UTF_8

from .datacleaner import DataCleaner


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

        self.save_data = save_data

        # Get local directory
        cwd = os.getcwd()
        self.local_path = (
            cwd.replace("\\", "/")
            + "/supreme_court_predictions/data/clean_convokit/"
        )
        print(f"Working in {self.local_path}")

        # Set output path
        self.output_path = self.local_path.replace(
            "clean_convokit", "statistics"
        )
        print(f"Data will be saved to {self.output_path}")

        # Download necessary data
        if not downloaded_clean_corpus:
            print("Cleaned corpus not present, downloading...")
            cleaner = DataCleaner(True)
            cleaner.parse_all_data()
        else:
            print("Cleaned corpus already downloaded")

    def fix_indices(self, main, sub):
        """
        Provides new pandas MultiIndices for a DataFrame.

        :param main: the main index name.
        :param sub: a list of the sub-indices.

        :return pandas.MultiIndex object containing the new indices.
        """

        new_indices = []
        for index in sub:
            new_indices.append((main, index))
        return pd.MultiIndex.from_tuples(new_indices)

    def get_count_desc(self, df, cols):
        """
        Calculates descriptive statistics of counts for a given DataFrame.

        :param df: DataFrame to provide count statistics for.
        :param cols: Columns from the DataFrame from which to retrieve counts.

        :return DataFrame of descriptive statistics.
        """

        df_stats = df.loc[:, cols].apply(lambda x: x.value_counts())
        df_stats.index.name = None
        df_stats = df_stats.T.stack()
        df_stats.name = "counts"
        df_stats = df_stats.to_frame()

        return df_stats

    def get_advocate_desc(self):
        """
        Calculates descriptive statistics for the advocates DataFrame.

        :return DataFrame of descriptive statistics.
        """
        # Load data
        advocates = pd.read_csv(self.local_path + "advocates_df.csv")

        # Get advocate side descriptives
        advocate_stats = self.get_count_desc(advocates, ["side"])
        sides = [
            "for petitioner",
            "for respondent",
            "unknown",
            "for amicus curiae",
        ]

        advocate_stats.index = self.fix_indices("side", sides)

        # Add total roles, total advocates, and aggregate roles to descriptives
        advocate_stats.loc[("total advocates", ""), :] = len(
            advocates.loc[:, "advocate"].unique()
        )

        advocate_stats.loc[("total roles"), :] = len(
            advocates.loc[:, "role"].unique()
        )

        advocate_stats = pd.concat(
            [advocate_stats, self.clean_roles(advocates)]
        )

        return advocate_stats

    def clean_roles(self, df):
        """
        Finds common roles for the "role" feature of the advocates dataframe
        and returns descriptive statistics for each role type.

        :param df: The advocates dataframe.
        :return DataFrame: Descriptive statistics providing counts of each role
                           type.
        """
        roles = list(np.char.lower(np.array(df.loc[:, "role"]).astype(str)))

        # Collect aggregate roles based on key words
        clean_roles = {
            "inferred": 0,
            "for respondent": 0,
            "for petitioner": 0,
            "for amicus curiae": 0,
        }
        for role in roles:
            if "inferred" in role:
                clean_roles["inferred"] = clean_roles.get("inferred") + 1
            elif "respondent" in role or "appelle" in role:
                clean_roles["for respondent"] = (
                    clean_roles.get("for respondent") + 1
                )
            elif "petitioner" in role or "appellant" in role:
                clean_roles["for petitioner"] = (
                    clean_roles.get("for petitioner") + 1
                )
            elif "amicus curiae" in role:
                clean_roles["for amicus curiae"] = (
                    clean_roles.get("for amicus curiae") + 1
                )
            else:
                clean_roles[role] = clean_roles.get(role, 0) + 1

        # Convert role counts to a DataFrame
        indices = self.fix_indices("aggregate roles", list(clean_roles.keys()))

        clean_roles_df = pd.DataFrame.from_dict(
            clean_roles, orient="index", columns=["counts"]
        )
        clean_roles_df.index = indices

        return clean_roles_df

    def get_conversation_desc(self):
        """
        Calculates descriptive statistics for the conversations DataFrame.

        :return DataFrame of descriptive statistics.
        """
        # Load data
        conversations = pd.read_csv(self.local_path + "conversations_df.csv")

        # Get dataframe of winning side descriptive stats
        winning_sides = [
            "for petitioner",
            "for respondent",
            "unclear",
            "unavailable",
        ]

        conversation_stats = self.get_count_desc(
            conversations, ["winning_side"]
        )

        conversation_stats.index = self.fix_indices(
            "winning side", winning_sides
        )

        # Add case count descriptives
        conversation_stats.loc[("total cases", ""), :] = len(
            conversations.loc[:, "case_id"].unique()
        )

        return conversation_stats

    def get_speaker_desc(self):
        """
        Calculates descriptive statistics for the speakers DataFrame.

        :return DataFrame of descriptive statistics.
        """
        # Load data
        speakers = pd.read_csv(self.local_path + "speakers_df.csv")

        # Get descriptive stats
        speakers_stats = self.get_count_desc(speakers, ["speaker_role"])

        speaker_roles = ["inaudible (U)", "justice (J)", "unknown (A)"]
        speakers_stats.index = self.fix_indices("speaker role", speaker_roles)

        # Add count of unique speaker names and keys
        speakers_stats.loc[("speaker names", ""), :] = len(
            speakers.loc[:, "speaker_name"].unique()
        )
        speakers_stats.loc[("speaker keys", ""), :] = len(
            speakers.loc[:, "speaker_key"].unique()
        )

        return speakers_stats

    def get_utterances_desc(self):
        """
        Calculates descriptive statistics for the utterances DataFrame.

        :return DataFrame of descriptive statistics.
        """
        # Load data
        utterances = pd.read_csv(self.local_path + "utterances_df.csv")

        # Get average number of utterances per case
        avg_utterance = utterances.loc[:, "case_id"].value_counts().mean()

        # Get average number of unique speakers per case
        avg_speakers = len(
            utterances.loc[:, ["case_id", "speaker"]].value_counts()
        ) / len(utterances.loc[:, "case_id"].unique())

        descriptives = {
            "num of utterances": avg_utterance,
            "num of speakers": avg_speakers,
        }

        utterances_descriptives = pd.DataFrame.from_dict(
            descriptives, orient="index", columns=["average"]
        )

        return utterances_descriptives

    def get_voters_desc(self):
        """
        Calculates descriptive statistics for the voters DataFrame.

        :return DataFrame of descriptive statistics.
        """
        # Load data
        voters = pd.read_csv(self.local_path + "voters_df.csv")

        # Get descriptive stats of votes
        voter_stats = self.get_count_desc(voters, ["vote"])
        voter_stats.loc[("justices", ""), :] = len(
            voters.loc[:, "voter"].unique()
        )

        return voter_stats

    def parse_all_data(self):
        """
        Calculates descriptive statistics for all of the supreme corpus
        DataFrames and exports them to csv/excel (if applicable).
        """

        print("Grabbing advocates descriptive statistics...")
        self.advocates_stats = self.get_advocate_desc()

        print("Grabbing conversations descriptive statistics...")
        self.conversations_stats = self.get_conversation_desc()

        print("Grabbing speakers descriptive statistics...")
        self.speakers_stats = self.get_speaker_desc()

        print("Grabbing voters descriptive statistics...")
        self.voters_stats = self.get_voters_desc()

        print("Grabbing utterances descriptive statistics...")
        self.utterances_stats = self.get_utterances_desc()

        if self.save_data:
            # Outputting to CSVs
            descriptives = [
                self.advocates_stats,
                self.conversations_stats,
                self.speakers_stats,
                self.voters_stats,
                self.utterances_stats,
            ]
            outpaths = [
                self.output_path + "/advocates_stats.csv",
                self.output_path + "/conversations_stats.csv",
                self.output_path + "/speakers_stats.csv",
                self.output_path + "/voters_stats.csv",
                self.output_path + "/utterances_stats.csv",
            ]

            for idx, desc in enumerate(descriptives):
                desc.to_csv(outpaths[idx], index=True, encoding=ENCODING_UTF_8)

            # Outputting to a single excel
            desc_out = self.output_path + "/descriptive_statistics.xlsx"
            with pd.ExcelWriter(desc_out) as writer:
                self.advocates_stats.to_excel(writer, sheet_name="advocates")
                self.conversations_stats.to_excel(
                    writer, sheet_name="conversations"
                )
                self.speakers_stats.to_excel(writer, sheet_name="speakers")
                self.voters_stats.to_excel(writer, sheet_name="voters")
                self.utterances_stats.to_excel(writer, sheet_name="utterances")

            print("Data saved to " + self.output_path)
