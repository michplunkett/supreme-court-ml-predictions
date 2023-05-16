"""
This file houses the class that is used to provide descriptive summary
statistics for the convokit datasets.
"""
import numpy as np
import pandas as pd

from supreme_court_predictions.util.contants import ENCODING_UTF_8
from supreme_court_predictions.util.functions import (
    debug_print,
    get_full_data_pathway,
)


class DescriptiveStatistics:
    """
    This class houses the functions needed to provide and export descriptive
    statistics of the convokit data.
    """

    def __init__(self, debug_mode=False, print_to_csv=True):
        self.advocates_stats = None
        self.cases_stats = None
        self.debug_mode = debug_mode
        self.print_to_csv = print_to_csv
        self.speakers_stats = None
        self.utterances_stats = None
        self.voters_stats = None

        # Get local directory
        self.local_path = get_full_data_pathway("clean_convokit/")
        debug_print(f"Working in {self.local_path}", self.debug_mode)

        # Set output path
        self.output_path = get_full_data_pathway("statistics/")
        debug_print(
            f"Data will be saved to {self.output_path}", self.debug_mode
        )

    @staticmethod
    def fix_indices(main, sub):
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

    @staticmethod
    def get_count_statistics(df, cols):
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

        # get percentages
        if len(df_stats) > 1:
            total = sum(df_stats.loc[:, "counts"])
            df_stats.loc[:, "percentages"] = (
                df_stats.loc[:, "counts"] / total
            ) * 100

        return df_stats

    def get_advocate_statistics(self):
        """
        Calculates descriptive statistics for the advocates DataFrame.

        :return DataFrame of descriptive statistics.
        """
        # Load data
        advocates = pd.read_csv(self.local_path + "advocates_df.csv")

        # Get advocate side descriptive statistics
        advocate_stats = self.get_count_statistics(advocates, ["side"])
        sides = [
            "for petitioner",
            "for respondent",
        ]

        advocate_stats.index = self.fix_indices("side", sides)

        # Add total roles, total advocates, and aggregate roles to descriptive
        # statistics
        advocate_stats.loc[("total advocates", ""), "counts"] = len(
            advocates.loc[:, "advocate"].unique()
        )

        advocate_stats.loc["total roles", "counts"] = len(
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

        # Collect aggregate roles based on keywords
        clean_roles = {
            "inferred": 0,
            "for respondent": 0,
            "for petitioner": 0,
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
            else:
                clean_roles[role] = clean_roles.get(role, 0) + 1

        # Convert role counts to a DataFrame
        indices = self.fix_indices("aggregate roles", list(clean_roles.keys()))

        clean_roles_df = pd.DataFrame.from_dict(
            clean_roles, orient="index", columns=["counts"]
        )
        clean_roles_df.index = indices

        # add percentages
        total = sum(clean_roles_df.loc[:, "counts"])
        clean_roles_df.loc[:, "percentages"] = (
            clean_roles_df.loc[:, "counts"] / total
        ) * 100

        return clean_roles_df

    def get_speaker_statistics(self):
        """
        Calculates descriptive statistics for the speakers DataFrame.

        :return DataFrame of descriptive statistics.
        """
        # Load data
        speakers = pd.read_csv(self.local_path + "speakers_df.csv")

        # Get descriptive statistics
        speakers_stats = self.get_count_statistics(speakers, ["speaker_type"])

        speaker_types = ["advocate (A)", "justice (J)"]
        speakers_stats.index = self.fix_indices("speaker type", speaker_types)

        # Add count of unique speaker names and keys
        speakers_stats.loc[("speaker names", ""), "counts"] = len(
            speakers.loc[:, "speaker_name"].unique()
        )
        speakers_stats.loc[("speaker keys", ""), "counts"] = len(
            speakers.loc[:, "speaker_key"].unique()
        )

        return speakers_stats

    def get_utterance_statistics(self):
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

        descriptive_statistics = {
            "num of utterances": avg_utterance,
            "num of speakers": avg_speakers,
        }

        utterance_descriptive_statistics = pd.DataFrame.from_dict(
            descriptive_statistics, orient="index", columns=["average"]
        )

        return utterance_descriptive_statistics

    def get_voters_statistics(self):
        """
        Calculates descriptive statistics for the voters DataFrame.

        :return DataFrame of descriptive statistics.
        """
        # Load data
        voters = pd.read_csv(self.local_path + "voters_df.csv")

        # Get descriptive statistics for votes
        voter_statistics = self.get_count_statistics(voters, ["vote"])
        vote_types = ["for petitioner", "for respondent"]
        voter_statistics.index = self.fix_indices("votes", vote_types)

        voter_statistics.loc[("justices", ""), "counts"] = len(
            voters.loc[:, "voter"].unique()
        )

        voter_statistics = pd.concat(
            [voter_statistics, self.votes_by_justice(voters)]
        )

        return voter_statistics

    def votes_by_justice(self, voters):
        """
        Calculates the proportion of votes in favor of the petitioner by
        justice.

        :param voters: Dataframe including voter information.
        :return Dataframe of descriptive statistics for votes by SCOTUS
                justices.
        """

        # Get justices
        justices = voters.loc[:, "voter"].unique()
        indices = self.fix_indices("justice", justices)
        voters_df = pd.DataFrame(
            index=indices, columns=["counts", "percentages"]
        )

        for justice in justices:
            # Get counts
            voters_df.loc[("justice", justice), "counts"] = len(
                voters.loc[voters.loc[:, "voter"] == justice, :]
            )

            # Get percent for petitioner
            voters_df.loc[("justice", justice), "percentages"] = voters.loc[
                voters.loc[:, "voter"] == justice, "vote"
            ].sum() / len(voters.loc[voters.loc[:, "voter"] == justice, :])

        return voters_df

    def get_cases_statistics(self):
        """
        Calculates descriptive statistics for the cases DataFrame.

        :return DataFrame of descriptive statistics.
        """
        # Load data
        cases = pd.read_csv(self.local_path + "cases_df.csv")

        # Get dataframe of winning side descriptive stats
        winning_sides = [
            "for petitioner",
            "for respondent",
        ]

        case_statistics = self.get_count_statistics(cases, ["win_side"])

        case_statistics.index = self.fix_indices("win side", winning_sides)

        # Get counts
        ct_values = []
        attributes = ["id", "court", "year", "petitioner", "respondent"]

        for attr in attributes:
            ct_values.append(len(cases.loc[:, attr].unique()))

        # Get min/max - years
        min_year = min(cases.loc[:, "year"].unique())
        max_year = max(cases.loc[:, "year"].unique())

        # Add counts to df
        for attr, count in zip(attributes, ct_values):
            attr = attr + "s"
            if attr == "years":
                attr = f"years ({min_year} to {max_year})"
            elif attr == "ids":
                attr = "cases"
            case_statistics.loc[(attr, ""), "counts"] = count

        return case_statistics

    def parse_all_data(self):
        """
        Calculates descriptive statistics for all supreme corpus DataFrames
        and exports them to csv/excel (if applicable).
        """
        debug_print("Grabbing case descriptive statistics...", self.debug_mode)
        self.cases_stats = self.get_cases_statistics()

        debug_print(
            "Grabbing advocates descriptive statistics...", self.debug_mode
        )
        self.advocates_stats = self.get_advocate_statistics()

        debug_print(
            "Grabbing speakers descriptive statistics...", self.debug_mode
        )
        self.speakers_stats = self.get_speaker_statistics()

        debug_print(
            "Grabbing voters descriptive statistics...", self.debug_mode
        )
        self.voters_stats = self.get_voters_statistics()

        debug_print(
            "Grabbing utterances descriptive statistics...", self.debug_mode
        )
        self.utterances_stats = self.get_utterance_statistics()

        # Outputting to CSVs
        descriptive_statistics = [
            self.advocates_stats,
            self.cases_stats,
            self.speakers_stats,
            self.voters_stats,
            self.utterances_stats,
        ]
        output_paths = [
            self.output_path + "/advocates_stats.csv",
            self.output_path + "/cases_stats.csv",
            self.output_path + "/speakers_stats.csv",
            self.output_path + "/voters_stats.csv",
            self.output_path + "/utterances_stats.csv",
        ]

        for idx, desc_stats in enumerate(descriptive_statistics):
            # Print statistics to CSV files
            if self.print_to_csv:
                desc_stats.to_csv(
                    output_paths[idx], index=True, encoding=ENCODING_UTF_8
                )
            # Print statistics to std.out
            # Create the title of statistical output from the output title
            if self.debug_mode:
                statistics_title = (
                    output_paths[idx]
                    .split("/")[-1]
                    .replace("_", " ")
                    .replace(".csv", "")
                    .replace("stats", "statistics")
                    .title()
                )
                print(statistics_title)
                print(desc_stats)
                if idx != len(descriptive_statistics) - 1:
                    print("\n")

        # Outputting to a single Excel file
        desc_out = self.output_path + "/descriptive_statistics.xlsx"
        # Known ExcelWriter issue:
        # https://github.com/pylint-dev/pylint/issues/3060
        # pylint: disable=abstract-class-instantiated
        with pd.ExcelWriter(desc_out) as writer:
            # pylint: enable=abstract-class-instantiated
            self.advocates_stats.to_excel(writer, sheet_name="advocates")
            self.cases_stats.to_excel(writer, sheet_name="cases")
            self.speakers_stats.to_excel(writer, sheet_name="speakers")
            self.voters_stats.to_excel(writer, sheet_name="voters")
            self.utterances_stats.to_excel(writer, sheet_name="utterances")

        debug_print("Data saved to " + self.output_path, self.debug_mode)
