"""
This file provides tokens per case.
"""
import pandas as pd

from supreme_court_predictions.util.contants import ENCODING_UTF_8
from supreme_court_predictions.util.files import get_full_pathway


class TokenAggregations:
    def __init__(self, save_data=True):
        self.all_tokens = None
        self.advocate_tokens = None
        self.adversary_tokens = None
        self.judge_tokens = None

        self.save_data = save_data

        # Get local directory
        self.local_path = get_full_pathway(
            "/supreme_court_predictions/data/clean_convokit/"
        )
        print(f"Working in {self.local_path}")

        # Set output path
        self.output_path = get_full_pathway(
            "/supreme_court_predictions/data/processed/"
        )
        print(f"Data will be saved to {self.output_path}")

        # Get advocate and voter side dataframes
        self.winside = self.get_win_side()
        self.voteside = self.get_vote_side()
        self.advocateside = self.get_advocate_side()
        self.utterances = self.get_utterances()
        self.utterance_sides = self.append_side(self.utterances)

    def get_utterances(self):
        """
        Load the utterances dataframe. Keep only the relevant columns, i.e.,
        "case_id", "speaker", "tokens", and "speaker_type".

        :return A dataframe of case utterances.
        """
        utterances = pd.read_csv(self.local_path + "utterances_df.csv")
        return utterances.loc[
            :, ["case_id", "speaker", "tokens", "speaker_type"]
        ]

    def get_win_side(self):
        """
        Get the winning side of the cases.

        :return A dataframe containing case IDs and the winning side (0=for
                respondent, 1=for petitioner)
        """
        winside = pd.read_csv(self.local_path + "cases_df.csv").rename(
            columns={"id": "case_id"}
        )
        return winside.loc[:, ["case_id", "win_side"]]

    def get_vote_side(self):
        """
        Get the voting side of the cases (for Judges only).

        :return A dataframe containing case IDs and the voting side (0=for
                respondent, 1=for petitioner)
        """
        voteside = pd.read_csv(self.local_path + "voters_df.csv")
        voteside = voteside.rename(
            columns={"vote": "side", "voter": "advocate"}
        )
        return voteside.loc[:, ["case_id", "advocate", "side"]]

    def get_advocate_side(self):
        """
        Get the advocating side of the cases (for non-Judges only).

        :return A dataframe containing case IDs and the advocating side (0=for
                respondent, 1=for petitioner)
        """

        advocateside = pd.read_csv(self.local_path + "advocates_df.csv")
        return advocateside.loc[:, ["case_id", "advocate", "side"]]

    def get_case_tokens(self, ut_tokens):
        """
        Returns a dataframe of utterances tokens per case, including winside of
        the case. The ut_tokens dataframe is expected to have the columns
        "case_id" and "tokens".

        :param cases: A dataframe of case utterances to aggregate tokens of.
        """

        # Aggregate tokens by case_id
        case_ids = ut_tokens.loc[:, "case_id"].unique()
        agg_tokens = {"case_id": [], "tokens": []}

        for case in case_ids:
            tokens = []
            agg_tokens["case_id"].append(case)
            for token in ut_tokens.loc[
                ut_tokens.loc[:, "case_id"] == case, "tokens"
            ]:
                # Preprocess instances - from string to list
                token = token.strip("[")
                token = token.strip("]")
                token = token.replace("'", "")
                token = token.replace(" ", "")
                token = token.split(",")

                tokens.extend(token)

            agg_tokens["tokens"].append(tokens)

        agg_tokens = pd.DataFrame.from_dict(agg_tokens)

        # merging WINSIDE onto tokens
        agg_tokens_winside = pd.merge(
            agg_tokens, self.winside, how="left", on="case_id"
        )

        return agg_tokens_winside

    def get_all_case_tokens(self):
        """
        Gets all the tokens for a given case and the outcome of the case.

        :return A dataframe of case utterances to aggregate tokens of.
        """
        return self.get_case_tokens(
            self.utterance_sides.loc[:, ["case_id", "tokens"]]
        )

    def append_side(self, utterances):
        """
        Adds the speaker's advocate or vote side to the utterances dataframe.

        :param ut: The utterances dataframe; must have only the columns case_id,
                speaker, speaker_type, and tokens

        :returns A dataframe with the side of the speaker appended to
                 utterances, also removing speaker accounts who don't have a
                 side.
        """

        # Renaming the speaker column for easier merging
        ut = utterances.rename(columns={"speaker": "advocate"})

        # Merging utterances with voter (judge) and advocate sides
        ut_sides = pd.merge(
            ut,
            pd.concat([self.voteside, self.advocateside]),
            how="left",
            left_on=["case_id", "advocate"],
            right_on=["case_id", "advocate"],
        )

        # Remove NA values
        ut_sides = ut_sides.loc[~ut_sides.loc[:, "side"].isna(), :]
        ut_sides = ut_sides.astype({"side": "int32"})

        return ut_sides

    def get_advocate_case_tokens(self, advocate=True):
        """
        Get all of the tokens for individuals either in favor of the petitioner
        or opposed to the petitioner.

        :param advocate: Whether to find the tokens for those in favor
                         (advocate=True) or opposed (advocate=False)
        :return A dataframe of tokens per case for petitioner advocates.
        """

        if advocate:
            ut = self.utterance_sides.loc[
                self.utterance_sides.loc[:, "side"] == 1, :
            ]
        else:
            ut = self.utterance_sides.loc[
                self.utterance_sides.loc[:, "side"] == 0, :
            ]
        return self.get_case_tokens(ut.loc[:, ["case_id", "tokens"]])

    def get_judge_case_tokens(self):
        """
        Get all of the tokens for only judges.

        :return A dataframe of tokens per judge per case.
        """

        ut = self.utterance_sides.loc[
            self.utterance_sides.loc[:, "speaker_type"] == "J", :
        ]
        return self.get_case_tokens(ut.loc[:, ["case_id", "tokens"]])

    def parse_all_data(self):
        """
        Generates token aggregations for 1) all speakers, 2) only advocates, 3)
        only adversaries, 4) only judges, and appends the winning side of the
        case. DataFrames and exports them to csv/excel (if applicable).
        """

        print("Grabbing token aggregation for all cases...")
        self.all_tokens = self.get_all_case_tokens()

        print("Grabbing token aggregation for advocates...")
        self.advocate_tokens = self.get_advocate_case_tokens(True)

        print("Grabbing token aggregation for adversaries...")
        self.adversary_tokens = self.get_advocate_case_tokens(False)

        print("Grabbing token aggregation for judges...")
        self.judge_tokens = self.get_judge_case_tokens()

        if self.save_data:
            print("Exporting files...")
            # Outputting to CSVs
            aggregations = [
                self.all_tokens,
                self.advocate_tokens,
                self.adversary_tokens,
                self.judge_tokens,
            ]
            outpaths = [
                self.output_path + "/case_aggregations.csv",
                self.output_path + "/advocate_aggregations.csv",
                self.output_path + "/adversary_aggregations.csv",
                self.output_path + "/judge_aggregations.csv",
            ]

            for idx, desc in enumerate(aggregations):
                desc.to_csv(outpaths[idx], index=True, encoding=ENCODING_UTF_8)

            print("Data saved to " + self.output_path)
