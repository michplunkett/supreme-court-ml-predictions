"""
This file provides aggregate tokens per case.
"""
import pickle

import pandas as pd

from supreme_court_predictions.util.functions import (
    debug_print,
    get_full_data_pathway,
)

MIN_SENTENCE_LEN = 5
SPLIT_PUNC = "."
FALSE_PUNCT = ["no.", "v."]


class TokenAggregations:
    """
    This class aggregates tokens on a per case basis.
    """

    def __init__(self, debug_mode=False):
        self.adversary_tokens = None
        self.advocate_tokens = None
        self.all_tokens = None
        self.debug_mode = debug_mode
        self.judge_tokens = None

        # Get local directory
        self.local_path = get_full_data_pathway("clean_convokit/")
        debug_print(f"Working in {self.local_path}", self.debug_mode)

        # Set output path
        self.output_path = get_full_data_pathway("processed/")
        debug_print(
            f"Data will be saved to {self.output_path}", self.debug_mode
        )

        # Get advocate and voter side dataframes
        self.win_side = self.get_win_side()
        self.vote_side = self.get_vote_side()
        self.advocate_side = self.get_advocate_side()
        self.utterances = self.get_utterances()
        self.utterance_sides = self.append_side(self.utterances)

    @staticmethod
    def get_sentence_count(text):
        """
        Calculates the number of sentences and the number of words per sentence
        in a text.

        :param text (str): The text to generate sentence counts from.
        :returns (int) Number of sentences in the given text.
        """
        # Remove false punctuation
        for punc in FALSE_PUNCT:
            if punc in text:
                text = text.replace(punc, "")

        # Get individual sentences
        text_lst = text.split(SPLIT_PUNC)
        text_lst = [
            sen.strip() for sen in text_lst if len(sen) >= MIN_SENTENCE_LEN
        ]

        return len(text_lst)

    def engineer_features(self, ut):
        """
        Engineers features for the utterances dataset, including number of
        sentences and number of words per utterance.

        :param (ut): Utterances dataframe to engineer features for.
        :returns (DataFrame): Utterances dataframe with added columns for number
                              of words and sentences.
        """
        new_ut = ut.copy()

        # Add sentence count to utterances
        new_ut.loc[:, "num_sentences"] = new_ut.loc[:, "text"].map(
            self.get_sentence_count
        )

        # Add word count to utterances
        new_ut.loc[:, "num_words"] = new_ut.loc[:, "tokens"].map(
            lambda x: len(x)
        )

        return new_ut

    def get_utterances(self):
        """
        Load the utterances dataframe. Keep only the relevant columns, i.e.,
        "case_id", "speaker", "tokens", and "speaker_type".

        :return A dataframe of case utterances.
        """
        utterances = pd.read_csv(self.local_path + "utterances_df.csv")
        ut_eng = self.engineer_features(utterances)

        return ut_eng.loc[
            :,
            [
                "case_id",
                "speaker",
                "speaker_type",
                "tokens",
                "num_sentences",
                "num_words",
            ],
        ]

    def get_win_side(self):
        """
        Get the winning side of the cases.

        :return A dataframe containing case IDs and the winning side (0=for
            respondent, 1=for petitioner)
        """
        win_side_df = pd.read_csv(self.local_path + "cases_df.csv").rename(
            columns={"id": "case_id"}
        )
        return win_side_df.loc[:, ["year", "case_id", "win_side"]]

    def get_vote_side(self):
        """
        Get the voting side of the cases (for Judges only).

        :return A dataframe containing case IDs and the voting side (0=for
            respondent, 1=for petitioner)
        """
        vote_side_df = pd.read_csv(self.local_path + "voters_df.csv")
        vote_side_df = vote_side_df.rename(
            columns={"vote": "side", "voter": "advocate"}
        )
        return vote_side_df.loc[:, ["case_id", "advocate", "side"]]

    def get_advocate_side(self):
        """
        Get the advocating side of the cases (for non-Judges only).

        :return A dataframe containing case IDs and the advocating side (0=for
            respondent, 1=for petitioner)
        """
        advocate_side_df = pd.read_csv(self.local_path + "advocates_df.csv")
        return advocate_side_df.loc[:, ["case_id", "advocate", "side"]]

    def get_case_tokens(self, utterance_tokens):
        """
        Returns a dataframe of utterances tokens per case, including win_side of
        the case. The utterance_tokens dataframe is expected to have the columns
        "case_id" and "tokens".

        :param utterance_tokens: A dataframe of case utterances to aggregate
            tokens of.
        """
        # Aggregate tokens by case_id
        case_ids = utterance_tokens.loc[:, "case_id"].unique()
        agg_tokens = {
            "case_id": [],
            "tokens": [],
            "avg_num_sentences": [],
            "avg_num_words": [],
        }

        replacement = "[]' "

        for case in case_ids:
            tokens = []
            agg_tokens["case_id"].append(case)

            # Add aggregate tokens
            for token in utterance_tokens.loc[
                utterance_tokens.loc[:, "case_id"] == case, "tokens"
            ]:
                # Preprocess instances - from string to list
                token = token.translate(str.maketrans("", "", replacement))
                token = token.split(",")

                tokens.extend(token)
            agg_tokens["tokens"].append(tokens)

            # Add average sentence count
            agg_tokens["avg_num_sentences"].append(
                utterance_tokens.loc[
                    utterance_tokens.loc[:, "case_id"] == case, "num_sentences"
                ].mean()
            )

            # Add average word count
            agg_tokens["avg_num_words"].append(
                utterance_tokens.loc[
                    utterance_tokens.loc[:, "case_id"] == case, "num_words"
                ].mean()
            )

        agg_tokens = pd.DataFrame.from_dict(agg_tokens)

        # merging win_side onto tokens
        agg_tokens_win_side = pd.merge(
            agg_tokens, self.win_side, how="left", on="case_id"
        )

        return agg_tokens_win_side

    def get_all_case_tokens(self):
        """
        Gets all the tokens for a given case and the outcome of the case.

        :return A dataframe of case utterances to aggregate tokens of.
        """
        return self.get_case_tokens(
            self.utterance_sides.loc[
                :, ["case_id", "tokens", "num_sentences", "num_words"]
            ]
        )

    def append_side(self, utterances):
        """
        Adds the speaker's advocate or vote side to the utterances dataframe.

        :param utterances: The utterances dataframe; must have only the columns
            case_id, speaker, speaker_type, and tokens

        :returns A dataframe with the side of the speaker appended to
            utterances, also removing speaker accounts who don't have a
            side.
        """
        # Renaming the speaker column for easier merging
        ut = utterances.rename(columns={"speaker": "advocate"})

        # Merging utterances with voter (judge) and advocate sides
        ut_sides = pd.merge(
            ut,
            pd.concat([self.vote_side, self.advocate_side]),
            how="left",
            on=["case_id", "advocate"],
        )

        # Remove NA values
        ut_sides = ut_sides.loc[~ut_sides.loc[:, "side"].isna(), :]
        ut_sides = ut_sides.astype({"side": "int32"})

        return ut_sides

    def get_advocate_case_tokens(self, advocate=True):
        """
        Get all tokens for individuals either in favor of the petitioner or
        opposed to the petitioner.

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
        return self.get_case_tokens(
            ut.loc[:, ["case_id", "tokens", "num_sentences", "num_words"]]
        )

    def get_judge_case_tokens(self):
        """
        Get all of the tokens for only judges.

        :return A dataframe of tokens per judge per case.
        """
        ut = self.utterance_sides.loc[
            self.utterance_sides.loc[:, "speaker_type"] == "J", :
        ]
        judge_tokens = self.get_case_tokens(
            ut.loc[:, ["case_id", "tokens", "num_sentences", "num_words"]]
        )
        return judge_tokens

    def parse_all_data(self):
        """
        Generates token aggregations for 1) all speakers, 2) only advocates, 3)
        only adversaries, 4) only judges, and appends the winning side of the
        case. DataFrames and exports them as pickle objects (if applicable).
        """
        debug_print(
            "Grabbing token aggregation for all cases...", self.debug_mode
        )
        self.all_tokens = self.get_all_case_tokens()

        debug_print(
            "Grabbing token aggregation for advocates...", self.debug_mode
        )
        self.advocate_tokens = self.get_advocate_case_tokens(True)

        debug_print(
            "Grabbing token aggregation for adversaries...", self.debug_mode
        )
        self.adversary_tokens = self.get_advocate_case_tokens(False)

        debug_print("Grabbing token aggregation for judges...", self.debug_mode)
        self.judge_tokens = self.get_judge_case_tokens()

        debug_print("Exporting files...")
        # Outputting to CSVs
        aggregations = [
            self.all_tokens,
            self.advocate_tokens,
            self.adversary_tokens,
            self.judge_tokens,
        ]
        output_paths = [
            self.output_path + "/case_aggregations.p",
            self.output_path + "/advocate_aggregations.p",
            self.output_path + "/adversary_aggregations.p",
            self.output_path + "/judge_aggregations.p",
        ]

        for idx, agg in enumerate(aggregations):
            pickle.dump(agg, open(output_paths[idx], "wb"))

        debug_print("Data saved to " + self.output_path, self.debug_mode)
