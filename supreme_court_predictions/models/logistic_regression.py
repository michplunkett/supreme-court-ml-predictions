"""
TODO: Need file document string
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression as skLR
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from supreme_court_predictions.util.files import get_full_data_pathway


class LogisticRegression:
    """
    TODO: Need document string
    """

    def __init__(self):
        self.local_path = get_full_data_pathway("processed/")

        self.total_utterances = pd.read_pickle(
            self.local_path + "case_aggregations.p"
        )
        self.advocate_utterances = pd.read_pickle(
            self.local_path + "advocate_aggregations.p"
        )
        self.adversary_utterances = pd.read_pickle(
            self.local_path + "adversary_aggregations.p"
        )
        self.judge_utterances = pd.read_pickle(
            self.local_path + "judge_aggregations.p"
        )

        self.run_regression()

    @staticmethod
    def logistic_regression(df):
        """
        Given a dataframe, run a logistic regression on the tokens column
        and return the accuracy score.

        :param df: A dataframe containing the tokens column.
        """
        vectorizer = CountVectorizer(max_features=5000)
        vectorize_document = df.loc[:, "tokens"].apply(" ".join)
        print("Creating bag of words")
        bag_of_words_x = vectorizer.fit_transform(vectorize_document)

        # TODO: after Chay's merge of dataframe, this can be updated.
        bag_of_words_y = np.random.randint(0, 2, len(vectorize_document))

        X_train, X_test, y_train, y_test = train_test_split(
            bag_of_words_x, bag_of_words_y, test_size=0.20, random_state=123
        )

        print("Starting the Logistic Regression")
        regressor = skLR()

        # Fit the classifier on the training data
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)

        return accuracy_score(y_true=y_test, y_pred=y_pred)

    def run_regression(self):
        """
        Runs the logistic regression on all four dataframes.
        """

        dfs = [
            self.total_utterances,
            self.judge_utterances,
            self.advocate_utterances,
            self.adversary_utterances,
        ]

        for df in dfs:
            print("------------------------------------------")
            print(f"Running regression on {df}...")
            acc = self.logistic_regression(df)
            print(f"Accuracy score: {acc}")
            print("------------------------------------------")
