"""
This LogisticRegression class runs logistic regression 
on utterance data from the Supreme Court dataset. This class aims to predict
the results of a case based on the text learned from utterances. 
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
    A class that runs logistic regression on aggregated utterance and cases data 
    from the Supreme Court dataset.
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
        Perform logistic regression on the given dataframe of utterance data.
        It regresses on the entire dataset and regresses for judges, advocates, 
        and adversaries.

        Args:
            df (pd.DataFrame): DataFrame containing utterance data

        Returns:
            float: Accuracy score of the logistic regression model
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

        print("Starting the Logistic Regression on utterances")
        regressor = skLR()

        # Fit the classifier on the training data
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)

        return accuracy_score(y_true=y_test, y_pred=y_pred)

    def run_regression(self):
        """
        Run logistic regression on each type of aggregated utterance data.
        It regresses on the entire dataset and regresses for judges, advocates, 
        and adversaries.
        """

        dfs = [
            self.total_utterances,
            self.judge_utterances,
            self.advocate_utterances,
            self.adversary_utterances,
        ]

        for df in dfs:
            print("------------------------------------------")
            print("Running regression on total utterances...")
            acc = self.logistic_regression(df)
            print(f"Accuracy score: {acc}")
            print("------------------------------------------")