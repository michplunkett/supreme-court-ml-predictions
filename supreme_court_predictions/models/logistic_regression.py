"""
This LogisticRegression class runs logistic regression 
on utterance data from the Supreme Court dataset. This class aims to predict
the results of a case based on the text learned from utterances. 
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression as skLR
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from supreme_court_predictions.models.model import Model
from supreme_court_predictions.util.files import get_full_data_pathway


class LogisticRegression(Model):
    """
    A class that runs logistic regression on aggregated utterance and cases data
    from the Supreme Court dataset.
    """

    def __init__(
        self,
        max_features=5000,
        test_size=0.20,
        randomstate=123,
        max_iter=1000,
        print_results=True,
    ):
        self.local_path = get_full_data_pathway("processed/")
        self.max_features = max_features
        self.test_size = test_size
        self.randomstate = randomstate
        self.max_iter = max_iter
        self.print = print_results

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

    def create(self, df):
        """
        Creates and runs a logistic regression on the given dataframe of
        utterance data.

        :param df: DataFrame containing utterance data

        :return (regressor, y_test, y_pred): A tuple that contains the
        regression model, test y-data, the predicted y-data.
        """
        vectorizer = CountVectorizer(
            analyzer="word", max_features=self.max_features
        )
        vectorize_document = df.loc[:, "tokens"].apply(" ".join)
        print("Creating bag of words")
        bag_of_words_x = vectorizer.fit_transform(vectorize_document)

        bag_of_words_y = df.loc[:, "win_side"]

        X_train, X_test, y_train, y_test = train_test_split(
            bag_of_words_x,
            bag_of_words_y,
            test_size=self.test_size,
            random_state=self.randomstate,
            stratify=bag_of_words_y,
        )

        print("Starting the Logistic Regression")
        regressor = skLR(max_iter=self.max_iter)

        # Fit the classifier on the training data
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)

        return regressor, y_test, y_pred

    def run(self):
        """
        Runs the create function on each type of aggregated utterance.
        """

        dfs = [
            self.total_utterances,
            self.judge_utterances,
            self.advocate_utterances,
            self.adversary_utterances,
        ]
        df_names = [
            "total_utterances",
            "judge_utterances",
            "advocate_utterances",
            "adversary_utterances",
        ]

        accuracies = []

        for df in dfs:
            try:
                acc = self.create_and_measure(df, accuracy_score)
                accuracies.append(acc)
            except ValueError:
                print("------------------------------------------")
                print("Error: training data is not big enough for this subset")
                print("------------------------------------------")

        # Print the results, if applicable
        if self.print:
            self.print_results("regression", accuracies, df_names)

        return accuracies
