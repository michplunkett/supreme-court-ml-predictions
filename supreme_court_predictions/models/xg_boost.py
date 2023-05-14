"""
This file contains the XGBoost class that runs a gradient boosted tree model on
utterance data from the Supreme Court dataset. This class aims to predict the
results of a case based on the text learned from utterances.
"""

import os.path

import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from supreme_court_predictions.models.model import Model
from supreme_court_predictions.util.contants import SEED_CONSTANT
from supreme_court_predictions.util.functions import get_full_data_pathway

# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import GridSearchCV


class XGBoost(Model):
    """
    A class that runs a gradient boosted tree model on aggregated utterance and
    cases data from the Supreme Court dataset.
    """

    def __init__(
        self,
        debug_mode=False,
        dfs=[
            "case_aggregations.p",
            "judge_aggregations.p",
            "advocate_aggregations.p",
            "adversary_aggregations.p",
        ],
    ):
        self.accuracies = []
        self.debug_mode = debug_mode
        self.local_path = get_full_data_pathway("processed/")
        self.name = "Gradient Boosted Tree Model"

        # Dataframes and df names to run models against
        self.dataframes = []
        self.dataframe_names = []

        for df in dfs:
            # Make sure it's a file name
            if os.path.isfile(self.local_path + df):
                # Use the correct file reading function
                read_func = (
                    pd.read_pickle if df.split(".")[-1] == "p" else pd.read_csv
                )
                self.dataframes.append(read_func(self.local_path + df))

                # Add name of file
                self.dataframe_names.append(df.split(".")[0])

    def create(self, df):
        """
        Creates and runs a gradient boosted tree model on the given dataframe of
        utterance data.

        :param df: DataFrame containing utterance data

        :return (regressor, y_test, y_pred): A tuple that contains the
        regression model, test y-data, the predicted y-data.
        """
        self.print("Creating bag of words")

        vectorizer = CountVectorizer(analyzer="word", max_features=5000)
        vectorize_document = df.loc[:, "tokens"].apply(" ".join)
        bag_of_words_x = vectorizer.fit_transform(vectorize_document)
        bag_of_words_y = df.loc[:, "win_side"]

        X_train, X_test, y_train, y_test = train_test_split(
            bag_of_words_x,
            bag_of_words_y,
            test_size=0.20,
            random_state=123,
            stratify=bag_of_words_y,
        )

        self.print("Starting the XGBoost model")

        xgb_model = xgb.XGBClassifier(
            max_depth=7,
            n_estimators=300,
            objective="binary:logistic",
            random_state=SEED_CONSTANT,
            tree_method="gpu_hist",
            predictor="gpu_predictor",
        )

        # Fit the classifier on the training data
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)

        return xgb_model, y_test, y_pred

    def run(self):
        """
        Runs the create function on each type of aggregated utterance.
        """

        for df, df_name in zip(self.dataframes, self.dataframe_names):
            try:
                acc = self.create_and_measure(df, accuracy_score)
                self.accuracies.append(acc)

                # Print the results, if applicable
                self.print_results(self.name.lower(), acc, df_name)
            except ValueError:
                self.print("------------------------------------------")
                self.print(
                    "Error: training data is not big enough for this subset"
                )
                self.print("------------------------------------------")

        return self.accuracies

    def __repr__(self):
        """
        Overwrites default string representation of Model

        :return string representation of Model
        """

        parameters = []  # TODO - add me
        parameter_names = []  # TODO - add me

        s = f"MODEL TYPE: {self.name}\n"
        s += "PARAMETERS: \n"
        for parameter, name in zip(parameters, parameter_names):
            s += f"\t{name}: {str(parameter)}\n"

        s += "ACCURACIES: "
        for name, acc in zip(self.dataframe_names, self.accuracies):
            s += f"\n\t{name}: {str(acc)}"

        return s
