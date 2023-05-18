"""
This file contains the XGBoost class that runs a gradient boosted tree model on
utterance data from the Supreme Court dataset. This class aims to predict the
results of a case based on the text learned from utterances.
"""

import os.path

import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from supreme_court_predictions.models.model import Model
from supreme_court_predictions.util.constants import SEED_CONSTANT
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
        eta=0.3,
        max_depth=7,
        max_features=5000,
        n_estimators=100,
        subsample=1,
        test_size=0.20,
    ):
        # Model outputs
        self.accuracies = {}
        self.f1 = {}
        self.models = {}
        self.confusion_matrix = {}
        self.dataframes = []
        self.dataframe_names = []

        # Data and display
        self.debug_mode = debug_mode
        self.local_path = get_full_data_pathway("processed/")
        self.name = "Gradient Boosted Tree Model"

        # Parameters (model and others)
        self.max_features = max_features
        self.test_size = test_size
        # Model params
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.eta = eta
        self.subsample = subsample

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

        vectorizer = CountVectorizer(
            analyzer="word", max_features=self.max_features
        )
        vectorize_document = df.loc[:, "tokens"].apply(" ".join)
        bag_of_words_x = vectorizer.fit_transform(vectorize_document)
        bag_of_words_y = df.loc[:, "win_side"]

        # Append bag of words to other attributes in df
        new_df = df.drop(columns=["case_id", "tokens", "win_side"])
        new_df = pd.concat(
            [
                new_df,
                pd.DataFrame(
                    bag_of_words_x.toarray(),
                    columns=vectorizer.get_feature_names_out(),
                ),
            ],
            axis=1,
        )

        X_train, X_test, y_train, y_test = train_test_split(
            new_df,
            bag_of_words_y,
            test_size=self.test_size,
            random_state=SEED_CONSTANT,
            stratify=bag_of_words_y,
        )

        self.print("Starting the XGBoost model")

        xgb_model = xgb.XGBClassifier(
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            eta=self.eta,
            subsample=self.subsample,
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
                model, acc, f1, cm = self.create_and_measure(df)
                self.models[df_name] = model
                self.accuracies[df_name] = acc
                self.f1[df_name] = f1
                self.confusion_matrix[df_name] = cm

                # Print the results, if applicable
                self.print_results(self.name.lower(), acc, f1, df_name)
            except ValueError:
                self.print("------------------------------------------")
                self.print(
                    "Error: training data is not big enough for this subset"
                )
                self.print("------------------------------------------")

        return self.accuracies, self.f1, self.confusion_matrix

    def __repr__(self):
        """
        Overwrites default string representation of Model.

        :return string representation of Model
        """

        parameters = [
            self.max_features,
            self.test_size,
            self.max_depth,
            self.n_estimators,
            self.eta,
            self.subsample,
        ]

        parameter_names = [
            "Maximum Features",
            "Test Size",
            "Maximum Depth",
            "Number of Trees",
            "Learning Rate",
            "Subsample",
        ]

        return_str = f"MODEL TYPE: {self.name}\n"
        return_str += "PARAMETERS: \n"
        for parameter, name in zip(parameters, parameter_names):
            return_str += f"\t{name}: {str(parameter)}\n"

        return_str += "ACCURACIES: \n"
        for name, acc in self.accuracies.items():
            return_str += f"\t{name}: {str(acc)}\n"

        return_str += "F1 SCORES: "
        for name, f1 in self.f1.items():
            return_str += f"\n\t{name}: {str(f1)}"

        return return_str
