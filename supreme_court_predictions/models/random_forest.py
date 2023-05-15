import os.path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from supreme_court_predictions.models.model import Model
from supreme_court_predictions.util.contants import SEED_CONSTANT
from supreme_court_predictions.util.functions import get_full_data_pathway


class RandomForest(Model):
    """
    This class runs sklearn random forest on aggregated utterances
    from the Supreme Court dataset.
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
        max_depth=None,
        max_features=5000,
        num_trees=100,
        test_size=0.20,
    ):
        self.accuracies = []
        self.debug_mode = debug_mode
        self.local_path = get_full_data_pathway("processed/")
        self.max_depth = max_depth
        self.max_features = max_features
        self.name = "Random Forest"
        self.num_trees = num_trees
        self.test_size = test_size

        # Dataframes and df names to run models against
        self.dataframes = []
        self.dataframe_names = []

        for df in dfs:
            # Ensure the file exists
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
        Creates and runs a random forest on the given dataframe of
        utterance data.

        :param df: DataFrame containing utterance data

        :return (forest, y_test, y_pred): A tuple that contains the
        forest model, test y-data, the predicted y-data.
        """
        self.print("Creating bag of words")
        vectorizer = CountVectorizer(
            analyzer="word", max_features=self.max_features
        )
        vectorize_document = df.loc[:, "tokens"].apply(" ".join)
        bag_of_words_x = vectorizer.fit_transform(vectorize_document)

        bag_of_words_y = df.loc[:, "win_side"]

        X_train, X_test, y_train, y_test = train_test_split(
            bag_of_words_x,
            bag_of_words_y,
            test_size=self.test_size,
            random_state=SEED_CONSTANT,
            stratify=bag_of_words_y,
        )

        self.print("Starting the Random Forest")
        forest = RandomForestClassifier(
            n_estimators=self.num_trees, max_depth=self.max_depth
        )

        # Fit the classifier on the training data
        forest.fit(X_train, y_train)
        y_pred = forest.predict(X_test)

        return forest, y_test, y_pred

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
                    "Error: training data is not big enough for this " "subset"
                )
                self.print("------------------------------------------")

        return self.accuracies

    def __repr__(self):
        """
        Overwrites default string representation of Model.

        :return string representation of Model
        """

        parameters = [
            self.max_features,
            self.test_size,
            self.num_trees,
            self.max_depth,
        ]
        parameter_names = [
            "Maximum Features",
            "Test Size",
            "Number of Trees",
            "Maximum Depth",
        ]

        return_str = f"MODEL TYPE: {self.name}\n"
        return_str += "PARAMETERS: \n"
        for parameter, name in zip(parameters, parameter_names):
            return_str += f"\t{name}: {str(parameter)}\n"

        return_str += "ACCURACIES: "
        for name, acc in zip(self.dataframe_names, self.accuracies):
            return_str += f"\n\t{name}: {str(acc)}"

        return return_str
