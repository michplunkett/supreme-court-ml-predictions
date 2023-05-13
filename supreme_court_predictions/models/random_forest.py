import os.path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from supreme_court_predictions.util.contants import SEED_CONSTANT
from supreme_court_predictions.util.files import get_full_data_pathway


class RandomForest:
    """
    This class runs sklearn random forest on aggregated utterance
    from the Supreme Court dataset.
    """

    def __init__(
        self,
        dfs=[
            "case_aggregations.p",
            "judge_aggregations.p",
            "advocate_aggregations.p",
            "adversary_aggregations.p",
        ],
        max_features=5000,
        test_size=0.20,
        num_trees=100,
        max_depth=None,
        print_results=True,
    ):
        self.local_path = get_full_data_pathway("processed/")
        self.max_features = max_features
        self.test_size = test_size
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.print = print_results
        self.name = "Random Forest"
        self.accuracies = []

        # Dataframes and df names to run models against
        self.dataframes = []
        self.dataframe_names = []

        for df in dfs:
            # make sure its a file name
            if os.path.isfile(self.local_path + df):
                # pickle file
                if (df.split(".")[-1]) == "p":
                    self.dataframes.append(pd.read_pickle(self.local_path + df))

                # csv file
                else:
                    self.dataframes.append(pd.read_csv(self.local_path + df))

                # add name of file
                self.dataframe_names.append(df.split(".")[0])

    def create(self, df):
        """
        Creates and runs a random forest on the given dataframe of
        utterance data.

        :param df: DataFrame containing utterance data

        :return (forest, y_test, y_pred): A tuple that contains the
        forest model, test y-data, the predicted y-data.
        """
        if self.print:
            print("Creating bag of words")
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

        if self.print:
            print("Starting the Random Forest")
        forest = RandomForestClassifier(
            num_trees=self.num_trees, max_depth=self.max_depth
        )

        # Fit the classifier on the training data
        forest.fit(X_train, y_train)
        y_pred = forest.predict(X_test)

        return forest, y_test, y_pred

    def run(self):
        """
        Runs the create function on each type of aggregated utterance.
        """

        for df in self.dataframes:
            try:
                # accuracy score method to be used later from ABC
                _, y_test, y_pred = self.create(df)
                acc = accuracy_score(y_true=y_test, y_pred=y_pred)
                self.accuracies.append(acc)
            except ValueError:
                print("------------------------------------------")
                print("Error: training data is not big enough for this subset")
                print("------------------------------------------")

        # Print the results, if applicable
        if self.print:
            # This will also be later used from ABC
            for acc, df_name in zip(self.accuracies, self.dataframe_names):
                print("------------------------------------------")
                print(f"Running a {self.name.lower()} on {df_name}...")
                print(f"Accuracy score: {acc}")
                print("------------------------------------------")

        return self.accuracies

    def __repr__(self):
        """
        Overwrites default string representation of Model

        :return string representation of Model
        """

        parameters = [self.max_features, self.test_size, self.max_iter]
        parameter_names = [
            "Maximum Features:",
            "Test Size:",
            "Maximum Iterations:",
        ]

        return_str = f"MODEL TYPE: {self.name}\n"
        return_str += "PARAMETERS: \n"
        for parameter, name in zip(parameters, parameter_names):
            return_str += f"\t{name}: {str(parameter)}\n"

        return_str += "ACCURACIES: "
        for name, acc in zip(self.dataframe_names, self.accuracies):
            return_str += f"\n\t{name}: {str(acc)}"

        return return_str
