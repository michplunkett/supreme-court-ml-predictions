"""
This file contains the class that is the basis for all models in this package.
"""

from abc import ABC, abstractmethod


class Model(ABC):
    """
    This class sets basis for what makes up a model.
    """

    @abstractmethod
    def create(self, df):
        """
        Creates the model and returns an accuracy score.

        :param df: A dataframe used to create the model.
        :return: A sklearn model.
        """

    def create_and_measure(self, df, accuracy_measure):
        """
        Takes in a dataframe and returns the applicable accuracy measurement.

        :param df: A dataframe used to create the model.
        :param accuracy_measure: A function that is used to measure accuracy
        on the model.
        :return: Float of some accuracy measurement.
        """
        _, y_test, y_pred = self.create(df)

        return accuracy_measure(y_true=y_test, y_pred=y_pred)

    @abstractmethod
    def run(self):
        """
        Runs the model on its respective data.
        """

    @staticmethod
    def print_results(model_name="", accuracy_scores=[], dataframe_names=[]):
        """
        Prints the results of running the model.

        :param model_name (str): The name of the model.
        :param accuracy_score (list): The accuracy scores generated across for
                                      the dataframes ran in the model.
        :param dataframe_names (list): Name of the dataframes the model was ran
                                       against.
        """

        assert len(accuracy_scores) == len(dataframe_names)

        for acc, df_name in zip(accuracy_scores, dataframe_names):
            print("------------------------------------------")
            print(f"Running a {model_name} on {df_name}...")
            print(f"Accuracy score: {acc}")
            print("------------------------------------------")
