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

        :param pandas.DataFrame df: A dataframe used to create the model.
        :param function accuracy_measure: A function that is used to measure
        accuracy on the model.
        :return: Float of some accuracy measurement.
        """
        _, y_test, y_pred = self.create(df)

        return accuracy_measure(y_true=y_test, y_pred=y_pred)

    @abstractmethod
    def run(self):
        """
        Runs the model on its respective data.
        """

    @abstractmethod
    def __repr__(self):
        """
        Overwrites default string representation
        """

    @staticmethod
    def print_results(model_name="", accuracy_score=None, dataframe_name=None):
        """
        Prints the results of running the model.

        :param str model_name: The name of the model.
        :param list accuracy_score: The accuracy scores generated across for
        the dataframes ran in the model.
        :param list dataframe_name: Name of the dataframe used to create the
        model.
        """
        print("------------------------------------------")
        print(f"Running a {model_name} on {dataframe_name}...")
        print(f"Accuracy score: {accuracy_score}")
        print("------------------------------------------")
