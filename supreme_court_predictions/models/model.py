"""
This file contains the class that is the basis for all models in this package.
"""

from abc import ABC, abstractmethod

from sklearn.metrics import accuracy_score


class Model(ABC):
    """
    This class sets basis for what makes up a model.
    """

    @staticmethod
    @abstractmethod
    def create(df):
        """
        Creates the model and returns an accuracy score.

        :param df: A dataframe used to create the model.
        :return: A sklearn model.
        """

    def create_and_measure(self, df):
        """
        Takes in a dataframe and returns the applicable accuracy measurement.

        :param df: A dataframe used to create the model.
        :return: Float of some accuracy measurement.
        """
        _, y_test, y_pred = self.create(df)

        return accuracy_score(y_true=y_test, y_pred=y_pred)

    @abstractmethod
    def run(self):
        """
        Runs the model on its respective data.
        """
