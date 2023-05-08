"""
This file contains the class that is the basis for all models in this package.
"""
from abc import ABC, abstractmethod


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
        :return: Float of some accuracy measurement.
        """

    @abstractmethod
    def run(self):
        """
        Runs the model on its respective data.
        """
