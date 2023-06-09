"""
This file contains the class that is the basis for all models in this package.
"""
import time
from abc import ABC, abstractmethod

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from supreme_court_predictions.util.functions import debug_print


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

    def create_and_measure(self, df):
        """
        Takes in a dataframe and returns the applicable accuracy measurement.

        :param pandas.DataFrame df: A dataframe used to create the model.
        :param function accuracy_measure: A function that is used to measure
        accuracy on the model.
        :return: Float of accuracy.
        :return: Float of F1 score.
        :return: Confusion matrix for model
        """

        start = time.perf_counter()
        model, y_test, y_pred = self.create(df)
        stop = time.perf_counter()

        return (
            model,
            accuracy_score(y_true=y_test, y_pred=y_pred),
            f1_score(y_true=y_test, y_pred=y_pred),
            confusion_matrix(y_true=y_test, y_pred=y_pred),
            stop - start,
        )

    @abstractmethod
    def run(self):
        """
        Runs the model on its respective data.
        """

    def print(self, message: str):
        """
        Handles the decision to print a message to standard out, and does so
        if the application is in debug mode.

        :param message: The message to be printed.
        """
        debug_print(message, self.debug_mode)

    @abstractmethod
    def __repr__(self):
        """
        Overwrites default string representation.
        """

    def print_results(
        self,
        model_name: str = "",
        acc_score: list = None,
        f1: float = None,
        execution_time: float = None,
        dataframe_name: str = None,
    ):
        """
        Prints the results of running the model.

        :param str model_name: The name of the model.
        :param list acc_score: The accuracy scores generated across for
            the dataframes ran in the model.
        :param float f1: The F1 score for the model.
        :param float execution_time: The time it takes to execute the model.
        :param str dataframe_name: Name of the dataframe used to create the
            model.
        """
        if self.debug_mode:
            print("------------------------------------------")
            print(f"Running a {model_name} on {dataframe_name}...")
            print(f"Accuracy score: {acc_score}")
            print(f"F1 score: {f1:04f}")
            print(f"Execution time: {execution_time:04f} seconds")
            print("------------------------------------------")
