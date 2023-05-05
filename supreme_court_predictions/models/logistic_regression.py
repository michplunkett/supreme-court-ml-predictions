"""
TODO: Need file document string
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression as skLR
from sklearn.model_selection import train_test_split

from supreme_court_predictions.util.files import get_full_data_pathway


class LogisticRegression:
    """
    TODO: Need document string
    """

    def __init__(self):
        self.local_path = get_full_data_pathway("clean_convokit/")

        self.utterances_df = pd.read_pickle(self.local_path + "utterances_df.p")
        self.logistic_regression()

    def logistic_regression(self):
        """
        TODO: Need document string
        """
        vectorizer = CountVectorizer()
        vectorize_document = self.utterances_df.loc[:, "tokens"].apply(" ".join)
        print("Creating bag of words")
        bag_of_words_x = vectorizer.fit_transform(vectorize_document)

        # TODO: after Chay's merge of dataframe, this can be updated.
        bag_of_words_y = np.random.randint(0, 2, len(vectorize_document))

        X_train, X_test, y_train, y_test = train_test_split(
            bag_of_words_x, bag_of_words_y, test_size=0.25, random_state=123
        )

        print("Starting the Logistic Regression on utterances")
        regressor = skLR()

        # Fit the classifier on the training data
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)

        print("Prediction done: ", y_pred)
