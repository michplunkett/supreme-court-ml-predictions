import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from supreme_court_predictions.util.files import get_full_pathway


class Logistic:
    def __init__(self):
        self.local_path = get_full_pathway(
            "/supreme_court_predictions/data/clean_convokit/"
        )

        self.utterances_df = pd.read_pickle(self.local_path + "utterances_df.p")
        self.regress_logistic()

    def regress_logistic(self):
        vectorizer = CountVectorizer()
        vectorize_document = self.utterances_df.loc[:, "tokens"].apply(" ".join)
        print("Creating bag of words")
        bag_of_words_x = vectorizer.fit_transform(vectorize_document)

        # TODO: after Chay's merge of dataframe, this can be updated.
        bag_of_words_y = np.random.randint(0, 2, len(vectorize_document))

        X_train, X_test, y_train, y_test = train_test_split(
            bag_of_words_x, bag_of_words_y, test_size=0.25, random_state=123
        )

        print("Logistic Regression underway \n")
        regressor = LogisticRegression()

        # Fit the classifier on the training data
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)

        print("Prediction done: ", y_pred)
