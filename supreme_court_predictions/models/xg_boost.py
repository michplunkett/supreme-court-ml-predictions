"""
This XGBoost runs a gradient boosted tree model 
on utterance data from the Supreme Court dataset. This class aims to predict
the results of a case based on the text learned from utterances. 
"""

# General imports
import pandas as pd
import xgboost as xgb

# Sklearn imports
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Local imports
from supreme_court_predictions.util.files import get_full_data_pathway

# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import GridSearchCV


class XGBoost:
    """
    A class that runs logistic regression on aggregated utterance and cases data
    from the Supreme Court dataset.
    """

    def __init__(self):
        self.local_path = get_full_data_pathway("processed/")

        self.total_utterances = pd.read_pickle(
            self.local_path + "case_aggregations.p"
        )
        self.advocate_utterances = pd.read_pickle(
            self.local_path + "advocate_aggregations.p"
        )
        self.adversary_utterances = pd.read_pickle(
            self.local_path + "adversary_aggregations.p"
        )
        self.judge_utterances = pd.read_pickle(
            self.local_path + "judge_aggregations.p"
        )

        self.run_regression()

    def xg_boost(self, df):
        """
        Perform logistic regression on the given dataframe of utterance data.
        It regresses on the entire dataset and regresses for judges, advocates,
        and adversaries.

        :param
            df (pd.DataFrame): DataFrame containing utterance data

        :return
            float: Accuracy score of the logistic regression model
        """
        vectorizer = CountVectorizer(analyzer="word", max_features=5000)
        vectorize_document = df.loc[:, "tokens"].apply(" ".join)
        print("Creating bag of words")
        bag_of_words_x = vectorizer.fit_transform(vectorize_document)
        bag_of_words_y = df.loc[:, "win_side"]

        X_train, X_test, y_train, y_test = train_test_split(
            bag_of_words_x,
            bag_of_words_y,
            test_size=0.20,
            random_state=123,
            stratify=bag_of_words_y,
        )

        print("Starting the XGBoost model")
        xgb_model = xgb.XGBClassifier(
            max_depth=7,
            n_estimators=300,
            objective="binary:logistic",
            random_state=1,
            tree_method="gpu_hist",
            predictor="gpu_predictor",
        )

        # Fit the classifier on the training data
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)

        return accuracy_score(y_true=y_test, y_pred=y_pred)

    def run_xgboost(self):
        """
        Run logistic regression on each type of aggregated utterance data.
        It regresses on the entire dataset and regresses for judges, advocates,
        and adversaries.
        """

        dfs = [
            ("total_utterances", self.total_utterances),
            ("judge_utterances", self.judge_utterances),
            ("advocate_utterances", self.advocate_utterances),
            ("adversary_utterances", self.adversary_utterances),
        ]

        for df_name, df in dfs:
            try:
                print("------------------------------------------")
                print(f"Running xgboost on {df_name}...")
                acc = self.xg_boost(df)
                print(f"Accuracy score: {acc}")
                print("------------------------------------------")
            except ValueError:
                print("------------------------------------------")
                print("Error: training data is not big enough for this subset")
                print("------------------------------------------")
