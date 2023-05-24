from collections import Counter

import pandas as pd
import xgboost as xgb
from scipy.stats import mode
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from supreme_court_predictions.util.constants import SEED_CONSTANT
from supreme_court_predictions.util.functions import (
    debug_print,
    get_full_data_pathway,
)


class Simulate:
    def __init__(
        self,
        debug_mode=False,
        eta=0.3,
        max_depth=5000,
        max_features=5000,
        max_iter=1000,
        num_trees=100,
        subsample=1,
    ):
        self.debug_mode = debug_mode
        self.max_features = max_features
        list_of_models = [
            LogisticRegression(max_iter=max_iter, random_state=SEED_CONSTANT),
            RandomForestClassifier(n_estimators=num_trees, max_depth=max_depth),
            xgb.XGBClassifier(
                max_depth=max_depth,
                n_estimators=num_trees,
                eta=eta,
                subsample=subsample,
                objective="binary:logistic",
                random_state=SEED_CONSTANT,
                tree_method="hist",
                predictor="cpu_predictor",
            ),
        ]
        data_tuple = self.merge_vectorize_data()
        for model in list_of_models:
            self.simulate_model(input_model=model, data_tuple=data_tuple)

    def merge_vectorize_data(self):
        local_path = get_full_data_pathway("clean_convokit/")
        # Use the correct file reading function
        simulation_utterance = pd.read_pickle(local_path + "utterances_df.p")
        simulation_utterance = simulation_utterance.loc[
            simulation_utterance.loc[:, "speaker_type"] == "J", :
        ]
        simulation_utterance = (
            simulation_utterance.groupby(["case_id", "speaker"])["tokens"]
            .apply(sum)
            .reset_index()
        )

        voters = pd.read_csv(local_path + "voters_df.csv")

        merged_df = pd.merge(
            simulation_utterance,
            voters,
            left_on=["case_id", "speaker"],
            right_on=["case_id", "voter"],
        )
        vectorizer = CountVectorizer(
            analyzer="word", max_features=self.max_features
        )
        merged_df["tokens"] = merged_df["tokens"].apply(" ".join)
        bag_of_words = vectorizer.fit_transform(merged_df["tokens"])
        return merged_df, bag_of_words, vectorizer

    def simulate_model(self, input_model, data_tuple):
        merged_df, bag_of_words, vectorizer = data_tuple
        # Initialize dictionaries to store models and scores
        models = {}
        accuracies = {}
        f1_scores = {}
        predictions = {}

        # Iterate over each unique speaker
        for speaker in merged_df["speaker"].unique():
            # Subset the data for the current speaker
            speaker_df = merged_df[merged_df["speaker"] == speaker]
            predictions[speaker] = {}

            # If there's only one instance for a class,
            # predict it as the single class
            if len(speaker_df["vote"].unique()) == 1:
                speaker_df["vote"].iloc[0]
                accuracies[speaker] = 1.0
                f1_scores[speaker] = 1.0

            else:
                try:
                    token_df = pd.DataFrame(
                        bag_of_words.toarray(),
                        columns=vectorizer.get_feature_names_out(),
                    )
                    token_df["case_id"] = merged_df["case_id"].values
                    # Split the data into training and test sets
                    X_train, X_test, y_train, y_test = train_test_split(
                        token_df,
                        merged_df["vote"],
                        test_size=0.2,
                        random_state=42,
                        stratify=merged_df["vote"],
                    )

                except ValueError:
                    debug_print(
                        "Dataset too small for splitting", self.debug_mode
                    )
                    continue

                if len(y_test) != 0:
                    try:
                        # Fit the logistic regression model if there is
                        # more than one instance of each class
                        # Make predictions for the test set
                        case_ids = X_test["case_id"].values

                        X_train = X_train.drop(columns="case_id")
                        X_test = X_test.drop(columns="case_id")

                        # For each speaker column, train on tokens as x
                        # and vote as y
                        input_model.fit(X_train, y_train)
                        y_pred = input_model.predict(X_test)

                        for case_id, pred_speaker_tuple in zip(
                            case_ids, [(pred, speaker) for pred in y_pred]
                        ):
                            predictions[case_id] = predictions.get(
                                case_id, []
                            ) + [pred_speaker_tuple]

                        # Calculate the accuracy and F1-score
                        acc = accuracy_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred, average="macro")

                        # Store the model and scores in the dictionaries
                        models[speaker] = input_model
                        accuracies[speaker] = acc
                        f1_scores[speaker] = f1

                        # Print the prediction
                        print(f"Predicted for judge: {speaker}")
                    except ValueError:
                        debug_print("Prediction Error", self.debug_mode)

        print("Models by judges:", models)
        print("Accuracies by judges:", accuracies)
        print("F1 scores by judges:", f1_scores)

        majority_predictions, actual_values_dict = {}, {}

        # Create dictionary for predictions
        for case_id, pred_speaker_tuples in predictions.items():
            # Extract all predictions for this case_id
            only_predictions = [tup[0] for tup in pred_speaker_tuples]
            counter = Counter(only_predictions)

            if len(counter.most_common(1)):
                majority_predictions[case_id] = counter.most_common(1)[0][0]

        # Create dictionary for actual values
        for case_id, actual_value in zip(case_ids, y_test):
            actual_values_dict[case_id] = actual_values_dict.get(
                case_id, []
            ) + [actual_value]

        actual_values = [
            mode(actual_values_dict[case_id], keepdims=True).mode[0]
            for case_id in majority_predictions.keys()
        ]
        predicted_values = list(majority_predictions.values())

        # Calculate the accuracy
        if len(actual_values) > 0 and len(predicted_values) > 0:
            total_accuracy = accuracy_score(actual_values, predicted_values)
            print("Total Accuracy for All Cases:", total_accuracy)
