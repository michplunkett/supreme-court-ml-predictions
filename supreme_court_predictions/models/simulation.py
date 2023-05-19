import os
import pandas as pd

from supreme_court_predictions.util.constants import SEED_CONSTANT
from supreme_court_predictions.util.functions import get_full_data_pathway

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from supreme_court_predictions.models.service import (
    run_linear_regression,
    run_random_forest,
    run_xg_boost,
)

"""
TODO: 1. Aggregate the prediction for each judge for each case. 
    2. Count them up for majority rule and decide if you win or lose case
    3. Compare this prediction with actual win_side for accuracy matrix.
"""

# run_linear_regression()
local_path = get_full_data_pathway("clean_convokit/")
if os.path.isfile(local_path + "utterances_df.p"):
    # Use the correct file reading function
    simulation_utterance = pd.read_pickle(local_path + "utterances_df.p")
    simulation_utterance = simulation_utterance.loc[simulation_utterance.loc[:, 'speaker_type'] == 'J', :]
    simulation_utterance = simulation_utterance.groupby(['case_id', 'speaker'])['tokens'].apply(sum).reset_index()

if os.path.isfile(local_path + "voters_df.csv"):
    voters = pd.read_csv(local_path + "voters_df.csv")
    

merged_df = pd.merge(simulation_utterance, voters, left_on=['case_id', 'speaker'], right_on=['case_id', 'voter'])

# for each speaker column, train on tokens as x and vote as y. 
# logistic regression 
vectorizer = CountVectorizer(analyzer='word', max_features=5000)
merged_df['tokens'] = merged_df['tokens'].apply(' '.join)
bag_of_words = vectorizer.fit_transform(merged_df['tokens'])

# Initialize dictionaries to store models and scores
models = {}
accuracies = {}
f1_scores = {}

# Iterate over each unique speaker
for speaker in merged_df['speaker'].unique():
    # Subset the data for the current speaker
    speaker_df = merged_df[merged_df['speaker'] == speaker]

    # If there's only one instance for a class, predict it as the single class
    if len(speaker_df['vote'].unique()) == 1:
        default_class = speaker_df['vote'].iloc[0]
        accuracies[speaker] = 1.0
        f1_scores[speaker] = 1.0
    else:
        try:
            # Split the data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                bag_of_words[speaker_df.index],
                speaker_df['vote'],
                test_size=0.2,
                random_state=42,
                stratify=speaker_df['vote']
            )

            # Fit the logistic regression model if there are more than one instance of each class
            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(X_train, y_train)

            # Make predictions for the test set
            y_pred = lr.predict(X_test)

            # Calculate the accuracy and F1-score
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')

            # Store the model and scores in the dictionaries
            models[speaker] = lr
            accuracies[speaker] = acc
            f1_scores[speaker] = f1

            # Predict the single instance of the class
            if len(y_test) == 0:
                y_pred_single = lr.predict(X_test[:1])
                # check if the judge X_test
            else:
                y_pred_single = default_class if y_pred[0] != y_pred[1] else y_pred[0]

            # Print the prediction
            print("Prediction for single instance:", y_pred_single)
        except ValueError:
            print("Dataset too small")

print('Models:', models)
print('Accuracies:', accuracies)
print('F1 scores:', f1_scores)