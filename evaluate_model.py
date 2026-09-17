import pandas as pd
import pickle
import math
import re
from collections import defaultdict


# ==================================================
# SETTINGS
# ==================================================

DATASET_FILE = "apple_support_intents.csv"
MODEL_FILE = "apple_support_naive_bayes_model.pkl"


# ==================================================
# LOAD DATASET
# ==================================================

print("Reading dataset...")

df = pd.read_csv(DATASET_FILE)

df = df.dropna(
    subset=["customer_text", "intent"]
).copy()

print("Total records:", len(df))


# ==================================================
# LOAD MODEL
# ==================================================

print("Loading trained model...")

with open(MODEL_FILE, "rb") as file:
    model = pickle.load(file)

classes = model["classes"]
class_probability = model["class_probability"]
word_counts = model["word_counts"]
total_words = model["total_words"]
vocabulary_size = model["vocabulary_size"]


# ==================================================
# TOKENIZER
# ==================================================

def tokenize(text):

    text = str(text).lower()

    return re.findall(
        r"[a-zA-Z0-9]+",
        text
    )


# ==================================================
# PREDICTION
# ==================================================

def predict(text):

    words = set(
        tokenize(text)
    )

    scores = {}

    for intent in classes:

        score = math.log(
            class_probability[intent]
        )

        total = (
            total_words[intent]
            + vocabulary_size
        )

        for word in words:

            count = word_counts[
                intent
            ].get(word, 0)

            probability = (
                (count + 1)
                / total
            )

            score += math.log(
                probability
            )

        scores[intent] = score

    return max(
        scores,
        key=scores.get
    )


# ==================================================
# USE SAME TEST SPLIT
# ==================================================

print("Creating test set...")

# Use deterministic split based on the same seed
import random

random.seed(42)

indices = list(range(len(df)))

random.shuffle(indices)

test_count = int(
    len(indices) * 0.20
)

test_indices = indices[:test_count]

test_df = df.iloc[
    test_indices
].copy()


# ==================================================
# CONFUSION MATRIX
# ==================================================

print("\nEvaluating model...")

matrix = defaultdict(
    lambda: defaultdict(int)
)


for _, row in test_df.iterrows():

    actual = row["intent"]

    predicted = predict(
        row["customer_text"]
    )

    matrix[actual][predicted] += 1


# ==================================================
# ACCURACY
# ==================================================

correct = 0
total = len(test_df)

for actual in matrix:

    for predicted in matrix[actual]:

        if actual == predicted:

            correct += matrix[
                actual
            ][predicted]


accuracy = (
    correct / total
) * 100


print("\n======================================")
print("MODEL EVALUATION")
print("======================================")

print(
    f"Correct: {correct}"
)

print(
    f"Total: {total}"
)

print(
    f"Accuracy: {accuracy:.2f}%"
)


# ==================================================
# CONFUSION DETAILS
# ==================================================

print("\n======================================")
print("MOST COMMON CONFUSIONS")
print("======================================")


confusions = []

for actual in matrix:

    for predicted in matrix[actual]:

        if actual != predicted:

            count = matrix[
                actual
            ][predicted]

            confusions.append(
                (
                    count,
                    actual,
                    predicted
                )
            )


confusions.sort(
    reverse=True
)


for count, actual, predicted in confusions[:20]:

    print(
        f"{count:5} | "
        f"Actual: {actual} "
        f"-> Predicted: {predicted}"
    )


# ==================================================
# PER-INTENT ACCURACY
# ==================================================

print("\n======================================")
print("PER-INTENT ACCURACY")
print("======================================")


for intent in classes:

    actual_total = sum(
        matrix[intent].values()
    )

    correct_total = matrix[
        intent
    ][intent]

    if actual_total > 0:

        percentage = (
            correct_total
            / actual_total
        ) * 100

    else:

        percentage = 0

    print(
        f"{intent:35} "
        f"{percentage:.2f}%"
    )


# ==================================================
# DONE
# ==================================================

print("\n======================================")
print("EVALUATION COMPLETED")
print("======================================")