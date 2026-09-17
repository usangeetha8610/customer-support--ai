import pandas as pd
import math
import re
import pickle
import random
from collections import Counter, defaultdict


# ==================================================
# SETTINGS
# ==================================================

DATASET_FILE = "apple_support_intents_improved.csv"
MODEL_FILE = "apple_support_naive_bayes_improved.pkl"

TEST_SIZE = 0.20
RANDOM_SEED = 42


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
# READ DATASET
# ==================================================

print("Reading improved intent dataset...")

df = pd.read_csv(DATASET_FILE)

df = df.dropna(
    subset=["customer_text", "intent"]
).copy()

print(
    "Total records:",
    len(df)
)


# ==================================================
# CLEAN DATA
# ==================================================

df["customer_text"] = (
    df["customer_text"]
    .astype(str)
    .str.strip()
)

df = df[
    df["customer_text"] != ""
].copy()

print(
    "Records after cleaning:",
    len(df)
)


# ==================================================
# TRAIN / TEST SPLIT
# ==================================================

indices = list(
    range(len(df))
)

random.seed(
    RANDOM_SEED
)

random.shuffle(
    indices
)

test_count = int(
    len(indices) * TEST_SIZE
)

test_indices = indices[
    :test_count
]

train_indices = indices[
    test_count:
]

train_df = df.iloc[
    train_indices
].copy()

test_df = df.iloc[
    test_indices
].copy()


print(
    "\nTraining records:",
    len(train_df)
)

print(
    "Testing records:",
    len(test_df)
)


# ==================================================
# INTENTS
# ==================================================

classes = sorted(
    train_df["intent"].unique()
)

print(
    "\nNumber of intents:",
    len(classes)
)


# ==================================================
# WORD COUNTS
# ==================================================

word_counts = defaultdict(
    Counter
)

total_words = Counter()

class_counts = Counter()


print(
    "\nBuilding vocabulary..."
)


for _, row in train_df.iterrows():

    intent = row["intent"]

    words = tokenize(
        row["customer_text"]
    )

    word_counts[
        intent
    ].update(words)

    total_words[
        intent
    ] += len(words)

    class_counts[
        intent
    ] += 1


# ==================================================
# VOCABULARY
# ==================================================

vocabulary = set()

for intent in classes:

    vocabulary.update(
        word_counts[intent].keys()
    )

vocabulary_size = len(
    vocabulary
)

print(
    "Vocabulary size:",
    vocabulary_size
)


# ==================================================
# CLASS PROBABILITIES
# ==================================================

total_training_records = len(
    train_df
)

class_probability = {}

for intent in classes:

    class_probability[
        intent
    ] = (
        class_counts[intent]
        / total_training_records
    )


# ==================================================
# TRAIN NAIVE BAYES
# ==================================================

print(
    "\nTraining Naive Bayes model..."
)


# ==================================================
# PREDICTION FUNCTION
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

        denominator = (
            total_words[intent]
            + vocabulary_size
        )

        for word in words:

            count = word_counts[
                intent
            ].get(word, 0)

            probability = (
                count + 1
            ) / denominator

            score += math.log(
                probability
            )

        scores[intent] = score

    return max(
        scores,
        key=scores.get
    )


# ==================================================
# TEST MODEL
# ==================================================

print(
    "\nTesting model..."
)

correct = 0

for _, row in test_df.iterrows():

    actual = row["intent"]

    predicted = predict(
        row["customer_text"]
    )

    if actual == predicted:

        correct += 1


accuracy = (
    correct
    / len(test_df)
) * 100


# ==================================================
# PERFORMANCE
# ==================================================

print(
    "\n======================================"
)

print(
    "IMPROVED MODEL PERFORMANCE"
)

print(
    "======================================"
)

print(
    "Correct predictions:",
    correct
)

print(
    "Total test records:",
    len(test_df)
)

print(
    f"Accuracy: {accuracy:.2f}%"
)


# ==================================================
# SAVE MODEL
# ==================================================

model = {

    "classes": classes,

    "class_probability":
        class_probability,

    "word_counts":
        dict(word_counts),

    "total_words":
        dict(total_words),

    "vocabulary_size":
        vocabulary_size
}


with open(
    MODEL_FILE,
    "wb"
) as file:

    pickle.dump(
        model,
        file
    )


# ==================================================
# SAMPLE PREDICTIONS
# ==================================================

print(
    "\n======================================"
)

print(
    "SAMPLE PREDICTIONS"
)

print(
    "======================================"
)


samples = [

    "My iPhone battery is draining very quickly",

    "My WiFi is not connecting to my iPhone",

    "I cannot sign into my Apple ID",

    "My iCloud backup is not working",

    "My iPhone screen is not responding",

    "The latest iOS update is causing problems",

    "Apple Music is not playing my songs",

    "My App Store application keeps crashing",

    "I cannot charge my iPhone",

    "My iPhone is locked and I cannot access it"

]


for message in samples:

    prediction = predict(
        message
    )

    print(
        "\nCustomer:",
        message
    )

    print(
        "Predicted Intent:",
        prediction
    )


# ==================================================
# DONE
# ==================================================

print(
    "\n======================================"
)

print(
    "IMPROVED MODEL TRAINING COMPLETED"
)

print(
    "======================================"
)

print(
    "Model file:",
    MODEL_FILE
)

print(
    f"Accuracy: {accuracy:.2f}%"
)