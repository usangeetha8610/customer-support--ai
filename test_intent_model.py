import pickle
import math
import re
from collections import Counter


# ==================================================
# LOAD TRAINED MODEL
# ==================================================

MODEL_FILE = "apple_support_naive_bayes_model.pkl"

print("Loading trained model...")

with open(MODEL_FILE, "rb") as file:
    model = pickle.load(file)

classes = model["classes"]
class_probability = model["class_probability"]
word_counts = model["word_counts"]
total_words = model["total_words"]
vocabulary = model["vocabulary"]
vocabulary_size = model["vocabulary_size"]


# ==================================================
# TOKENIZER
# ==================================================

def tokenize(text):

    text = text.lower()

    words = re.findall(
        r"[a-zA-Z0-9]+",
        text
    )

    return words


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

    # Sort intents by score
    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked


# ==================================================
# TEST EXAMPLES
# ==================================================

test_messages = [

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


# ==================================================
# DISPLAY PREDICTIONS
# ==================================================

print("\n======================================")
print("AI INTENT PREDICTIONS")
print("======================================")


for message in test_messages:

    predictions = predict(message)

    best_intent = predictions[0][0]

    print("\nCustomer:")
    print(message)

    print(
        "Predicted Intent:",
        best_intent
    )


# ==================================================
# INTERACTIVE TEST
# ==================================================

print("\n======================================")
print("INTERACTIVE TEST")
print("======================================")

print(
    "Type a customer message."
)

print(
    "Type 'exit' to stop."
)


while True:

    message = input(
        "\nCustomer: "
    )

    if message.lower() == "exit":

        print("Program stopped.")

        break

    predictions = predict(message)

    best_intent = predictions[0][0]

    print(
        "Predicted Intent:",
        best_intent
    )