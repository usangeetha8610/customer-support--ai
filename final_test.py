import pickle
import math
import re


MODEL_FILE = "apple_support_naive_bayes_model.pkl"


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading original trained model...")

with open(MODEL_FILE, "rb") as file:
    model = pickle.load(file)


classes = model["classes"]
class_probability = model["class_probability"]
word_counts = model["word_counts"]
total_words = model["total_words"]
vocabulary_size = model["vocabulary_size"]


# ==========================================
# TOKENIZER
# ==========================================

def tokenize(text):

    text = str(text).lower()

    return re.findall(
        r"[a-zA-Z0-9]+",
        text
    )


# ==========================================
# PREDICTION
# ==========================================

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


# ==========================================
# TEST CASES
# ==========================================

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

    "My iPhone is locked and I cannot access it",

    "My keyboard is typing the wrong letters",

    "My camera is not working",

    "Bluetooth is not connecting",

    "I forgot my Apple ID password",

    "My iPhone keeps restarting after the update"

]


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\n======================================")
print("FINAL AI INTENT TEST")
print("======================================")


for number, message in enumerate(
    test_messages,
    start=1
):

    prediction = predict(message)

    print(
        f"\n{number}. Customer:"
    )

    print(
        message
    )

    print(
        "Predicted Intent:",
        prediction
    )


# ==========================================
# INTERACTIVE TEST
# ==========================================

print("\n======================================")
print("INTERACTIVE AI TEST")
print("======================================")

print(
    "Enter a customer problem."
)

print(
    "Type 'exit' to stop."
)


while True:

    message = input(
        "\nCustomer: "
    )

    if message.lower() == "exit":

        print(
            "Program stopped."
        )

        break

    if message.strip() == "":

        print(
            "Please enter a message."
        )

        continue

    prediction = predict(
        message
    )

    print(
        "Predicted Intent:",
        prediction
    )