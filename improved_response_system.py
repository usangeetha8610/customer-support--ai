import pandas as pd
import pickle
import math
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==================================================
# FILES
# ==================================================

MODEL_FILE = "apple_support_naive_bayes_model.pkl"
PAIR_FILE = "apple_support_pairs.csv"


# ==================================================
# LOAD NAIVE BAYES MODEL
# ==================================================

print("Loading AI intent model...")

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
# INTENT PREDICTION
# ==================================================

def predict_intent(text):

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

            count = word_counts[intent].get(
                word,
                0
            )

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
# LOAD APPLESUPPORT DATA
# ==================================================

print("Loading AppleSupport response pairs...")

df = pd.read_csv(
    PAIR_FILE
)

print(
    "Total response pairs:",
    len(df)
)


# ==================================================
# CLEAN DATA
# ==================================================

df = df[
    df["customer_message"].notna()
    &
    df["applesupport_response"].notna()
].copy()

df["customer_message"] = (
    df["customer_message"]
    .astype(str)
    .str.strip()
)

df["applesupport_response"] = (
    df["applesupport_response"]
    .astype(str)
    .str.strip()
)

df = df[
    (df["customer_message"] != "")
    &
    (df["applesupport_response"] != "")
].copy()


print(
    "Usable response pairs:",
    len(df)
)


# ==================================================
# TF-IDF VECTORIZER
# ==================================================

print("\nBuilding TF-IDF model...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_features=50000
)


# ==================================================
# CREATE TF-IDF MATRIX
# ==================================================

customer_messages = (
    df["customer_message"]
    .tolist()
)

tfidf_matrix = vectorizer.fit_transform(
    customer_messages
)

print(
    "TF-IDF matrix created."
)

print(
    "Number of features:",
    len(vectorizer.get_feature_names_out())
)


# ==================================================
# FIND BEST RESPONSE
# ==================================================

def find_best_response(customer_message):

    query_vector = vectorizer.transform(
        [customer_message]
    )

    similarities = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    best_index = similarities.argmax()

    best_score = similarities[
        best_index
    ]

    best_response = df.iloc[
        best_index
    ]["applesupport_response"]

    matched_customer = df.iloc[
        best_index
    ]["customer_message"]

    return (
        best_response,
        matched_customer,
        best_score
    )


# ==================================================
# START AI SYSTEM
# ==================================================

print("\n======================================")
print("IMPROVED AI CUSTOMER SUPPORT SYSTEM")
print("======================================")

print(
    "Enter a customer problem."
)

print(
    "Type 'exit' to stop."
)


# ==================================================
# INTERACTIVE SYSTEM
# ==================================================

while True:

    customer_message = input(
        "\nCustomer: "
    )


    # ----------------------------------------------
    # EXIT
    # ----------------------------------------------

    if customer_message.lower() == "exit":

        print(
            "\nAI Customer Support stopped."
        )

        break


    # ----------------------------------------------
    # EMPTY INPUT
    # ----------------------------------------------

    if not customer_message.strip():

        print(
            "Please enter a customer problem."
        )

        continue


    # ----------------------------------------------
    # INTENT
    # ----------------------------------------------

    intent = predict_intent(
        customer_message
    )


    # ----------------------------------------------
    # RESPONSE MATCHING
    # ----------------------------------------------

    response, matched_customer, score = (
        find_best_response(
            customer_message
        )
    )


    # ----------------------------------------------
    # DISPLAY
    # ----------------------------------------------

    print(
        "\n--------------------------------------"
    )

    print(
        "CUSTOMER MESSAGE:"
    )

    print(
        customer_message
    )

    print(
        "\nAI PREDICTED INTENT:"
    )

    print(
        intent
    )

    print(
        "\nMOST SIMILAR HISTORICAL CUSTOMER:"
    )

    print(
        matched_customer
    )

    print(
        "\nCOSINE SIMILARITY:"
    )

    print(
        round(score, 3)
    )

    print(
        "\nAPPLE SUPPORT RESPONSE:"
    )

    print(
        response
    )

    print(
        "--------------------------------------"
    )