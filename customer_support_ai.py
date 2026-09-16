import pandas as pd
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# FILE NAMES
# ============================================================

MODEL_FILE = "apple_support_naive_bayes_model.pkl"
RESPONSE_FILE = "apple_support_pairs.csv"


# ============================================================
# LOAD AI MODEL
# ============================================================

print("Loading AI model...")

model_data = joblib.load(MODEL_FILE)

vectorizer = None
model = None

# Handle dictionary-based saved model
if isinstance(model_data, dict):

    vectorizer = model_data.get("vectorizer")
    model = model_data.get("model")

    if model is None:
        model = model_data.get("classifier")

# Handle model saved directly
else:
    model = model_data


# ============================================================
# LOAD RESPONSE PAIRS
# ============================================================

print("Loading AppleSupport response pairs...")

df = pd.read_csv(RESPONSE_FILE)

print("Total response pairs:", len(df))

print()
print("Dataset columns:")
print(list(df.columns))


# ============================================================
# FIND CORRECT COLUMNS
# ============================================================

customer_column = None
response_column = None


possible_customer_columns = [
    "customer_message",
    "customer_tweet",
    "customer_text"
]

possible_response_columns = [
    "applesupport_response",
    "response",
    "support_response"
]


for column in possible_customer_columns:

    if column in df.columns:
        customer_column = column
        break


for column in possible_response_columns:

    if column in df.columns:
        response_column = column
        break


if customer_column is None:

    raise ValueError(
        "Customer message column was not found in apple_support_pairs.csv"
    )


if response_column is None:

    raise ValueError(
        "AppleSupport response column was not found in apple_support_pairs.csv"
    )


print("Customer column:", customer_column)
print("Response column:", response_column)


# ============================================================
# CLEAN RESPONSE DATA
# ============================================================

df = df.dropna(
    subset=[
        customer_column,
        response_column
    ]
)

df[customer_column] = df[customer_column].astype(str)
df[response_column] = df[response_column].astype(str)


df = df[
    (df[customer_column].str.strip() != "") &
    (df[response_column].str.strip() != "")
]


# Remove duplicate pairs

df = df.drop_duplicates(
    subset=[
        customer_column,
        response_column
    ]
).reset_index(drop=True)


print("Usable response pairs:", len(df))


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Remove Twitter usernames
    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    # Keep letters and numbers
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


df["clean_customer"] = df[
    customer_column
].apply(clean_text)


# ============================================================
# BUILD TF-IDF DATABASE
# ============================================================

print()
print("Building TF-IDF database...")


tfidf_vectorizer = TfidfVectorizer(

    # Single words + two-word phrases
    ngram_range=(1, 2),

    # Ignore extremely rare words
    min_df=2,

    # Limit vocabulary size
    max_features=100000,

    # Better handling of repeated words
    sublinear_tf=True
)


tfidf_matrix = tfidf_vectorizer.fit_transform(
    df["clean_customer"]
)


print("TF-IDF database ready.")


# ============================================================
# KEYWORD INTENT RULES
# ============================================================

KEYWORD_RULES = {

    "Battery & Charging": [

        "battery",
        "battery drain",
        "battery draining",
        "battery life",
        "charge",
        "charging",
        "charger",
        "won't charge",
        "wont charge",
        "not charging",
        "charging problem"

    ],


    "Connectivity": [

        "wifi",
        "wi-fi",
        "bluetooth",
        "internet",
        "network",
        "hotspot",
        "cellular",
        "mobile data",
        "connection",
        "connect",
        "connecting",
        "not connecting"

    ],


    "Apple ID & Account": [

        "apple id",
        "appleid",
        "sign into apple",
        "sign in apple",
        "sign into my apple id",
        "login",
        "log in",
        "forgot password",
        "apple id password",
        "account password"

    ],


    "iCloud & Backup": [

        "icloud",
        "icloud backup",
        "backup",
        "restore backup",
        "icloud storage"

    ],


    "Music & Media": [

        "apple music",
        "itunes",
        "music",
        "song",
        "songs",
        "playlist",
        "album",
        "video",
        "podcast",
        "apple tv"

    ],


    "Payments & Subscriptions": [

        "payment",
        "paid",
        "billing",
        "bill",
        "subscription",
        "subscribe",
        "refund",
        "purchase",
        "credit",
        "charged",
        "charge for"

    ],


    "Display, Keyboard & Camera": [

        "screen",
        "display",
        "touchscreen",
        "touch screen",
        "keyboard",
        "typing",
        "type",
        "wrong letters",
        "camera",
        "camera not working",
        "flash"

    ],


    "Messages & Communication": [

        "imessage",
        "i message",
        "message",
        "messages",
        "sms",
        "text message",
        "facetime",
        "mail",
        "email"

    ],


    "iOS Update & Software Issues": [

        "ios update",
        "ios",
        "software update",
        "software",
        "update",
        "updated",
        "upgraded",
        "upgrade",
        "after update",
        "after upgrading"

    ],


    "Security & Access": [

        "passcode",
        "security",
        "locked",
        "lock screen",
        "fingerprint",
        "touch id",
        "face id",
        "cannot access",
        "can't access",
        "access denied"

    ],


    "Apps & App Problems": [

        "app",
        "application",
        "app store",
        "crashing",
        "app crash",
        "apps crashing",
        "application crashing",
        "maps",
        "spotify"

    ],


    "General Device Issues": [

        "iphone",
        "ipad",
        "macbook",
        "mac",
        "device",
        "phone",
        "freezing",
        "frozen",
        "restarting",
        "restart",
        "won't turn on",
        "wont turn on",
        "not turning on"

    ]
}


# ============================================================
# KEYWORD INTENT DETECTION
# ============================================================

def keyword_intent(text):

    text_lower = text.lower()

    scores = {}


    for intent, keywords in KEYWORD_RULES.items():

        score = 0

        for keyword in keywords:

            if keyword in text_lower:

                # Multi-word keywords get higher priority
                if " " in keyword:
                    score += 3

                else:
                    score += 1


        scores[intent] = score


    best_intent = max(
        scores,
        key=scores.get
    )

    best_score = scores[best_intent]


    if best_score == 0:

        return None, 0


    return best_intent, best_score


# ============================================================
# PREDICT CUSTOMER INTENT
# ============================================================

def predict_intent(text):

    keyword_result, keyword_score = keyword_intent(text)


    # --------------------------------------------------------
    # Strong keyword rules get priority
    # --------------------------------------------------------

    if keyword_result is not None:

        if keyword_score >= 2:

            return keyword_result


    # --------------------------------------------------------
    # Machine learning prediction
    # --------------------------------------------------------

    try:

        cleaned = clean_text(text)


        # If model has a vectorizer
        if vectorizer is not None:

            X = vectorizer.transform(
                [cleaned]
            )

            prediction = model.predict(X)[0]

            return prediction


    except Exception:

        pass


    # --------------------------------------------------------
    # Fallback to keyword result
    # --------------------------------------------------------

    if keyword_result is not None:

        return keyword_result


    return "Other"


# ============================================================
# FIND BEST RESPONSE
# ============================================================

def find_best_response(
    customer_message,
    intent
):

    cleaned_message = clean_text(
        customer_message
    )


    # Convert customer message to TF-IDF
    query_vector = tfidf_vectorizer.transform(
        [cleaned_message]
    )


    # Calculate cosine similarity
    similarities = cosine_similarity(
        query_vector,
        tfidf_matrix
    )[0]


    # Get top 100 matching messages
    top_indices = similarities.argsort()[-100:][::-1]


    # Keywords belonging to predicted intent
    intent_keywords = KEYWORD_RULES.get(
        intent,
        []
    )


    best_index = None
    best_score = -1


    # --------------------------------------------------------
    # Compare top matching responses
    # --------------------------------------------------------

    for index in top_indices:

        original_message = df.iloc[index][
            customer_column
        ]

        original_clean = clean_text(
            original_message
        )


        similarity_score = similarities[index]


        # Keyword bonus
        keyword_bonus = 0


        for keyword in intent_keywords:

            if (
                keyword in cleaned_message
                and keyword in original_clean
            ):

                keyword_bonus += 0.05


        # Final score
        final_score = (
            similarity_score
            + keyword_bonus
        )


        if final_score > best_score:

            best_score = final_score

            best_index = index


    # --------------------------------------------------------
    # Fallback response
    # --------------------------------------------------------

    if best_index is None:

        return (
            "We'd like to look into this issue with you. "
            "Please contact Apple Support for further assistance.",
            0
        )


    response = df.iloc[best_index][
        response_column
    ]


    return response, best_score


# ============================================================
# SYSTEM HEADER
# ============================================================

print()

print("======================================")
print("AI CUSTOMER SUPPORT SYSTEM")
print("======================================")

print("Enter a customer problem.")
print("Type 'exit' to stop.")

print()


# ============================================================
# INTERACTIVE CUSTOMER SUPPORT
# ============================================================

while True:

    customer_message = input(
        "Customer: "
    ).strip()


    # --------------------------------------------------------
    # Exit
    # --------------------------------------------------------

    if customer_message.lower() == "exit":

        print()
        print("AI Customer Support stopped.")

        break


    # --------------------------------------------------------
    # Empty input
    # --------------------------------------------------------

    if customer_message == "":

        print(
            "Please enter a customer problem."
        )

        continue


    # --------------------------------------------------------
    # Detect accidental Python command
    # --------------------------------------------------------

    lower_message = customer_message.lower()


    if (
        "python.exe" in lower_message
        or "customer_support_ai.py" in lower_message
    ):

        print()

        print(
            "Please enter a customer problem, "
            "not a Python command."
        )

        print()

        continue


    # --------------------------------------------------------
    # Predict intent
    # --------------------------------------------------------

    intent = predict_intent(
        customer_message
    )


    # --------------------------------------------------------
    # Find Apple Support response
    # --------------------------------------------------------

    response, score = find_best_response(
        customer_message,
        intent
    )


    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print()

    print("--------------------------------------")

    print("CUSTOMER MESSAGE:")
    print(customer_message)

    print()

    print("AI PREDICTED INTENT:")
    print(intent)

    print()

    print("MATCHING SCORE:")
    print(round(score, 3))

    print()

    print("APPLE SUPPORT RESPONSE:")
    print(response)

    print("--------------------------------------")

    print()
    