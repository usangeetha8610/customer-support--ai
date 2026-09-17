import re
import joblib
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = BASE_DIR / "apple_support_naive_bayes_model.pkl"
RESPONSE_FILE = BASE_DIR / "apple_support_pairs.csv"


# ============================================================
# 2. INTENT KEYWORDS
# ============================================================

INTENT_KEYWORDS = {

    "Battery & Charging": [
        ("battery", 6),
        ("battery life", 8),
        ("draining", 7),
        ("drain", 6),
        ("charge", 5),
        ("charging", 6),
        ("charger", 5),
        ("not charging", 8),
        ("won't charge", 8),
        ("doesn't charge", 8),
        ("charging slowly", 7),
        ("battery health", 7),
        ("overheating", 5),
        ("power", 3),
    ],

    "Connectivity": [
        ("wifi", 7),
        ("wi-fi", 7),
        ("wireless", 4),
        ("bluetooth", 8),
        ("internet", 6),
        ("network", 5),
        ("cellular", 6),
        ("mobile data", 7),
        ("data connection", 7),
        ("hotspot", 6),
        ("airdrop", 5),
        ("connection", 4),
        ("not connecting", 6),
        ("can't connect", 7),
        ("cannot connect", 7),
    ],

    "Apple ID & Account": [
        ("apple id", 10),
        ("icloud account", 7),
        ("sign in", 6),
        ("signin", 6),
        ("login", 5),
        ("log in", 5),
        ("password", 6),
        ("forgot password", 8),
        ("account", 4),
        ("verification", 5),
        ("verification code", 7),
        ("locked account", 8),
        ("cannot sign in", 8),
        ("can't sign in", 8),
    ],

    "iCloud & Backup": [
        ("icloud", 7),
        ("icloud backup", 10),
        ("backup", 7),
        ("restore backup", 8),
        ("backup failed", 9),
        ("backup not working", 9),
        ("icloud storage", 7),
        ("sync", 5),
        ("synchronization", 5),
        ("cloud storage", 6),
        ("restore", 4),
    ],

    "Display, Keyboard & Camera": [
        ("screen", 8),
        ("display", 8),
        ("touchscreen", 9),
        ("touch screen", 9),

        ("screen not responding", 15),
        ("screen is not responding", 15),
        ("display not responding", 15),
        ("display is not responding", 15),
        ("touchscreen not responding", 15),
        ("touch screen not responding", 15),

        ("screen not working", 12),
        ("screen is not working", 12),
        ("display not working", 12),
        ("display is not working", 12),
        ("touch not working", 12),
        ("touch screen not working", 12),
        ("touchscreen not working", 12),

        ("keyboard", 8),
        ("typing", 5),
        ("letters", 4),
        ("wrong letters", 7),

        ("camera", 8),
        ("camera not working", 10),
        ("camera issue", 8),

        ("flash", 5),
        ("brightness", 5),
    ],

    "Apps & App Problems": [
        ("app store", 8),
        ("appstore", 8),
        ("app", 5),
        ("apps", 5),
        ("application", 5),
        ("crash", 7),
        ("crashing", 8),
        ("download app", 7),
        ("install app", 7),
        ("app not working", 8),
        ("application not working", 8),
        ("facebook", 4),
        ("instagram", 4),
        ("whatsapp", 4),
        ("youtube app", 5),
    ],

    "Music & Media": [
        ("apple music", 10),
        ("music", 6),
        ("song", 5),
        ("songs", 5),
        ("playlist", 7),
        ("playlists", 7),
        ("podcast", 7),
        ("podcasts", 7),

        ("video", 8),
        ("videos", 8),
        ("video playback", 10),
        ("playing video", 10),
        ("playing videos", 10),
        ("videos are not playing", 12),
        ("video not playing", 12),
        ("video won't play", 12),
        ("video does not play", 12),
        ("video doesn't play", 12),
        ("can't play video", 12),
        ("cannot play video", 12),
        ("unable to play video", 12),
        ("playback", 8),
        ("media playback", 9),

        ("movie", 5),
        ("movies", 5),
        ("tv", 4),
        ("media", 5),
        ("audio", 5),
        ("sound", 4),
        ("speaker", 4),
        ("streaming", 6),
    ],

    "Messages & Communication": [
        ("message", 7),
        ("messages", 7),
        ("imessage", 10),
        ("i message", 9),
        ("sms", 7),
        ("text", 5),
        ("text message", 8),
        ("facetime", 8),
        ("call", 5),
        ("calling", 6),
        ("phone call", 7),
        ("cannot send message", 9),
        ("can't send message", 9),
    ],

    "Payments & Subscriptions": [
        ("payment", 8),
        ("payments", 8),
        ("subscription", 8),
        ("subscriptions", 8),
        ("refund", 8),
        ("billing", 7),
        ("charged", 6),
        ("charge money", 7),
        ("purchase", 7),
        ("purchased", 7),
        ("credit card", 6),
        ("debit card", 6),
        ("app store purchase", 9),
        ("subscription cancelled", 9),
    ],

    "Security & Access": [
        ("security", 8),
        ("secure", 6),
        ("hack", 8),
        ("hacked", 9),
        ("stolen", 8),
        ("theft", 8),
        ("unauthorized", 9),
        ("privacy", 7),
        ("passcode", 7),
        ("password attack", 9),
        ("security issue", 9),
    ],

    "iOS Update & Software Issues": [
        ("ios update", 10),
        ("ios updates", 10),
        ("update", 6),
        ("updates", 6),
        ("software update", 9),
        ("software", 5),
        ("latest ios", 10),
        ("new ios", 8),
        ("ios version", 8),
        ("updating", 6),
        ("update problem", 9),
        ("update problems", 9),
        ("after update", 8),
        ("update failed", 10),
        ("cannot update", 10),
        ("can't update", 10),
    ],

    "General Device Issues": [
        ("iphone", 2),
        ("ipad", 2),
        ("device", 3),
        ("phone", 2),
        ("restart", 5),
        ("reboot", 5),
        ("freeze", 6),
        ("frozen", 6),
        ("slow", 4),
        ("not responding", 5),
        ("won't turn on", 8),
        ("doesn't turn on", 8),
        ("not turning on", 8),
        ("device issue", 6),
        ("iphone issue", 5),
    ],

    "Other": []
}


# ============================================================
# 3. NORMALIZE TEXT
# ============================================================

def normalize_text(text):

    if text is None:
        return ""

    text = str(text).lower()

    text = re.sub(r"http\S+", " ", text)

    text = re.sub(r"@\w+", " ", text)

    text = re.sub(r"#", " ", text)

    text = re.sub(r"[^a-z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# 4. SPECIAL INTENT RULES
# ============================================================

def special_intent(text):

    text = normalize_text(text)

    # --------------------------------------------------------
    # Generic questions
    # --------------------------------------------------------

    generic_questions = [
        "question about my apple device",
        "question about my iphone",
        "question about my ipad",
        "question about my device",
        "general question",
        "just a question",
        "i have a question",
        "need some information",
        "need information about my device"
    ]

    for phrase in generic_questions:

        if phrase in text:

            return "Other", 20

    # --------------------------------------------------------
    # Screen / Display problems
    # --------------------------------------------------------

    screen_phrases = [
        "screen not responding",
        "screen is not responding",
        "display not responding",
        "display is not responding",
        "touchscreen not responding",
        "touch screen not responding",
        "screen not working",
        "screen is not working",
        "display not working",
        "display is not working",
        "touch not working",
        "touch screen not working",
        "touchscreen not working"
    ]

    for phrase in screen_phrases:

        if phrase in text:

            return "Display, Keyboard & Camera", 20

    return None, 0


# ============================================================
# 5. KEYWORD INTENT DETECTION
# ============================================================

def keyword_intent(text):

    text = normalize_text(text)

    # Check special rules first
    special, special_score = special_intent(text)

    if special is not None:

        return special, special_score

    scores = {}

    for intent, keywords in INTENT_KEYWORDS.items():

        score = 0

        for keyword, weight in keywords:

            if keyword in text:

                score += weight

        scores[intent] = score

    if not scores:

        return "Other", 0

    best_intent = max(
        scores,
        key=scores.get
    )

    best_score = scores[best_intent]

    if best_score == 0:

        return "Other", 0

    return best_intent, best_score


# ============================================================
# 6. HYBRID INTENT DETECTION
# ============================================================

def hybrid_intent(text, model):

    clean_text = normalize_text(text)

    # ========================================================
    # SPECIAL RULE 1: SCREEN
    # ========================================================

    screen_phrases = [
        "screen not responding",
        "screen is not responding",
        "display not responding",
        "display is not responding",
        "touchscreen not responding",
        "touch screen not responding",
        "screen not working",
        "screen is not working",
        "display not working",
        "display is not working",
        "touch not working",
        "touch screen not working",
        "touchscreen not working"
    ]

    for phrase in screen_phrases:

        if phrase in clean_text:

            return (
                "Display, Keyboard & Camera",
                "Display, Keyboard & Camera",
                "Other"
            )

    # ========================================================
    # SPECIAL RULE 2: GENERIC QUESTIONS
    # ========================================================

    generic_question_phrases = [
        "question about my apple device",
        "question about my iphone",
        "question about my ipad",
        "question about my device",
        "general question",
        "just a question",
        "i have a question",
        "need some information",
        "need information about my device"
    ]

    for phrase in generic_question_phrases:

        if phrase in clean_text:

            return (
                "Other",
                "Other",
                "Other"
            )

    # ========================================================
    # KEYWORD PREDICTION
    # ========================================================

    keyword_prediction, keyword_score = (
        keyword_intent(text)
    )

    # ========================================================
    # ML PREDICTION
    # ========================================================

    try:

        ml_prediction = model.predict(
            [text]
        )[0]

    except Exception:

        ml_prediction = "Other"

    # ========================================================
    # HYBRID DECISION
    # ========================================================

    if keyword_score >= 8:

        final_intent = keyword_prediction

    elif keyword_prediction == "Other":

        final_intent = ml_prediction

    else:

        final_intent = keyword_prediction

    return (
        final_intent,
        keyword_prediction,
        ml_prediction
    )


# ============================================================
# 7. LOAD AI MODEL
# ============================================================

def load_ai_model():

    print("\nLoading AI model...")

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"\nModel file not found:\n{MODEL_FILE}"
        )

    model = joblib.load(
        MODEL_FILE
    )

    print(
        "AI model loaded successfully."
    )

    return model


# ============================================================
# 8. LOAD RESPONSE DATA
# ============================================================

def load_response_data():

    print(
        "\nLoading Apple Support response pairs..."
    )

    if not RESPONSE_FILE.exists():

        raise FileNotFoundError(
            f"\nResponse file not found:\n{RESPONSE_FILE}"
        )

    df = pd.read_csv(
        RESPONSE_FILE,
        low_memory=False
    )

    print(
        "Total response pairs:",
        len(df)
    )

    print("\nDataset columns:")

    print(
        list(df.columns)
    )

    customer_candidates = [
        "customer_message",
        "customer_tweet",
        "customer_text",
        "tweet",
        "text"
    ]

    response_candidates = [
        "applesupport_response",
        "apple_support_response",
        "response",
        "response_text",
        "reply"
    ]

    customer_column = None
    response_column = None

    for col in customer_candidates:

        if col in df.columns:

            customer_column = col

            break

    for col in response_candidates:

        if col in df.columns:

            response_column = col

            break

    if customer_column is None:

        raise ValueError(
            "Could not find customer message column."
        )

    if response_column is None:

        raise ValueError(
            "Could not find Apple Support response column."
        )

    print(
        "\nCustomer column:",
        customer_column
    )

    print(
        "Response column:",
        response_column
    )

    df = df[
        df[customer_column].notna()
        &
        df[response_column].notna()
    ].copy()

    df[customer_column] = (
        df[customer_column]
        .astype(str)
    )

    df[response_column] = (
        df[response_column]
        .astype(str)
    )

    df = df[
        (df[customer_column].str.strip() != "")
        &
        (df[response_column].str.strip() != "")
    ]

    df.reset_index(
        drop=True,
        inplace=True
    )

    print(
        "\nUsable response pairs:",
        len(df)
    )

    return (
        df,
        customer_column,
        response_column
    )


# ============================================================
# 9. BUILD TF-IDF DATABASE
# ============================================================

def build_tfidf_database(
    df,
    customer_column
):

    print(
        "\nBuilding TF-IDF response database..."
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=60000,
        ngram_range=(1, 2)
    )

    customer_texts = (
        df[customer_column]
        .fillna("")
        .astype(str)
        .tolist()
    )

    matrix = vectorizer.fit_transform(
        customer_texts
    )

    print(
        "TF-IDF matrix shape:",
        matrix.shape
    )

    return vectorizer, matrix


# ============================================================
# 10. KEYWORD OVERLAP
# ============================================================

def keyword_overlap(
    user_text,
    candidate_text
):

    user_words = set(
        normalize_text(
            user_text
        ).split()
    )

    candidate_words = set(
        normalize_text(
            candidate_text
        ).split()
    )

    if not user_words:

        return 0

    common_words = (
        user_words
        .intersection(candidate_words)
    )

    return (
        len(common_words)
        /
        len(user_words)
    )


# ============================================================
# 11. CANDIDATE INTENT
# ============================================================

def candidate_intent(text):

    intent, score = keyword_intent(
        text
    )

    return intent


# ============================================================
# 12. FIND BEST RESPONSE
# ============================================================

def find_best_response(
    user_text,
    predicted_intent,
    df,
    customer_column,
    response_column,
    vectorizer,
    matrix
):

    user_vector = vectorizer.transform(
        [user_text]
    )

    similarities = cosine_similarity(
        user_vector,
        matrix
    ).flatten()

    candidate_count = min(
        100,
        len(similarities)
    )

    top_indices = np.argsort(
        similarities
    )[-candidate_count:][::-1]

    best_index = None
    best_score = -1

    for index in top_indices:

        candidate_text = str(
            df.iloc[index][
                customer_column
            ]
        )

        tfidf_score = similarities[index]

        overlap_score = keyword_overlap(
            user_text,
            candidate_text
        )

        candidate_int = candidate_intent(
            candidate_text
        )

        intent_bonus = 0

        if candidate_int == predicted_intent:

            intent_bonus = 0.25

        keyword_bonus = (
            overlap_score * 0.20
        )

        final_score = (
            tfidf_score * 0.55
            +
            keyword_bonus
            +
            intent_bonus
        )

        # ----------------------------------------------------
        # Music & Media boost
        # ----------------------------------------------------

        if predicted_intent == "Music & Media":

            candidate_lower = normalize_text(
                candidate_text
            )

            media_words = [
                "video",
                "videos",
                "music",
                "song",
                "songs",
                "playlist",
                "podcast",
                "movie",
                "movies",
                "playback",
                "media"
            ]

            media_match = sum(
                1
                for word in media_words
                if word in candidate_lower
            )

            final_score += (
                media_match * 0.03
            )

        # ----------------------------------------------------
        # Connectivity boost
        # ----------------------------------------------------

        if predicted_intent == "Connectivity":

            candidate_lower = normalize_text(
                candidate_text
            )

            connectivity_words = [
                "wifi",
                "wi-fi",
                "bluetooth",
                "internet",
                "network",
                "cellular",
                "mobile data",
                "hotspot",
                "connection"
            ]

            connectivity_match = sum(
                1
                for word in connectivity_words
                if word in candidate_lower
            )

            final_score += (
                connectivity_match * 0.03
            )

        # ----------------------------------------------------
        # Display boost
        # ----------------------------------------------------

        if predicted_intent == (
            "Display, Keyboard & Camera"
        ):

            candidate_lower = normalize_text(
                candidate_text
            )

            display_words = [
                "screen",
                "display",
                "touch",
                "touchscreen",
                "keyboard",
                "typing",
                "camera"
            ]

            display_match = sum(
                1
                for word in display_words
                if word in candidate_lower
            )

            final_score += (
                display_match * 0.03
            )

        if final_score > best_score:

            best_score = final_score

            best_index = index

    if best_index is None:

        return (
            "We'd like to help you with this. "
            "Please send us a DM with more details.",
            0
        )

    best_response = str(
        df.iloc[best_index][
            response_column
        ]
    )

    return (
        best_response,
        best_score
    )


# ============================================================
# 13. INVALID INPUT
# ============================================================

def is_invalid_input(text):

    text = text.strip()

    if not text:

        return True

    if len(text) < 3:

        return True

    return False


# ============================================================
# 14. MAIN AI SYSTEM
# ============================================================

def main():

    print("=" * 65)

    print(
        "      APPLE SUPPORT CUSTOMER SUPPORT AI"
    )

    print("=" * 65)

    # Load model
    model = load_ai_model()

    # Load response dataset
    (
        df,
        customer_column,
        response_column
    ) = load_response_data()

    # Build TF-IDF
    (
        vectorizer,
        matrix
    ) = build_tfidf_database(
        df,
        customer_column
    )

    print(
        "\nAI system is ready."
    )

    print(
        "\nType your Apple device problem."
    )

    print(
        "Type 'exit' to stop."
    )

    print("=" * 65)

    # ========================================================
    # INTERACTIVE LOOP
    # ========================================================

    while True:

        try:

            user_text = input(
                "\nCustomer: "
            ).strip()

        except KeyboardInterrupt:

            print(
                "\n\nExiting..."
            )

            break

        # Exit
        if user_text.lower() in [
            "exit",
            "quit",
            "q"
        ]:

            print(
                "\nThank you for using Apple Support AI."
            )

            break

        # Invalid input
        if is_invalid_input(
            user_text
        ):

            print(
                "\nPlease enter a valid problem."
            )

            continue

        # Hybrid prediction
        (
            predicted_intent,
            keyword_prediction,
            ml_prediction
        ) = hybrid_intent(
            user_text,
            model
        )

        # Find response
        (
            response,
            score
        ) = find_best_response(
            user_text,
            predicted_intent,
            df,
            customer_column,
            response_column,
            vectorizer,
            matrix
        )

        # Display
        print(
            "\nPredicted Intent:",
            predicted_intent
        )

        print(
            "Keyword Intent:",
            keyword_prediction
        )

        print(
            "ML Intent:",
            ml_prediction
        )

        print(
            "Matching Score:",
            round(score, 3)
        )

        print(
            "\nApple Support Response:"
        )

        print(response)

        print(
            "-" * 65
        )


# ============================================================
# 15. START PROGRAM
# ============================================================

if __name__ == "__main__":

    main()