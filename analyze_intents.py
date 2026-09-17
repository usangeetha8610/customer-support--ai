import pandas as pd
import re
from collections import Counter

# ==================================================
# SETTINGS
# ==================================================

INPUT_FILE = "apple_support_conversations.csv"

# Number of common words to display
TOP_WORDS = 50


# ==================================================
# STEP 1: Read conversation dataset
# ==================================================

print("Reading AppleSupport conversations...")

df = pd.read_csv(INPUT_FILE)

print("Total conversations:", len(df))


# ==================================================
# STEP 2: Extract customer messages
# ==================================================

print("Extracting customer messages...")

customer_messages = []

for conversation in df["full_conversation"].dropna():

    # Split conversation into individual lines
    lines = str(conversation).split("\n")

    for line in lines:

        line = line.strip()

        # Keep only Customer messages
        if line.startswith("Customer:"):

            message = line.replace(
                "Customer:",
                "",
                1
            ).strip()

            if message:
                customer_messages.append(message)


print(
    "Customer messages found:",
    len(customer_messages)
)


# ==================================================
# STEP 3: Combine customer messages
# ==================================================

all_text = " ".join(customer_messages).lower()


# ==================================================
# STEP 4: Clean text
# ==================================================

# Remove URLs
all_text = re.sub(
    r"http\S+|www\S+",
    "",
    all_text
)

# Keep letters and numbers
all_text = re.sub(
    r"[^a-z0-9\s]",
    " ",
    all_text
)

# Remove extra spaces
all_text = re.sub(
    r"\s+",
    " ",
    all_text
).strip()


# ==================================================
# STEP 5: Split into words
# ==================================================

words = all_text.split()


# ==================================================
# STEP 6: Remove common English words
# ==================================================

stop_words = {
    "the", "a", "an", "and", "or", "but",
    "is", "are", "was", "were", "be",
    "to", "of", "in", "on", "for",
    "with", "my", "me", "i", "it",
    "this", "that", "have", "has",
    "had", "do", "does", "did",
    "can", "could", "would", "should",
    "will", "just", "not", "no",
    "so", "if", "you", "your",
    "we", "our", "they", "their",
    "from", "at", "as", "about",
    "what", "when", "where", "why",
    "how", "was", "been", "im",
    "dont", "its", "ive", "cant"
}


filtered_words = [
    word
    for word in words
    if word not in stop_words
    and len(word) > 2
]


# ==================================================
# STEP 7: Count common words
# ==================================================

word_counts = Counter(
    filtered_words
)


# ==================================================
# STEP 8: Display common words
# ==================================================

print("\n======================================")
print("MOST COMMON CUSTOMER WORDS")
print("======================================")

for word, count in word_counts.most_common(TOP_WORDS):

    print(
        f"{word:25} {count}"
    )


# ==================================================
# STEP 9: Search for possible issue keywords
# ==================================================

issue_keywords = [
    "iphone",
    "ipad",
    "mac",
    "ios",
    "update",
    "battery",
    "charge",
    "charging",
    "icloud",
    "appleid",
    "account",
    "password",
    "app",
    "itunes",
    "music",
    "wifi",
    "bluetooth",
    "internet",
    "network",
    "email",
    "notification",
    "message",
    "imessage",
    "facetime",
    "screen",
    "camera",
    "backup",
    "restore",
    "payment",
    "purchase",
    "refund",
    "subscription",
    "security",
    "locked",
    "login"
]


# ==================================================
# STEP 10: Count issue keywords
# ==================================================

print("\n======================================")
print("POSSIBLE CUSTOMER ISSUE KEYWORDS")
print("======================================")

for keyword in issue_keywords:

    count = sum(
        1
        for word in words
        if word == keyword
    )

    if count > 0:

        print(
            f"{keyword:25} {count}"
        )


# ==================================================
# STEP 11: Display sample customer messages
# ==================================================

print("\n======================================")
print("SAMPLE CUSTOMER MESSAGES")
print("======================================")

for i, message in enumerate(
    customer_messages[:30],
    start=1
):

    print(
        f"{i}. {message}"
    )


print("\n======================================")
print("ANALYSIS COMPLETED")
print("======================================")