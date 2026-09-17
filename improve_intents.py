import pandas as pd
import re


# ==================================================
# SETTINGS
# ==================================================

INPUT_FILE = "apple_support_conversations.csv"
OUTPUT_FILE = "apple_support_intents_improved.csv"


# ==================================================
# READ DATA
# ==================================================

print("Reading conversation dataset...")

df = pd.read_csv(INPUT_FILE)

print("Total conversations:", len(df))


# ==================================================
# EXTRACT CUSTOMER MESSAGES
# ==================================================

def get_customer_text(conversation):

    if pd.isna(conversation):
        return ""

    lines = str(conversation).split("\n")

    messages = []

    for line in lines:

        line = line.strip()

        if line.startswith("Customer:"):

            message = line[
                len("Customer:"):
            ].strip()

            if message:
                messages.append(message)

    return " ".join(messages)


print("Extracting customer messages...")

df["customer_text"] = df[
    "full_conversation"
].apply(get_customer_text)


# ==================================================
# CLEAN TEXT
# ==================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


df["customer_clean"] = df[
    "customer_text"
].apply(clean_text)


# ==================================================
# IMPROVED INTENT CLASSIFIER
# ==================================================

def classify_intent(text):

    text = str(text).lower()


    # ------------------------------------------------
    # SECURITY & ACCESS
    # ------------------------------------------------

    security_patterns = [
        "passcode",
        "forgot passcode",
        "disable passcode",
        "locked out",
        "phone is locked",
        "iphone is locked",
        "activation lock",
        "find my iphone",
        "find my iphone",
        "lost iphone",
        "lost phone",
        "stolen iphone",
        "stolen phone",
        "security",
        "privacy",
        "fingerprint",
        "touch id",
        "face id"
    ]

    if any(
        word in text
        for word in security_patterns
    ):

        return "Security & Access"


    # ------------------------------------------------
    # DISPLAY / KEYBOARD / CAMERA
    # ------------------------------------------------

    display_patterns = [
        "screen",
        "display",
        "touch screen",
        "touchscreen",
        "screen not responding",
        "screen won't respond",
        "screen is not responding",
        "black screen",
        "blank screen",
        "flickering screen",
        "keyboard",
        "keypad",
        "typing",
        "autocorrect",
        "auto correct",
        "letters",
        "capital i",
        "camera",
        "camera app",
        "camera not working",
        "take pictures",
        "take photos",
        "photo",
        "photos",
        "flash"
    ]

    if any(
        word in text
        for word in display_patterns
    ):

        return "Display, Keyboard & Camera"


    # ------------------------------------------------
    # IOS UPDATE / SOFTWARE
    # ------------------------------------------------

    update_patterns = [
        "ios update",
        "software update",
        "software upgrade",
        "update failed",
        "update failure",
        "update problem",
        "update issue",
        "cannot update",
        "can't update",
        "couldn't update",
        "unable to update",
        "after updating",
        "after update",
        "after i updated",
        "updated to ios",
        "upgraded to ios",
        "ios 11",
        "ios11",
        "ios 12",
        "ios12",
        "ios 13",
        "ios13",
        "ios 14",
        "ios14",
        "ios 15",
        "ios15",
        "ios 16",
        "ios16",
        "ios 17",
        "ios17",
        "ios 18",
        "ios18",
        "ios 19",
        "ios19"
    ]

    if any(
        word in text
        for word in update_patterns
    ):

        return "iOS Update & Software Issues"


    # ------------------------------------------------
    # PAYMENTS & SUBSCRIPTIONS
    # ------------------------------------------------

    payment_patterns = [
        "payment",
        "paid",
        "purchase",
        "refund",
        "billing",
        "subscription",
        "charged",
        "charge",
        "credit",
        "debit",
        "money",
        "app store purchase"
    ]

    if any(
        word in text
        for word in payment_patterns
    ):

        return "Payments & Subscriptions"


    # ------------------------------------------------
    # APPLE ID & ACCOUNT
    # ------------------------------------------------

    account_patterns = [
        "apple id",
        "appleid",
        "apple account",
        "account",
        "password",
        "forgot password",
        "sign in",
        "signin",
        "login",
        "log in"
    ]

    if any(
        word in text
        for word in account_patterns
    ):

        return "Apple ID & Account"


    # ------------------------------------------------
    # ICLOUD & BACKUP
    # ------------------------------------------------

    cloud_patterns = [
        "icloud",
        "i cloud",
        "backup",
        "back up",
        "restore backup",
        "icloud storage"
    ]

    if any(
        word in text
        for word in cloud_patterns
    ):

        return "iCloud & Backup"


    # ------------------------------------------------
    # BATTERY & CHARGING
    # ------------------------------------------------

    battery_patterns = [
        "battery",
        "battery life",
        "battery drain",
        "battery draining",
        "charging",
        "charger",
        "charging cable",
        "won't charge",
        "not charging",
        "charge"
    ]

    if any(
        word in text
        for word in battery_patterns
    ):

        return "Battery & Charging"


    # ------------------------------------------------
    # CONNECTIVITY
    # ------------------------------------------------

    connectivity_patterns = [
        "wifi",
        "wi-fi",
        "bluetooth",
        "internet",
        "network",
        "cellular",
        "mobile data",
        "hotspot",
        "connectivity",
        "signal",
        "no signal"
    ]

    if any(
        word in text
        for word in connectivity_patterns
    ):

        return "Connectivity"


    # ------------------------------------------------
    # MESSAGES & COMMUNICATION
    # ------------------------------------------------

    communication_patterns = [
        "imessage",
        "i message",
        "messages",
        "message",
        "facetime",
        "face time",
        "text message",
        "sms",
        "notification",
        "notifications",
        "mail",
        "email"
    ]

    if any(
        word in text
        for word in communication_patterns
    ):

        return "Messages & Communication"


    # ------------------------------------------------
    # MUSIC & MEDIA
    # ------------------------------------------------

    media_patterns = [
        "apple music",
        "music",
        "itunes",
        "song",
        "songs",
        "album",
        "playlist",
        "podcast",
        "video",
        "movie",
        "movies",
        "apple tv"
    ]

    if any(
        word in text
        for word in media_patterns
    ):

        return "Music & Media"


    # ------------------------------------------------
    # APPS
    # ------------------------------------------------

    app_patterns = [
        "app store",
        "app",
        "apps",
        "application",
        "crash",
        "crashes",
        "crashing",
        "app won't open",
        "app not working"
    ]

    if any(
        word in text
        for word in app_patterns
    ):

        return "Apps & App Problems"


    # ------------------------------------------------
    # GENERAL DEVICE
    # ------------------------------------------------

    device_patterns = [
        "iphone",
        "ipad",
        "macbook",
        "mac",
        "apple watch",
        "phone",
        "device",
        "won't turn on",
        "not turning on",
        "not working",
        "doesn't work",
        "doesnt work",
        "problem",
        "issue",
        "restart",
        "restarting",
        "freezing",
        "frozen"
    ]

    if any(
        word in text
        for word in device_patterns
    ):

        return "General Device Issues"


    # ------------------------------------------------
    # OTHER
    # ------------------------------------------------

    return "Other"


# ==================================================
# ASSIGN NEW INTENTS
# ==================================================

print("Assigning improved intents...")

df["intent"] = df[
    "customer_clean"
].apply(classify_intent)


# ==================================================
# DISTRIBUTION
# ==================================================

print("\n======================================")
print("IMPROVED INTENT DISTRIBUTION")
print("======================================")

print(
    df["intent"].value_counts()
)


# ==================================================
# SAVE
# ==================================================

output = df[
    [
        "conversation_id",
        "customer_text",
        "full_conversation",
        "intent"
    ]
].copy()


output.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==================================================
# DONE
# ==================================================

print("\n======================================")
print("IMPROVED DATASET CREATED")
print("======================================")

print(
    "Records:",
    len(output)
)

print(
    "Intents:",
    output["intent"].nunique()
)

print(
    "Output:",
    OUTPUT_FILE
)