import pandas as pd

# ==================================================
# SETTINGS
# ==================================================

INPUT_FILE = "twcs.csv"
OUTPUT_FILE = "apple_support_conversations.csv"

BRAND = "AppleSupport"


# ==================================================
# STEP 1: Read the original dataset
# ==================================================

print("Reading original dataset...")

df = pd.read_csv(
    INPUT_FILE,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id"
    ]
)

print("Total tweets:", len(df))


# ==================================================
# STEP 2: Create tweet lookup
# ==================================================

print("Creating tweet lookup...")

tweet_lookup = df.set_index("tweet_id").to_dict("index")


# ==================================================
# STEP 3: Find AppleSupport tweets
# ==================================================

print("Finding AppleSupport tweets...")

apple_tweets = df[
    df["author_id"].astype(str).str.strip().str.lower()
    == BRAND.lower()
].copy()

print("AppleSupport tweets:", len(apple_tweets))


# ==================================================
# STEP 4: Build conversations
# ==================================================

print("Building conversations...")

conversations = []

for _, apple_row in apple_tweets.iterrows():

    current_id = apple_row["tweet_id"]

    conversation = []

    # Add the AppleSupport message
    conversation.append({
        "tweet_id": current_id,
        "author": "AppleSupport",
        "created_at": apple_row["created_at"],
        "text": apple_row["text"]
    })

    # Find the message AppleSupport replied to
    previous_id = apple_row["in_response_to_tweet_id"]

    steps = 0

    # Follow the conversation backwards
    while pd.notna(previous_id) and steps < 20:

        try:
            previous_id = int(previous_id)
        except:
            break

        if previous_id not in tweet_lookup:
            break

        previous_tweet = tweet_lookup[previous_id]

        # The tweet ID is the dictionary key
        tweet_id = previous_id

        # Determine the author
        if str(previous_tweet["author_id"]).strip().lower() == BRAND.lower():
            author = "AppleSupport"
        else:
            author = "Customer"

        conversation.append({
            "tweet_id": tweet_id,
            "author": author,
            "created_at": previous_tweet["created_at"],
            "text": previous_tweet["text"]
        })

        # Move to the previous tweet
        previous_id = previous_tweet["in_response_to_tweet_id"]

        steps += 1

    # Reverse so conversation starts from oldest message
    conversation.reverse()

    # Create readable conversation
    conversation_text = []

    for message in conversation:

        conversation_text.append(
            message["author"]
            + ": "
            + str(message["text"])
        )

    conversation_text = "\n".join(conversation_text)

    conversations.append({
        "conversation_id": current_id,
        "conversation": conversation_text,
        "number_of_messages": len(conversation)
    })


# ==================================================
# STEP 5: Create dataframe
# ==================================================

result = pd.DataFrame(conversations)


# ==================================================
# STEP 6: Remove duplicate conversations
# ==================================================

result = result.drop_duplicates(
    subset=["conversation"]
)


# ==================================================
# STEP 7: Save the conversations
# ==================================================

result.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==================================================
# STEP 8: Display results
# ==================================================

print("\n======================================")
print("CONVERSATION BUILDING COMPLETED")
print("======================================")

print("Number of conversations:", len(result))

print("\nFirst 3 conversations:\n")

for _, row in result.head(3).iterrows():

    print("--------------------------------------")
    print("Conversation ID:", row["conversation_id"])
    print("Number of messages:", row["number_of_messages"])
    print("--------------------------------------")
    print(row["conversation"])
    print()


print("Output file:")
print(OUTPUT_FILE)