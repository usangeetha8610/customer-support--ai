import pandas as pd

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

DATA_FILE = "twcs.csv"
OUTPUT_FILE = "apple_support_pairs.csv"

BRAND = "AppleSupport"


# --------------------------------------------------
# STEP 1: Read the dataset
# --------------------------------------------------

print("Reading dataset...")

df = pd.read_csv(
    DATA_FILE,
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


# --------------------------------------------------
# STEP 2: Extract AppleSupport tweets
# --------------------------------------------------

print("Finding AppleSupport tweets...")

apple = df[
    df["author_id"].astype(str).str.strip().str.lower()
    == BRAND.lower()
].copy()

print("AppleSupport tweets:", len(apple))


# --------------------------------------------------
# STEP 3: Create tweet lookup
# --------------------------------------------------

tweet_lookup = df.set_index("tweet_id")


# --------------------------------------------------
# STEP 4: Pair customer messages with
#         AppleSupport responses
# --------------------------------------------------

pairs = []

for _, brand_row in apple.iterrows():

    # AppleSupport tweet ID
    applesupport_id = brand_row["tweet_id"]

    # Customer tweet that AppleSupport replied to
    customer_id = brand_row["in_response_to_tweet_id"]

    # Skip if there is no previous tweet
    if pd.isna(customer_id):
        continue

    try:
        customer_id = int(customer_id)
    except:
        continue

    # Check whether customer tweet exists
    if customer_id not in tweet_lookup.index:
        continue

    customer_row = tweet_lookup.loc[customer_id]

    # Keep only incoming/customer tweets
    if str(customer_row["inbound"]).upper() != "TRUE":
        continue

    # Save the pair
    pairs.append({
        "customer_tweet_id": customer_id,
        "customer_message": customer_row["text"],
        "applesupport_tweet_id": applesupport_id,
        "applesupport_response": brand_row["text"],
        "customer_created_at": customer_row["created_at"],
        "response_created_at": brand_row["created_at"]
    })


# --------------------------------------------------
# STEP 5: Create dataframe
# --------------------------------------------------

result = pd.DataFrame(pairs)


# --------------------------------------------------
# STEP 6: Remove duplicates
# --------------------------------------------------

result = result.drop_duplicates(
    subset=[
        "customer_tweet_id",
        "applesupport_tweet_id"
    ]
)


# --------------------------------------------------
# STEP 7: Remove empty messages
# --------------------------------------------------

result = result.dropna(
    subset=[
        "customer_message",
        "applesupport_response"
    ]
)


# --------------------------------------------------
# STEP 8: Save the result
# --------------------------------------------------

result.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# STEP 9: Show results
# --------------------------------------------------

print("\n======================================")
print("EXTRACTION COMPLETED")
print("======================================")

print("AppleSupport tweets:", len(apple))
print("Customer → AppleSupport pairs:", len(result))

print("\nFirst 5 customer → AppleSupport pairs:\n")

print(
    result[
        [
            "customer_tweet_id",
            "customer_message",
            "applesupport_tweet_id",
            "applesupport_response"
        ]
    ].head(5).to_string(index=False)
)

print("\nOutput file created:")
print(OUTPUT_FILE)