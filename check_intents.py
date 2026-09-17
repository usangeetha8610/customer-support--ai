import pandas as pd
import random

# ==================================================
# SETTINGS
# ==================================================

INPUT_FILE = "apple_support_intents.csv"

# Number of random examples per intent
EXAMPLES_PER_INTENT = 5


# ==================================================
# STEP 1: Read dataset
# ==================================================

print("Reading intent dataset...")

df = pd.read_csv(INPUT_FILE)

print("Total records:", len(df))


# ==================================================
# STEP 2: Check missing values
# ==================================================

print("\n======================================")
print("MISSING VALUES")
print("======================================")

print(df[
    ["conversation_id", "customer_text", "intent"]
].isnull().sum())


# ==================================================
# STEP 3: Intent distribution
# ==================================================

print("\n======================================")
print("INTENT DISTRIBUTION")
print("======================================")

counts = df["intent"].value_counts()

print(counts)


# ==================================================
# STEP 4: Intent percentages
# ==================================================

print("\n======================================")
print("INTENT PERCENTAGES")
print("======================================")

percentages = (
    df["intent"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

for intent, percentage in percentages.items():

    print(
        f"{intent:35} {percentage}%"
    )


# ==================================================
# STEP 5: Random examples
# ==================================================

print("\n======================================")
print("RANDOM INTENT EXAMPLES")
print("======================================")


for intent in counts.index:

    print("\n")
    print("--------------------------------------")
    print("INTENT:", intent)
    print("--------------------------------------")

    subset = df[
        df["intent"] == intent
    ]

    number = min(
        EXAMPLES_PER_INTENT,
        len(subset)
    )

    examples = subset.sample(
        n=number,
        random_state=42
    )

    for index, row in examples.iterrows():

        print(
            "\nCustomer:",
            row["customer_text"]
        )

        print(
            "Assigned Intent:",
            row["intent"]
        )


# ==================================================
# STEP 6: Check duplicate customer texts
# ==================================================

print("\n======================================")
print("DUPLICATE CUSTOMER TEXT CHECK")
print("======================================")

duplicates = df[
    "customer_text"
].duplicated().sum()

print(
    "Duplicate customer texts:",
    duplicates
)


# ==================================================
# STEP 7: Final summary
# ==================================================

print("\n======================================")
print("QUALITY CHECK COMPLETED")
print("======================================")

print(
    "Total records:",
    len(df)
)

print(
    "Number of intents:",
    df["intent"].nunique()
)

print(
    "Duplicate customer texts:",
    duplicates
)