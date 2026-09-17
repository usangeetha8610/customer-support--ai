from customer_support_ai import load_ai_model, hybrid_intent


def main():

    print("\n==========================================")
    print("APPLE CUSTOMER SUPPORT AI")
    print("FINAL V2 HYBRID EVALUATION")
    print("==========================================\n")

    print("Loading actual V2 AI model...\n")

    model = load_ai_model()

    # =========================================================
    # TEST CASES
    # =========================================================

    test_cases = [

        # Battery & Charging
        ("My iPhone battery is draining very quickly",
         "Battery & Charging"),

        ("My iPhone is not charging",
         "Battery & Charging"),

        ("My battery health is getting worse",
         "Battery & Charging"),

        ("My charger is not working",
         "Battery & Charging"),

        # Connectivity
        ("My WiFi is not connecting to my iPhone",
         "Connectivity"),

        ("Bluetooth is not connecting",
         "Connectivity"),

        ("My internet is not working",
         "Connectivity"),

        ("Mobile data is not working",
         "Connectivity"),

        # Apple ID
        ("I cannot sign into my Apple ID",
         "Apple ID & Account"),

        ("I forgot my Apple ID password",
         "Apple ID & Account"),

        ("I cannot login to my Apple account",
         "Apple ID & Account"),

        # iCloud
        ("My iCloud backup is not working",
         "iCloud & Backup"),

        ("I cannot backup my iPhone to iCloud",
         "iCloud & Backup"),

        ("My iCloud storage is full",
         "iCloud & Backup"),

        # Display / Keyboard / Camera
        ("My iPhone screen is not responding",
         "Display, Keyboard & Camera"),

        ("My camera is not working",
         "Display, Keyboard & Camera"),

        ("My keyboard is typing the wrong letters",
         "Display, Keyboard & Camera"),

        ("My touchscreen is not working",
         "Display, Keyboard & Camera"),

        # Apps
        ("My App Store keeps crashing",
         "Apps & App Problems"),

        ("The app keeps crashing",
         "Apps & App Problems"),

        ("I cannot download an app",
         "Apps & App Problems"),

        ("My application is not working",
         "Apps & App Problems"),

        # Music & Media
        ("Apple Music is not working",
         "Music & Media"),

        ("My songs are not playing",
         "Music & Media"),

        ("My playlist is missing",
         "Music & Media"),

        ("My podcast is not playing",
         "Music & Media"),

        ("Videos are not playing on my iPhone",
         "Music & Media"),

        # Messages
        ("I cannot send messages",
         "Messages & Communication"),

        ("iMessage is not working",
         "Messages & Communication"),

        ("FaceTime is not working",
         "Messages & Communication"),

        # Payments
        ("I was charged for an App Store purchase",
         "Payments & Subscriptions"),

        ("I want a refund for an app",
         "Payments & Subscriptions"),

        ("My subscription payment failed",
         "Payments & Subscriptions"),

        # Security
        ("I think my account has been hacked",
         "Security & Access"),

        ("I have a security problem",
         "Security & Access"),

        # iOS Update
        ("The latest iOS update is causing problems",
         "iOS Update & Software Issues"),

        ("My iOS update failed",
         "iOS Update & Software Issues"),

        ("I cannot update my iPhone",
         "iOS Update & Software Issues"),

        # General Device
        ("My iPhone keeps restarting",
         "General Device Issues"),

        ("My iPhone is frozen",
         "General Device Issues"),

        ("My phone is very slow",
         "General Device Issues"),

        ("My iPhone will not turn on",
         "General Device Issues"),

        ("My device is having an issue",
         "General Device Issues"),

        # Other
        ("I have a question about my Apple device",
         "Other"),

    ]

    print(
        f"Running {len(test_cases)} test cases...\n"
    )

    # =========================================================
    # EVALUATION
    # =========================================================

    correct = 0
    incorrect = 0

    results = []

    for number, (text, expected) in enumerate(
        test_cases, 1
    ):

        # -----------------------------------------------------
        # UPDATED FOR NEW V2 HYBRID FUNCTION
        # -----------------------------------------------------

        (
            predicted,
            keyword_prediction,
            ml_prediction
        ) = hybrid_intent(
            text,
            model
        )

        # -----------------------------------------------------
        # Check result
        # -----------------------------------------------------

        is_correct = (
            predicted == expected
        )

        if is_correct:
            correct += 1
            status = "CORRECT"
        else:
            incorrect += 1
            status = "INCORRECT"

        # -----------------------------------------------------
        # Print result
        # -----------------------------------------------------

        print(
            f"{number:02d}. {text}"
        )

        print(
            f"    Expected : {expected}"
        )

        print(
            f"    Keyword  : {keyword_prediction}"
        )

        print(
            f"    ML       : {ml_prediction}"
        )

        print(
            f"    Hybrid   : {predicted}"
        )

        print(
            f"    {status}"
        )

        print("-" * 60)

        # -----------------------------------------------------
        # Save result
        # -----------------------------------------------------

        results.append({
            "test_number": number,
            "customer_message": text,
            "expected_intent": expected,
            "keyword_intent": keyword_prediction,
            "ml_intent": ml_prediction,
            "hybrid_intent": predicted,
            "result": status
        })

    # =========================================================
    # FINAL RESULTS
    # =========================================================

    total = len(test_cases)

    accuracy = (
        correct / total
    ) * 100

    print("\n==========================================")
    print("FINAL EVALUATION RESULT")
    print("==========================================")

    print(
        f"Total Test Cases : {total}"
    )

    print(
        f"Correct          : {correct}"
    )

    print(
        f"Incorrect        : {incorrect}"
    )

    print(
        f"Accuracy         : {accuracy:.2f}%"
    )

    print("==========================================")

    # =========================================================
    # SHOW INCORRECT CASES
    # =========================================================

    if incorrect > 0:

        print(
            "\nINCORRECT TEST CASES:"
        )

        for result in results:

            if result["result"] == "INCORRECT":

                print(
                    f"\n{result['test_number']}. "
                    f"{result['customer_message']}"
                )

                print(
                    f"Expected: "
                    f"{result['expected_intent']}"
                )

                print(
                    f"Hybrid: "
                    f"{result['hybrid_intent']}"
                )

    else:

        print(
            "\nAll test cases passed successfully!"
        )

    # =========================================================
    # SAVE CSV
    # =========================================================

    import csv

    output_file = (
        "final_v2_evaluation_results.csv"
    )

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "test_number",
                "customer_message",
                "expected_intent",
                "keyword_intent",
                "ml_intent",
                "hybrid_intent",
                "result"
            ]
        )

        writer.writeheader()

        writer.writerows(results)

    print(
        f"\nResults saved to:"
    )

    print(
        output_file
    )


if __name__ == "__main__":
    main()
    