import streamlit as st
import sys
from pathlib import Path

# Add project folder to Python path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Import functions from your existing AI
from customer_support_ai import (
    load_ai_model,
    load_response_data,
    build_tfidf_database,
    hybrid_intent,
    find_best_response
)


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Apple Support Customer AI",
    page_icon="🍎",
    layout="centered"
)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("🍎 Apple Support Customer AI")

st.write(
    "Describe your Apple device problem and the AI will "
    "predict the issue category and retrieve a relevant "
    "Apple Support response."
)


# ---------------------------------------------------------
# LOAD AI MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_system():

    model = load_ai_model()

    response_df, customer_col, response_col = load_response_data()

    vectorizer, tfidf_matrix = build_tfidf_database(
        response_df,
        response_col
    )

    return (
        model,
        response_df,
        customer_col,
        response_col,
        vectorizer,
        tfidf_matrix
    )


with st.spinner("Loading Apple Support AI..."):

    (
        model,
        response_df,
        customer_col,
        response_col,
        vectorizer,
        tfidf_matrix
    ) = load_system()


# ---------------------------------------------------------
# USER INPUT
# ---------------------------------------------------------

st.subheader("Describe your problem")

user_query = st.text_area(
    "Enter your Apple device problem:",
    placeholder="Example: My iPhone battery is draining very quickly",
    height=120
)


# ---------------------------------------------------------
# ASK BUTTON
# ---------------------------------------------------------

if st.button("🔍 Ask Apple Support", use_container_width=True):

    if not user_query.strip():

        st.warning("Please enter your problem.")

    else:

        # Get hybrid intent
        predicted_intent, keyword_intent_value, ml_intent = hybrid_intent(
            user_query,
            model
        )

        # Find best Apple Support response
        response, score = find_best_response(
            user_query,
            predicted_intent,
            response_df,
            customer_col,
            response_col,
            vectorizer,
            tfidf_matrix
        )

        # -------------------------------------------------
        # RESULTS
        # -------------------------------------------------

        st.success("Problem analyzed successfully!")

        st.subheader("🎯 Predicted Intent")

        st.info(predicted_intent)

        # -------------------------------------------------
        # MODEL INFORMATION
        # -------------------------------------------------

        with st.expander("View AI classification details"):

            st.write(
                "**Keyword Intent:**",
                keyword_intent_value
            )

            st.write(
                "**ML Intent:**",
                ml_intent
            )

            st.write(
                "**Hybrid Prediction:**",
                predicted_intent
            )

        # -------------------------------------------------
        # MATCHING SCORE
        # -------------------------------------------------

        st.subheader("📊 Response Matching Score")

        st.progress(
            min(max(float(score), 0.0), 1.0)
        )

        st.write(
            f"Matching Score: **{score:.3f}**"
        )

        # -------------------------------------------------
        # APPLE SUPPORT RESPONSE
        # -------------------------------------------------

        st.subheader("💬 Apple Support Response")

        st.success(response)


# ---------------------------------------------------------
# PROJECT INFORMATION
# ---------------------------------------------------------

st.divider()

st.caption(
    "AI-based Apple Customer Support System | "
    "Hybrid Intent Classification + TF-IDF Response Retrieval"
)