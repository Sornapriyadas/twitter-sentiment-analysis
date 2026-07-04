import streamlit as st
import joblib
import re
import pandas as pd
from xquik_export import load_xquik_rows

# Page settings
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Custom CSS
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.title {
    text-align: center;
    color: #00BFFF;
    font-size: 45px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: white;
    font-size: 20px;
    margin-bottom: 30px;
}

</style>
""", unsafe_allow_html=True)

# Title
st.markdown(
    '<p class="title">💬 Twitter Sentiment Analysis</p>',
    unsafe_allow_html=True
)

# Subtitle
st.markdown(
    '<p class="subtitle">AI/ML Internship Project using NLP & Machine Learning</p>',
    unsafe_allow_html=True
)

# User input
tweet = st.text_area(
    "Enter Tweet",
    placeholder="Type something like: I love this movie!"
)

# Clean text function
def clean_text(text):

    text = re.sub(r"http\\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()

    return text

# Predict sentiment
if st.button("Predict Sentiment"):

    cleaned_tweet = clean_text(tweet)

    # Convert text into vector
    tweet_vector = vectorizer.transform([cleaned_tweet])

    # Prediction
    prediction = model.predict(tweet_vector)

    st.markdown("---")

    if prediction[0] == 1:

        st.success("😊 Positive Sentiment")

    else:

        st.error("😠 Negative Sentiment")

st.markdown("---")
uploaded = st.file_uploader("Upload a CSV, JSON, or JSONL Xquik export", type=["csv", "json", "jsonl"])
if uploaded:
    rows = load_xquik_rows(uploaded.getvalue())
    if not rows:
        st.warning("Upload does not contain a supported tweet text column.")
    else:
        predictions = []
        for row in rows:
            cleaned_tweet = clean_text(row["tweet"])
            tweet_vector = vectorizer.transform([cleaned_tweet])
            prediction = model.predict(tweet_vector)[0]
            predictions.append("Positive" if prediction == 1 else "Negative")

        result_df = pd.DataFrame(rows)
        result_df["sentiment"] = predictions
        st.dataframe(result_df, use_container_width=True)
        st.download_button(
            "Download batch results",
            result_df.to_csv(index=False),
            file_name="xquik_sentiment_results.csv",
            mime="text/csv",
        )

# Footer
st.markdown("---")
st.caption("Made with ❤️ using Python, Streamlit, NLP & Scikit-learn")
