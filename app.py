import streamlit as st
import joblib
import re
import pandas as pd

# Page settings
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
TEXT_COLUMN_CANDIDATES = (
    "text",
    "tweet",
    "tweet_text",
    "full_text",
    "content",
    "body",
)

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

def predict_sentiment(text):

    cleaned_tweet = clean_text(str(text))

    tweet_vector = vectorizer.transform([cleaned_tweet])

    prediction = model.predict(tweet_vector)

    if prediction[0] == 1:
        return "Positive"

    return "Negative"

def find_text_column(columns):

    normalized_columns = {
        str(column).strip().lower(): column
        for column in columns
    }

    for column in TEXT_COLUMN_CANDIDATES:
        if column in normalized_columns:
            return normalized_columns[column]

    return None

# Predict sentiment
if st.button("Predict Sentiment"):

    sentiment = predict_sentiment(tweet)

    st.markdown("---")

    if sentiment == "Positive":

        st.success("😊 Positive Sentiment")

    else:

        st.error("😠 Negative Sentiment")

# Footer
st.markdown("---")
st.subheader("Analyze Tweets from CSV")
st.write("Upload a CSV with a tweet text column, such as text, tweet, tweet_text, full_text, content, or body.")

uploaded_file = st.file_uploader("Upload tweet CSV", type=["csv"])

if uploaded_file is not None:

    csv_data = pd.read_csv(uploaded_file)

    text_column = find_text_column(csv_data.columns)

    if text_column is None:
        st.error("CSV must include a text, tweet, tweet_text, full_text, content, or body column.")
    else:
        results = csv_data.copy()
        results["predicted_sentiment"] = (
            results[text_column]
            .fillna("")
            .astype(str)
            .map(predict_sentiment)
        )

        st.dataframe(
            results[[text_column, "predicted_sentiment"]].head(100),
            use_container_width=True,
        )
        st.download_button(
            "Download Predictions",
            results.to_csv(index=False).encode("utf-8"),
            "tweet_sentiment_predictions.csv",
            "text/csv",
        )

st.markdown("---")
st.caption("Made with ❤️ using Python, Streamlit, NLP & Scikit-learn")
