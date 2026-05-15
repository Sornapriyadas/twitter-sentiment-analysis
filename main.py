import pandas as pd
import re
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Column names
columns = ['target', 'id', 'date', 'flag', 'user', 'text']

# Load dataset
df = pd.read_csv(
    "training.1600000.processed.noemoticon.csv",
    encoding='latin-1',
    names=columns
)

# Keep only needed columns
df = df[['target', 'text']]

# Convert 4 into 1
df['target'] = df['target'].replace(4, 1)

# Clean text
def clean_text(text):

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()

    return text

df['text'] = df['text'].apply(clean_text)

# Convert text into numbers
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df['text'])

y = df['target']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Save model
joblib.dump(model, "sentiment_model.pkl")

# Save vectorizer
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model and vectorizer saved successfully!")