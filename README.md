# 💬 Twitter Sentiment Analysis (AI/ML Project)

## 📌 Overview
Twitter Sentiment Analysis is an AI/ML-based NLP project that predicts whether a tweet expresses a Positive or Negative sentiment. By analyzing tweet text data, the model helps understand public opinion, customer feedback, and social media trends.

---

## 🚀 Features
- Predict tweet sentiment (Positive/Negative)
- Text preprocessing and cleaning
- NLP-based feature extraction using TF-IDF
- Machine Learning model training and evaluation
- Interactive Streamlit web application
- Real-time sentiment prediction

---

## 🧠 Technologies Used
- Python
- Pandas & NumPy
- Scikit-learn
- NLP (Natural Language Processing)
- TF-IDF Vectorizer
- Streamlit
- Joblib

---

## 📂 Project Structure

Twitter-Sentiment-Analysis/
│
├── training.1600000.processed.noemoticon.csv
├── main.py
├── app.py
├── xquik_export.py
├── test_xquik_export.py
├── sentiment_model.pkl
├── vectorizer.pkl
├── requirements.txt
└── README.md

---

## 📊 Dataset
The dataset contains tweets with sentiment labels.

### Columns:
- Target
- Tweet Text
- User
- Date
- ID

### Target Variable:
- 0 → Negative Sentiment
- 1 → Positive Sentiment

---

## ⚙️ Installation

### Clone Repository
```bash
git clone https://github.com/Sornapriyadas/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Train Model
```bash
python main.py
```

### Run Streamlit App
```bash
streamlit run app.py
```

### Analyze an Exported Xquik File
Upload a CSV, JSON, or JSONL export in the Streamlit app. Supported text columns include `tweet`, `text`, `full_text`, `tweet_text`, `content`, and `body`.
📈 Model Building

The project uses:

TF-IDF Vectorizer
Logistic Regression
Evaluation Metric:
Accuracy Score
📉 Results
Achieved approximately 79% model accuracy
Successfully predicts tweet sentiments in real-time
🔮 Future Improvements
Add Neutral sentiment prediction
Improve UI design
Deploy using Streamlit Cloud
Use Deep Learning models like LSTM
🤝 Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests.

📜 License

This project is licensed under the MIT License.

👩‍💻 Author

Your Name

GitHub: https://github.com/Sornapriyadas

⭐ Acknowledgements
Kaggle Twitter Sentiment Dataset
Scikit-learn Documentation
Streamlit Community
