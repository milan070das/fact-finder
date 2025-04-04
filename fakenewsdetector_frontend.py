import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

nltk.download('stopwords')

# Load the trained model and vectorizer (assumed pre-trained and saved as .pkl files)
@st.cache_resource
def load_model():
    vectorizer = joblib.load("vectorizer.pkl")
    model = joblib.load("fake_news_model.pkl")
    return vectorizer, model

vectorizer, model = load_model()

# Text preprocessing function
def preprocess_text(text):
    stemmer = PorterStemmer()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [stemmer.stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

# Streamlit UI
st.title("Fake News Detector")
st.write("Enter a news article or upload a CSV file to check for fake news.")

# User input text
user_input = st.text_area("Enter news text:")
if st.button("Check News"):
    if user_input:
        processed_text = preprocess_text(user_input)
        input_vector = vectorizer.transform([processed_text])
        prediction = model.predict(input_vector)[0]
        result = "Fake News" if prediction == 1 else "Real News"
        st.write(f"Prediction: {result}")
    else:
        st.warning("Please enter some text.")

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "text" in df.columns:
        df["processed"] = df["text"].apply(preprocess_text)
        input_vectors = vectorizer.transform(df["processed"])
        df["prediction"] = model.predict(input_vectors)
        df["result"] = df["prediction"].apply(lambda x: "Fake News" if x == 1 else "Real News")
        st.write(df[["text", "result"]])
    else:
        st.error("CSV must contain a 'text' column.")