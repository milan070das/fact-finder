import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import os
import webbrowser
from datetime import datetime
import pyttsx3
import speech_recognition as sr
import google.generativeai as genai
import joblib
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from newspaper import Article
from PIL import Image
import platform

# Set up Streamlit page configuration
st.set_page_config(page_title="Fact Finder", layout="wide")

# Sidebar Navigation
st.sidebar.image("logo.png", use_column_width=True)
st.sidebar.title("🚀 Navigation")
tabs = st.sidebar.radio("Go to:", [
    "🏠 Home",
    "📰 Fake News Detector",
    "🧠 News Research",
    "🤖 Newzie",
    "❓ FAQ",
    "👨‍💻 Developed By"
])

nltk_data_dir = os.path.expanduser('~/nltk_data')
nltk.data.path.append(nltk_data_dir)
nltk.download('punkt', download_dir=nltk_data_dir)

os.environ["GEMINI_API_KEY"] = "YOUR_GEMINI_API_KEY"
os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash')

engine = pyttsx3.init()

def say(text):
    engine.say(text)
    engine.runAndWait()

def take_command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("🎤 Listening...")
        audio = r.listen(source)
        try:
            return r.recognize_google(audio, language='en-in').lower()
        except:
            return ""

@st.cache_resource
def load_model():
    vectorizer = joblib.load("vectorizer.pkl")
    model = joblib.load("fake_news_model.pkl")
    return vectorizer, model

vectorizer, fake_news_model = load_model()

def preprocess_text(text):
    stemmer = nltk.stem.PorterStemmer()
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower().split()
    text = [stemmer.stem(word) for word in text if word not in nltk.corpus.stopwords.words('english')]
    return ' '.join(text)

def extract_text_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return f"Failed to extract content from the URL. Error: {e}"

def get_formatted_date():
    today = datetime.now()
    if platform.system() == "Windows":
        return today.strftime('%B %#d %Y')
    else:
        return today.strftime('%B %-d %Y')

# Home Page
if tabs == "🏠 Home":
    st.title("Fact Finder: One Stop solution for all your NEWS!")
    st.image("ai_image.jpg", use_column_width=True)

# Fake News Detector
elif tabs == "📰 Fake News Detector":
    st.title("📰 Fake News Detector")
    option = st.selectbox("Select input type:", ["Text", "URL"])
    user_input = None
    if option == "Text":
        user_input = st.text_area("Enter news text here...")
        st.write(user_input)
    elif option == "URL":
        url_input = st.text_input("Enter the URL of the news article:")
        if st.button("Extract Text"):
            user_input = extract_text_from_url(url_input)
            st.write("Extracted Text:")
            st.write(user_input)
            st.session_state['user_input'] = user_input
    
    if st.button("Check News"):
        if 'user_input' in st.session_state and st.session_state['user_input']:
            user_input = st.session_state['user_input']  # Retrieve stored input
            processed_text = preprocess_text(user_input)
            input_vector = vectorizer.transform([processed_text])
            prediction = fake_news_model.predict(input_vector)[0]
            result = "Fake News" if prediction == 1 else "Real News"
            if prediction == 1:
                st.error(f"🛑 Prediction: {result}")
            else:
                st.success(f"✅ Prediction: {result}")
        else:
            st.warning("Please provide valid input.")

# News Research
elif tabs == "🧠 News Research":
    st.title("🧠 News Research Tool")
    urls = [st.text_input(f"URL {i+1}", key=f"url_input_{i+1}") for i in range(3)]
    
    if st.button("Process URLs"):
        loader = UnstructuredURLLoader(urls=[url for url in urls if url])
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        docs = text_splitter.split_documents(data)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        faiss_index = FAISS.from_documents(docs, embeddings)
        faiss_index.save_local("faiss_index")
        st.success("Processing Complete. Ask your questions!")

# AI Assistant
elif tabs == "🤖 Newzie":
    st.title("🤖 Newzie: Your Personal AI Assistant")
    query = st.text_input("Enter your query:")
    if st.button("Ask Newzie"):
        response = model.generate_content(prompt=query)
        st.success(f"🤖 {response}")

# FAQ
elif tabs == "❓ FAQ":
    st.title("❓ FAQ")
    st.write("Provide common questions and answers here.")

# Developed By
elif tabs == "👨‍💻 Developed By":
    st.title("👨‍💻 Developed By")
    st.markdown("Developed by your team with contact information.")