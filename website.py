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
from PIL import Image
from newspaper import Article
import platform
import nltk

nltk.download('stopwords')

# Set up Streamlit page configuration at the very beginning
st.set_page_config(page_title="Fact Finder", layout="wide")

# Sidebar Navigation with Icons
st.sidebar.image("logo.png", use_container_width=True)
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
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

nltk.data.path.append(nltk_data_dir)
with st.spinner("Downloading NLTK data..."):
    nltk.download('punkt', download_dir=nltk_data_dir)  # Download only necessary data

# Set up API key
os.environ["GEMINI_API_KEY"] = "AIzaSyA0jX1JDZD7Tkhgm4crgO08bAjG9KFBUYc"  # Replace with a secure method

GOOGLE_API_KEY = "AIzaSyA0jX1JDZD7Tkhgm4crgO08bAjG9KFBUYc"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# Initialize text-to-speech
engine = pyttsx3.init()

def say(text):
    """Converts text to speech."""
    engine.say(text)
    engine.runAndWait()

def take_command():
    """Captures user's voice input and converts it to text."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("🎤 Listening...")
        audio = r.listen(source)
        try:
            return r.recognize_google(audio, language='en-in').lower()
        except:
            return ""

def ai_response(prompt):
    """Generates AI response using Gemini API."""
    response = model.generate_content(prompt)
    return response.text.strip()

# Load the trained model and vectorizer (Fake News Detector)
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

# Home Page
if tabs == "🏠 Home":
    st.title("Fact Finder: One Stop solution for all your NEWS!")
    st.markdown("""
    **Explore AI-driven tools:**
    - 📰 **Fake News Detector**: Verify news credibility.
    - 🧠 **News Research**: Get a quick view of your news article.
    - 🤖 **Newzie**: Interact with an AI chatbot.
    """)
    st.image("ai_image.jpg", use_container_width=True)

# Fake News Detector
elif tabs == "📰 Fake News Detector":
    st.title("📰 Fake News Detector")
    option = st.selectbox("Select input type:", ["Text", "URL"])
    user_input = None
    if option == "Text":
        user_input = st.text_area("Enter news text here...")
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

# Gen AI News Research
# elif tabs == "🧠 News Research":
#     st.title("News Research Tool")
#     urls = [st.sidebar.text_input(f"URL {i+1}", key=f"url_input_{i+1}") for i in range(3)]
#     if st.sidebar.button("Process URLs") and any(urls):
#         st.write("Processing URLs...")
#         loader = UnstructuredURLLoader(urls=urls)
#         data = loader.load()
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
#         docs = text_splitter.split_documents(data)
#         embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#         faiss_index = FAISS.from_documents(docs, embeddings)
#         faiss_index.save_local("faiss_index")
#         st.success("Processing Complete. Ask your questions!")
elif tabs == "🧠 News Research":
    st.title("🧠News Research")
    
    st.write("Enter atmost 3 URLs to process:")
    urls = [st.text_input(f"URL {i+1}", key=f"url_input_{i+1}") for i in range(3)]
    
    if st.button("Process URLs") and any(urls):
        st.write("Processing URLs...")
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        docs = text_splitter.split_documents(data)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        faiss_index = FAISS.from_documents(docs, embeddings)
        faiss_index.save_local("faiss_index")
        
        st.success("Processing Complete. Ask your questions!")
    st.write('⭐Important Instruction: Please provide sensible questions relevant to the topic. Please refrain from asking irrelevant questions.')
    query = st.text_input('Ask a question:', placeholder="What do you want to know?")
    if query:
        vectorindex_openai = FAISS.load_local("faiss_index", GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0,max_tokens=None,timeout=None,max_retries=2,), retriever=vectorindex_openai.as_retriever())
        response = chain({"question": query}, return_only_outputs=True)
        with st.expander("Sources"):
            st.write(response['sources'])
        st.success(f"Answer: {response['answer']}")

# AI Assistant
elif tabs == "🤖 Newzie":
    st.title("🤖 Hello! I am Newzie, how can I help you today? ")
    query = st.text_input("Enter your query:", placeholder="Ask me anything!")
    st.write('⭐Important Instruction: Please provide sensible questions relevant to the topic. Please refrain from asking irrelevant questions.')
    use_voice = st.button("🎤 Use Voice Input")
    if use_voice:
        query = take_command()
        st.write(f"**You said:** {query}")
    if "the time" in query:
        strfTime = datetime.now().strftime("%H:%M:%S")
        st.write('Time: ',strfTime)
        today = datetime.now()
        formatted_date = today.strftime('%B %#d / %Y')
        st.write('Date: ',formatted_date)
    elif query:
        response = ai_response(query)
        st.success(f"🤖 {response}")

elif tabs == "🤖 Newzie":
    import queue
    import numpy as np
    from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
    import speech_recognition as sr

    st.title("🤖 Hello! I am Newzie, how can I help you today?")
    st.write('⭐Important Instruction: Please provide sensible questions relevant to the topic. Please refrain from asking irrelevant questions.')

    query = st.text_input("Enter your query:", placeholder="Ask me anything!")

    # Streamlit-WebRTC AudioProcessor
    class AudioProcessor(AudioProcessorBase):
        def __init__(self) -> None:
            self.buffer = queue.Queue()

        def recv(self, frame):
            audio = frame.to_ndarray()
            self.buffer.put(audio)
            return frame

    # Start webrtc streamer
    ctx = webrtc_streamer(
        key="voice",
        mode="sendonly",
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

    if ctx.audio_processor:
        audio_buffer = []
        while not ctx.audio_processor.buffer.empty():
            audio_buffer.append(ctx.audio_processor.buffer.get())

        if audio_buffer:
            audio_np = np.concatenate(audio_buffer, axis=0).astype(np.int16)
            audio_bytes = audio_np.tobytes()

            recognizer = sr.Recognizer()
            audio_data = sr.AudioData(audio_bytes, sample_rate=48000, sample_width=2)

            try:
                transcript = recognizer.recognize_google(audio_data)
                st.success(f"🗣️ You said: **{transcript}**")
                query = transcript  # override query with voice input
            except sr.UnknownValueError:
                st.warning("Sorry, I couldn't understand that.")
            except sr.RequestError as e:
                st.error(f"Could not request results; {e}")

    # Respond to text or voice query
    if query:
        if "the time" in query.lower():
            strfTime = datetime.now().strftime("%H:%M:%S")
            today = datetime.now()
            formatted_date = today.strftime('%B %-d, %Y') if platform.system() != 'Windows' else today.strftime('%B %#d, %Y')
            st.write("🕒 Time:", strfTime)
            st.write("📅 Date:", formatted_date)
        else:
            response = ai_response(query)
            st.success(f"🤖 {response}")

# Developed By Section
elif tabs == "👨‍💻 Developed By":
    st.title("👨‍💻 Developed By")
    st.markdown("""
    - 🎓 **Milan Das** Ph: 84820 09231
    - 🎓 **Kusal Hor** Ph: 82505 60516
    - 🎓 **Asmita Goswami** Ph: 98305 41045
    - 🎓 **Ishita Maity** Ph: 89182 35764
    - 🎓 **Subhodeep Routh** Ph: 83889 29595
    
    🤝 Thank you for using Fact Finder!
    """)
    st.write("😄**Don't hesitate to share your feedback with us and tell us how we can do better!**")
    image = Image.open("qr.jpg")
    image = image.resize((200, 200))
    st.image(image)
    # if st.button("Feedback Form"):
    #     webbrowser.open('https://forms.gle/XMk5oLhjoAgXoFPT9')
    if st.button("Feedback"):
        st.write("[Fill Feedback Form](https://forms.gle/XMk5oLhjoAgXoFPT9)")

