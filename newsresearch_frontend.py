import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
import nltk
import os

# Set up NLTK data directory
nltk_data_dir = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

nltk.data.path.append(nltk_data_dir)
with st.spinner("Downloading NLTK data..."):
    nltk.download('punkt', download_dir=nltk_data_dir)  # Download only necessary data
    st.success("NLTK data downloaded successfully!")

# Set Google API Key directly (No secrets.toml required)
GOOGLE_API_KEY = "AIzaSyA0jX1JDZD7Tkhgm4crgO08bAjG9KFBUYc"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY  # Directly set API key

st.title("Gen AI: News Research Tool 📈")
st.sidebar.title("Enter Your URLs Here")

# Initialize session state
if "URLS_INPUT" not in st.session_state:
    st.session_state.URLS_INPUT = []
if "check" not in st.session_state:
    st.session_state.check = False
if "vectorindex_openai" not in st.session_state:
    st.session_state.vectorindex_openai = None

main_placeholder = st.empty()
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Get URLs from the sidebar
st.session_state.URLS_INPUT = [st.sidebar.text_input(f"URL {i+1}", key=f"url_input_{i+1}") for i in range(3)]
st.session_state.URLS_INPUT = [url for url in st.session_state.URLS_INPUT if url]  # Remove empty URLs

# Process URLs
if st.sidebar.button("Process URLs") and st.session_state.URLS_INPUT:
    loader = UnstructuredURLLoader(urls=st.session_state.URLS_INPUT)
    main_placeholder.text("Data Loading...Started...✅")
    data = loader.load()

    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅")
    docs = text_splitter.split_documents(data)

    # Create embeddings and save FAISS index
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    st.session_state.vectorindex_openai = FAISS.from_documents(docs, embeddings)
    st.session_state.vectorindex_openai.save_local("faiss_index")

    st.session_state.check = True
    main_placeholder.success("Process Complete. Now you can ask questions...✅")

# Query section
if st.session_state.check:
    query = st.text_input('You can ask your question now')
    if query:
        vectorindex_openai = FAISS.load_local(
            "faiss_index", 
            GoogleGenerativeAIEmbeddings(model="models/embedding-001"), 
            allow_dangerous_deserialization=True
        )
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorindex_openai.as_retriever())
        response = chain({"question": query}, return_only_outputs=True)
        st.write("Answer: ", response['answer'])
        st.write("Source: ", response['sources'])