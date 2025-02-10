# MUST BE FIRST: Streamlit config
import streamlit as st

st.set_page_config(
    page_title="🏎️ Formula 1 Chatbot",
    page_icon="🏁",
    layout="centered"
)

# Clear session state
if 'initialized' not in st.session_state:
    st.session_state.clear()
    st.session_state.initialized = True

# Environment setup
import os
os.environ.update({
    "USER_AGENT": "F1-Chatbot/1.0 (Streamlit-App)",
    "KMP_DUPLICATE_LIB_OK": "TRUE"
})

# Essential imports
import re
import io
import numpy as np
import warnings
from langchain_community.document_loaders import WebBaseLoader
import google.generativeai as genai
import faiss
from google.oauth2 import service_account
from gtts import gTTS
import whisper
from datetime import datetime
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import tempfile

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize Gemini
try:
    credentials = service_account.Credentials.from_service_account_file('tough-dreamer-449219-p2-6a5c6c7ee6a3.json')
    genai.configure(credentials=credentials)
    model = genai.GenerativeModel('gemini-pro')
except Exception as e:
    st.error(f"Gemini initialization failed: {str(e)}")

# Website URLs
F1_URLS = [
    "https://www.formula1.com/en/latest.html",
    "https://www.espn.com/f1/",
    "https://www.bbc.com/sport/formula1",
    "https://www.planetf1.com",
    "https://www.motorsport.com/f1/",
    "https://www.racefans.net",
    "https://x.com/f1?lang=en",
    "https://www.fia.com/regulation/category/110",
    "https://www.statsf1.com/",
    "https://f1statblog.co.uk/",
    "https://www.f1-fansite.com/",
    "https://www.formula1.com/en/racing/2025.html",
    "https://www.autosport.com/f1/",
    "https://www.gpfans.com/",
    "https://www.f1technical.net/",
    "https://www.racingcircuits.info/formula-one/"
]
# Data processing
@st.cache_resource(show_spinner=False)
def load_f1_data():
    try:
        loader = WebBaseLoader(F1_URLS)
        documents = loader.load()
        
        def clean_text(text):
            return re.sub(r"[^\w\s]", "", re.sub(r"\s+", " ", text)).strip()
            
        return [" ".join(clean_text(doc.page_content).split()[i:i+500])
                for doc in documents
                for i in range(0, len(doc.page_content.split()), 500)]
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return []


# Embedding function
def get_embedding(text):
    try:
        return genai.embed_content(
            model="models/text-embedding-004",
            content=text
        )['embedding']
    except Exception as e:
        st.error(f"Embedding error: {str(e)}")
        return None

# FAISS index
@st.cache_resource(show_spinner=False)
def create_faiss_index(chunks):
    try:
        embeddings = [emb for chunk in chunks if (emb := get_embedding(chunk)) is not None]
        if not embeddings:
            raise ValueError("No valid embeddings generated")
            
        index = faiss.IndexFlatL2(len(embeddings[0]))
        index.add(np.array(embeddings).astype('float32'))
        return index
    except Exception as e:
        st.error(f"Index creation failed: {str(e)}")
        return None

#query generation
def route_query(query, index, chunks):
    try:
        if not query.strip():
            return "Error: Query cannot be empty."
        query_embedding = get_embedding(query)
        if query_embedding is None:
            return "Error: Failed to generate query embedding"
            
        # Search more chunks for better context
        _, indices = index.search(np.array(query_embedding).astype('float32').reshape(1, -1), 5)
        context = "\n\n".join([f"Context excerpt {i+1}: {chunks[idx]}" for i, idx in enumerate(indices[0])])
        
        # Structured prompt with clear instructions
        prompt = f"""**Analyze and Answer Instructions**
        You are an expert AI assistant. Follow these steps:

        1. **Understand**: Identify key elements in the question and context
        2. **Contextual Analysis**: 
        - Relevant context excerpts: 
        {context}
        - Connect context to question
        3. **Knowledge Integration**: Add technical details from your training
        4. **Reasoning Chain**: Explain step-by-step logic
        5. **Answer Formulation**: Provide final answer with clear formatting

        **Question:** {query}

        **Special Formatting:**
        - Put mathematical answers in \boxed{{}}
        - Highlight technical terms in **bold**
        - Use bullet points for steps
        - If uncertain, state confidence level (80% confident...)

            Now complete the analysis:"""
        
        response = model.generate_content(prompt)
        return response.text if response else "Error generating response"
    
    except Exception as e:
        return f"Processing error: {str(e)}"


# Main app
def main():
    st.title("🏎️ Formula 1 Chatbot")
    st.write("Real-time F1 insights powered by Gemini AI")
    
    # Load data
    with st.spinner("Loading Formula 1 data..."):
        chunks = load_f1_data()
        index = create_faiss_index(chunks)
        
    if not index or not chunks:
        st.error("Failed to initialize chatbot components")
        return
        
    # Text input
    query = st.text_input("Ask your F1 question:", key="text_input")
    audio_record=st.audio_input("Record your message:")
    if st.button("Submit_audio🏁",key="submit_audio") and audio_record:
        temp_dir = tempfile.gettempdir()
        temp_audio_path = os.path.join(temp_dir, "recorded_audio.wav")
        with open(temp_audio_path, "wb") as temp_audio_file:
            temp_audio_file.write(audio_record.read())
        
        transcription = whisper.load_model("small").transcribe(temp_audio_path, language="en")
        query_1 = transcription["text"]
        st.write(query_1)
        response = route_query(query_1, index, chunks)
        st.write("**Response:**", response)
        tts = gTTS(response)
        tts.save("response_audio.mp3")
        st.audio("response_audio.mp3",autoplay=False)

    if st.button("Submit_text 🏁",key="Submit_text") and query:
        with st.spinner("Analyzing..."):
            response = route_query(query, index, chunks)
            st.write("**Response:**", response)
            tts = gTTS(response)
            tts.save("response_text.mp3")
            st.audio("response_text.mp3",autoplay=False)
if __name__ == "__main__":
    main()
