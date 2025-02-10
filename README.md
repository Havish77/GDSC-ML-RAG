# GDSC-ML-RAG
The above code used to work properly buy due to dual booting I don't know how the code is not running and the time is up so had to submit the task without completion.. 
Formula 1 Chatbot 🏎️🏁
The Formula 1 Chatbot is an AI-powered assistant designed to provide real-time insights, news, and statistics about Formula 1 racing. Built using Streamlit, this chatbot integrates multiple technologies, including Google Gemini AI, FAISS (Facebook AI Similarity Search) indexing, and LangChain web scrapers, to fetch and process data from various F1-related sources. Users can interact with the chatbot through text input or voice commands, thanks to OpenAI Whisper for speech-to-text conversion and gTTS (Google Text-to-Speech) for audio responses.

Data Collection & Preprocessing

The chatbot scrapes Formula 1 news, stats, and updates from various online sources such as Formula1.com, ESPN, BBC Sport, PlanetF1, Motorsport.com, and more.
Using LangChain's WebBaseLoader, the text content from these sources is extracted, cleaned, and broken into manageable chunks for efficient processing.
Embedding & FAISS Indexing

The extracted text chunks are converted into numerical embeddings using Google's text-embedding models.
These embeddings are stored and indexed using FAISS, enabling fast and efficient similarity searches for relevant information based on user queries.
Query Processing & Response Generation

When a user submits a question (via text or voice), the chatbot converts the query into an embedding and searches the FAISS index to retrieve the most relevant context.
This retrieved data, along with the user query, is then passed to Google Gemini AI, which generates a well-structured and detailed response.
The response is formatted with bullet points, technical highlights, and confidence levels to ensure clarity and accuracy.
Speech-to-Text & Text-to-Speech Features

Users can record their queries using SoundDevice for real-time audio capture.
The recorded audio is transcribed using OpenAI's Whisper model.
The chatbot will convert its text responses into speech using gTTS, allowing users to listen to the answers.
