import os
import streamlit as st
from dotenv import load_dotenv
import whisper
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(page_title="Arabic/Urdu Audio to English Text")

st.header("🎙️ Arabic/Urdu Audio to English Text Converter")

# Function to transcribe Arabic/Urdu audio
def transcribe_audio(audio_file):
    model = whisper.load_model("base")  # Load Whisper model

    # Save audio temporarily
    with open("temp_audio.mp3", "wb") as f:
        f.write(audio_file.read())

    # Transcribe using Whisper (detects Arabic/Urdu, but does NOT auto-translate)
    result = model.transcribe("temp_audio.mp3")
    
    return result["text"]

# Function to translate text using Gemini AI
def translate_to_english(text):
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        st.error("⚠️ Gemini API Key is missing! Please check your .env file.")
        return None

    genai.configure(api_key=gemini_api_key)

    # Use Gemini AI for translation
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(f"Translate this Arabic/Urdu text to English: {text}")

    return response.text

# Upload audio file
audio_file = st.file_uploader("Upload an Arabic/Urdu audio file (MP3, WAV, M4A):", type=["mp3", "wav", "m4a"])

if audio_file:
    with st.spinner("Transcribing audio..."):
        original_text = transcribe_audio(audio_file)
    
    st.subheader("📜 Transcription (Original Arabic/Urdu)")
    st.write(original_text)

    # Translate to English
    with st.spinner("Translating to English..."):
        translated_text = translate_to_english(original_text)

    if translated_text:
        st.subheader("📖 English Translation")
        st.write(translated_text)
