
import os
import streamlit as st
from dotenv import load_dotenv
import whisper
import google.generativeai as genai
import textwrap
from IPython.display import Markdown

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(page_title="Audio to English")

st.header("🎙️ Translate Doctor-Patient Audio to English & Generate Insightful Summaries")

# Function to transcribe Arabic/Urdu audio
def transcribe_audio(audio_file):
    model = whisper.load_model("base")  # Load Whisper model

    # Save audio temporarily
    with open("temp_audio.mp3", "wb") as f:
        f.write(audio_file.read())

    # Transcribe using Whisper
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


# ========================= Q&A Chatbot Section ========================= #

st.header("💬 Generate Summary")

# Convert text to markdown format
def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# Function to get response from Gemini
def get_gemini_response(question):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(question)
    return response.text

# Initialize chatbot input
input_text = st.text_input(
    "👉 Copy the English transcription above and paste it here. Then write: 'Generate summary with keywords like patient gender, age, name, symptoms, and doctor’s advice.'", 
    key="input"
)
submit = st.button("generate summary")

# Process text-based questions
if submit:
    response = get_gemini_response(input_text)
    st.subheader("The Response is:")
    st.write(response)


