import os
import streamlit as st
from dotenv import load_dotenv
import whisper
import google.generativeai as genai
from gtts import gTTS

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(page_title="Audio Transcription & Arabic Speech")

st.header("🎙️ Audio to Arabic Speech Converter")

# Function to transcribe audio using Whisper
def transcribe_audio(audio_file):
    model = whisper.load_model("base")  # Load Whisper model

    # Save audio temporarily
    with open("temp_audio.mp3", "wb") as f:
        f.write(audio_file.read())

    # Transcribe using Whisper
    result = model.transcribe("temp_audio.mp3")
    
    return result["text"]

# Upload audio file
audio_file = st.file_uploader("Upload an audio file (MP3, WAV, M4A):", type=["mp3", "wav", "m4a"])

if audio_file:
    with st.spinner("Transcribing audio..."):
        transcription = transcribe_audio(audio_file)
    
    st.subheader("📜 Transcription")
    st.write(transcription)

    # ----------------- Translation Section -----------------
    st.subheader("🌍 Translating to Arabic")

    # Configure Gemini AI
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        st.error("Gemini API Key is missing! Please check your .env file.")
    else:
        genai.configure(api_key=gemini_api_key)

        # Function to translate text
        def translate_to_arabic(text):
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(f"Translate this to Arabic: {text}")
            return response.text

        # Translate
        with st.spinner("Translating..."):
            arabic_text = translate_to_arabic(transcription)

        st.subheader("📖 Arabic Translation")
        st.write(arabic_text)

        # ----------------- Text to Speech -----------------
        st.subheader("🔊 Convert Arabic Text to Speech")

        # Convert Arabic text to speech
        tts = gTTS(text=arabic_text, lang="ar")
        tts.save("arabic_audio.mp3")

        # Provide audio playback
        st.audio("arabic_audio.mp3", format="audio/mp3")

        # Provide a download button
        with open("arabic_audio.mp3", "rb") as file:
            st.download_button(label="📥 Download Arabic Audio", data=file, file_name="arabic_audio.mp3", mime="audio/mp3")
