import os
import streamlit as st
from dotenv import load_dotenv
import whisper
import google.generativeai as genai
from gtts import gTTS

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(page_title="English to Urdu Speech Converter")

st.header("🎙️ English Audio to Urdu Speech Converter")

# Function to transcribe English audio using Whisper
def transcribe_english_audio(audio_file):
    model = whisper.load_model("base")  # Load Whisper model

    # Save audio temporarily
    with open("temp_audio.mp3", "wb") as f:
        f.write(audio_file.read())

    # Transcribe using Whisper
    result = model.transcribe("temp_audio.mp3", language="en")  
    
    return result["text"]

# Upload audio file
audio_file = st.file_uploader("Upload an English audio file (MP3, WAV, M4A):", type=["mp3", "wav", "m4a"])

if audio_file:
    with st.spinner("Transcribing English audio..."):
        english_transcription = transcribe_english_audio(audio_file)
    
    st.subheader("📜 English Transcription")
    st.write(english_transcription)

    # ----------------- Translation Section -----------------
    st.subheader("🌍 Translating to Urdu")

    # Configure Gemini AI
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        st.error("Gemini API Key is missing! Please check your .env file.")
    else:
        genai.configure(api_key=gemini_api_key)

        # Function to translate text to Urdu
        def translate_to_urdu(text):
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(f"Translate this English text to Urdu: {text}")
            return response.text

        # Translate
        with st.spinner("Translating..."):
            urdu_text = translate_to_urdu(english_transcription)

        st.subheader("📖 Urdu Translation")
        st.write(urdu_text)

        # ----------------- Text to Speech -----------------
        st.subheader("🔊 Convert Urdu Text to Speech")

        # Convert Urdu text to speech
        tts = gTTS(text=urdu_text, lang="ur")
        tts.save("urdu_audio.mp3")

        # Provide audio playback
        st.audio("urdu_audio.mp3", format="audio/mp3")

        # Provide a download button
        with open("urdu_audio.mp3", "rb") as file:
            st.download_button(label="📥 Download Urdu Audio", data=file, file_name="urdu_audio.mp3", mime="audio/mp3")
