import os
import streamlit as st
from dotenv import load_dotenv
import whisper
import google.generativeai as genai

# Load API key from .env
load_dotenv()
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# Configure Gemini
if not GEMINI_API_KEY:
    st.error("❌ Gemini API key missing in .env file!")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# UI config
st.set_page_config(page_title="Doctor-Patient Audio Assistant", layout="wide")
st.title("🩺 Doctor-Patient Audio Assistant")

# Load Whisper model
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("medium")

whisper_model = load_whisper_model()

# Sidebar
st.sidebar.header("🛠️ Select Audio Input")
input_method = st.sidebar.radio("Choose input method:", ["🎧 Record Live", "📁 Upload File"])

st.sidebar.header("🌐 Select Language")
language_option = st.sidebar.radio("Choose the language of the audio:", ["Auto Detect", "Urdu", "English", "Arabic"])

# Language mapping
language_map = {
    "Urdu": "ur",
    "English": "en",
    "Arabic": "ar",
}
lang_code = language_map.get(language_option, None)

# Audio input
audio_file = None
if input_method == "🎧 Record Live":
    st.subheader("🎧 Record Live Audio")
    audio_file = st.audio_input("Record your audio here:", key="live_audio")
elif input_method == "📁 Upload File":
    st.subheader("📁 Upload Audio File")
    audio_file = st.file_uploader("Upload MP3/WAV/M4A file:", type=["mp3", "wav", "m4a"])

# Process audio
if audio_file:
    st.success("🎵 Audio input received!")
    st.audio(audio_file, format='audio/mp3')

    # Save temp file
    with open("temp_audio.mp3", "wb") as f:
        f.write(audio_file.read())

    # Transcription
    with st.spinner("🔍 Transcribing audio using Whisper..."):
        if lang_code:
            result = whisper_model.transcribe("temp_audio.mp3", language=lang_code)
        else:
            result = whisper_model.transcribe("temp_audio.mp3")
        original_text = result["text"]
        detected_lang = result["language"].lower()

    gemini_model = genai.GenerativeModel("gemini-1.5-flash")

    # Display language and transcription
    st.markdown(f"🌐 **Detected Language:** `{detected_lang.upper()}`")
    st.subheader("📜 Original Transcription")
    st.write(original_text)

    # Default: no translation
    translated_text = original_text

    # Translate to English if needed
    if detected_lang != "en":
        if st.button("🔁 Translate to English"):
            with st.spinner("🔁 Translating to English using Gemini..."):
                translation_prompt = f"Translate this doctor-patient conversation to English:\n\n{original_text}"
                translated_text = gemini_model.generate_content(translation_prompt).text
            st.subheader("📖 English Translation")
            st.write(translated_text)

    st.download_button("⬇️ Download Transcription", translated_text, file_name="transcription.txt")

    # --- Summary Section ---
    st.header("🧠 Generate Summary")
    summary_prompt = (
        "Summarize the following doctor-patient conversation and extract key info such as:\n"
        "- Patient Name\n"
        "- Gender\n"
        "- Age\n"
        "- Symptoms\n"
        "- Doctor's Advice\n\n"
        "Conversation:\n"
        f"{translated_text}"
    )

    # Layout columns for all summary buttons
    col1, col2, col3 = st.columns(3)

    # English Summary
    with col1:
        if st.button("📋 Generate Summary in English"):
            with st.spinner("🧠 Generating English summary using Gemini..."):
                summary_text_en = gemini_model.generate_content(summary_prompt).text
            st.subheader("📌 English Summary")
            st.write(summary_text_en)
            st.download_button("⬇️ Download English Summary", summary_text_en, "summary_english.txt")

    # Urdu Summary
    with col2:
        if st.button("📋 Generate Summary in Urdu"):
            with st.spinner("🔁 Generating Urdu summary using Gemini..."):
                summary_text_en_urdu = gemini_model.generate_content(summary_prompt).text
                urdu_prompt = f"Translate the following medical summary into Urdu:\n\n{summary_text_en_urdu}"
                urdu_summary = gemini_model.generate_content(urdu_prompt).text
            st.subheader("📌 Urdu Summary")
            st.write(urdu_summary)
            st.download_button("⬇️ Download Urdu Summary", urdu_summary, "summary_urdu.txt")

    # Arabic Summary
    with col3:
        if st.button("📋 Generate Summary in Arabic"):
            with st.spinner("🔁 Generating Arabic summary using Gemini..."):
                summary_text_en_arabic = gemini_model.generate_content(summary_prompt).text
                arabic_prompt = f"Translate the following medical summary into Arabic:\n\n{summary_text_en_arabic}"
                arabic_summary = gemini_model.generate_content(arabic_prompt).text
            st.subheader("📌 Arabic Summary")
            st.write(arabic_summary)
            st.download_button("⬇️ Download Arabic Summary", arabic_summary, "summary_arabic.txt")
