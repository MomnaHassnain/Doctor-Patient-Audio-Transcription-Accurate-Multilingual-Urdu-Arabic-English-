
import os
import streamlit as st
from dotenv import load_dotenv
import whisper  # OpenAI Whisper model for transcription
import torch  # Ensure Torch is installed for Whisper
import google.generativeai as genai  # Gemini AI for Q&A

# Load environment variables
load_dotenv()

# Set up Streamlit page
st.set_page_config(page_title="Doctor Patient conversation ")

st.header("🔍 Doctor Patient conversation")

# Function to transcribe audio using Whisper
def transcribe_audio(audio_file):
    model = whisper.load_model("base")  # Load Whisper model

    # Read audio file as bytes and save temporarily
    with open("temp_audio.mp3", "wb") as f:
        f.write(audio_file.read())

    # Transcribe using the saved file
    result = model.transcribe("temp_audio.mp3")
    
    return result["text"]

# Upload audio file
audio_file = st.file_uploader("Upload an audio file (MP3, WAV, M4A):", type=["mp3", "wav", "m4a"])

if audio_file:
    with st.spinner("Transcribing audio..."):
        transcription = transcribe_audio(audio_file)
    
    st.subheader("📜 Transcription")
    st.write(transcription)

# ----------------------------- Gemini AI Q&A Section -----------------------------
st.header("Doctor Patient conversation summary ")

# Retrieve and configure Gemini API Key
gemini_api_key = os.getenv("GEMINI_API_KEY")  # Corrected variable name
if not gemini_api_key:
    st.error("Gemini API Key is missing! Please check your .env file.")
else:
    genai.configure(api_key=gemini_api_key)

# Function to generate response using Gemini AI
def get_gemini_response(question):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(question)
    return response.text

# Input for Q&A
user_input = st.text_input("Enter your text: ", key="input")

# Submit button
if st.button("Generate summary"):
    if user_input.strip():
        with st.spinner("Generating response..."):
            response = get_gemini_response(user_input)

        st.subheader("🤖 AI Response:")
        st.write(response)
    else:
        st.warning("Please enter a question before submitting.")





# import os
# import streamlit as st
# from dotenv import load_dotenv
# import whisper  # OpenAI Whisper model for transcription
# import torch  # Ensure Torch is installed for Whisper

# # Load environment variables
# load_dotenv()

# # Function to transcribe audio using Whisper
# def transcribe_audio(audio_file):
#     model = whisper.load_model("base")  # Load Whisper model

#     # Read audio file as bytes
#     with open("temp_audio.mp3", "wb") as f:
#         f.write(audio_file.read())

#     # Transcribe using the saved file
#     result = model.transcribe("temp_audio.mp3")
    
#     return result["text"]

# # Streamlit UI
# st.set_page_config(page_title="Medical Transcription")

# st.header("🔍 Medical Transcription Generator")

# # Upload audio file
# audio_file = st.file_uploader("Upload an audio file (MP3, WAV, M4A):", type=["mp3", "wav", "m4a"])

# if audio_file:
#     with st.spinner("Transcribing audio..."):
#         transcription = transcribe_audio(audio_file)
    
#     st.subheader("📜 Transcription")
#     st.write(transcription)


# import pathlib
# import textwrap

# import google.generativeai as genai

# from IPython.display import display
# from IPython.display import Markdown


# def to_markdown(text):
#   text = text.replace('•', '  *')
#   return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# os.getenv("gemini_API_KEY")
# genai.configure(api_key=os.getenv("gemini_API_KEY"))

# ## Function to load OpenAI model and get respones

# def get_gemini_response(question):
#     model = genai.GenerativeModel('gemini-1.5-flash')
#     response = model.generate_content(question)
#     return response.text

# ##initialize our streamlit app

# st.set_page_config(page_title="Q&A Demo")

# st.header("Gemini Application")

# input=st.text_input("Input: ",key="input")


# submit=st.button("Ask the question")

# ## If ask button is clicked

# if submit:
    
#     response=get_gemini_response(input)
#     st.subheader("The Response is")
#     st.write(response)
