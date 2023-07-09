import streamlit as st
import vosk
import wave
import json
from pydub import AudioSegment
from io import BytesIO
from reportlab.pdfgen import canvas
# import openai

# Step 1: Set up Streamlit application
st.title("Basic Transcription")

# Step 2: Create audio input component
uploaded_file = st.file_uploader("Upload an audio file", type=['wav', 'ogg', 'mp3'])

# Step 3: Transcribe audio using Vosk API
if uploaded_file is not None:
    # Convert audio file to required format
    audio = AudioSegment.from_file(uploaded_file)
    audio = audio.set_channels(1)  # Convert to mono
    audio = audio.set_frame_rate(16000)  # Set sample rate to 16000 Hz

    # Save audio to a temporary WAV file
    temp_wav = BytesIO()
    audio.export(temp_wav, format='wav')
    temp_wav.seek(0)


    # Load Vosk API model
    model_path = 'small-en-in'
    model = vosk.Model(model_path)

    # Transcribe audio
    with wave.open(temp_wav, 'rb') as wav_file:
        recognizer = vosk.KaldiRecognizer(model, wav_file.getframerate())
        while True:
            data = wav_file.readframes(4000)
            if len(data) == 0:
                break
            recognizer.AcceptWaveform(data)

        # Get the final transcription result
        result = recognizer.FinalResult()
        text_output = json.loads(result)
        transcribed_text = text_output['text']


        # Step 4: Display transcribed text
        st.subheader("Transcribed Text")
        text_input = st.text_area("Edit the transcribed text", transcribed_text,height=200)
        # Step 5: Enable text editing

        # Step 6: Download text in PDF or TXT format
        if st.button("Download as PDF"):
            # Create a PDF document
            pdf_buffer = BytesIO()
            c = canvas.Canvas(pdf_buffer)
            c.setFont("Helvetica", 12)
            c.drawString(100, 700, text_input)
            c.save()

            # Download the PDF file
            st.download_button(
                label="Download PDF",
                data=pdf_buffer.getvalue(),
                file_name="transcription.pdf",
                mime="application/pdf"
            )

        if st.button("Download as TXT"):
            # Download the text file
            st.download_button(
                label="Download TXT",
                data=text_input,
                file_name="transcription.txt",
                mime="text/plain"
            )

