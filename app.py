import streamlit as st
import vosk
import wave
import json
import os
from pydub import AudioSegment
from io import BytesIO
from reportlab.pdfgen import canvas

st.title("Basic Transcription")

uploaded_file = st.file_uploader("Upload an audio file", type=['wav', 'ogg', 'mp3'])

if uploaded_file is not None:
    audio = AudioSegment.from_file(uploaded_file)
    audio = audio.set_channels(1)  
    audio = audio.set_frame_rate(16000)  

    temp_wav = BytesIO()
    audio.export(temp_wav, format='wav')
    temp_wav.seek(0)


    model_path = 'small-en-in'
    model = vosk.Model(model_path)

    with st.spinner("Transcribing Audio"):
        with wave.open(temp_wav, 'rb') as wav_file:
            recognizer = vosk.KaldiRecognizer(model, wav_file.getframerate())
            while True:
                data = wav_file.readframes(4000)
                if len(data) == 0:
                    break
                recognizer.AcceptWaveform(data)

            result = recognizer.FinalResult()
            text_output = json.loads(result)
            transcribed_text = text_output['text']


            st.subheader("Transcribed Text")
            text_input = st.text_area("Edit the transcribed text", transcribed_text,height=200)

            if st.button("Download as PDF"):
                pdf_buffer = BytesIO()
                c = canvas.Canvas(pdf_buffer)
                c.setFont("Helvetica", 12)
                c.drawString(100, 700, text_input)
                c.save()

                st.download_button(
                    label="Download PDF",
                    data=pdf_buffer.getvalue(),
                    file_name="transcription.pdf",
                    mime="application/pdf"
                )

            if st.button("Download as TXT"):
                st.download_button(
                    label="Download TXT",
                    data=text_input,
                    file_name="transcription.txt",
                    mime="text/plain"
                )

