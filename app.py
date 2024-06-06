
import torch
import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
from gtts import gTTS
from deep_translator import GoogleTranslator
from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy.audio.io.AudioFileClip import AudioFileClip
import base64

# Model and tokenizer loading
checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, torch_dtype=torch.float32)

# File loader and preprocessing
def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        final_texts = final_texts + text.page_content
    return final_texts

# LLM pipeline
def llm_pipeline(filepath):
    pipe_sum = pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer,
        max_length=500,
        min_length=50)
    input_text = file_preprocessing(filepath)
    result = pipe_sum(input_text)
    result = result[0]['summary_text']
    return result

# Translation
def translate_text(text, target_language='en'):
    translator = GoogleTranslator(source='auto', target=target_language)
    translated_text = translator.translate(text)
    return translated_text

# Text-to-Audio
def text_to_audio(text, language='en'):
    tts = gTTS(text=text, lang=language, slow=False)
    return tts

@st.cache_data
# Function to display the PDF of a given file 
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit code 
st.set_page_config(layout="wide")

def main():
    st.title("Document Summarization to Video")

    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

    if uploaded_file is not None:
        # Language selection dropdown
        selected_language = st.selectbox("Select Language:", ["en", "ur", "es", "fr", "de", "te", "sa", "hi"])

        if st.button("Generate Video"):
            # Save the uploaded file to a temporary location
            filepath = "data/" + uploaded_file.name
            with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())

            # Summarize the document
            summary = llm_pipeline(filepath)

            # Translate the summary to the selected language
            translated_summary = translate_text(summary, target_language=selected_language)

            # Text-to-Audio
            audio_result = text_to_audio(translated_summary, language=selected_language)
            audio_path = "static/audio.mp3"
            audio_result.save(audio_path)

            # Load the video clip of a person speaking (assuming you have a background.mp4 file)
            video_path = os.path.join("static", "background.mp4")
            if not os.path.exists(video_path):
                st.error("Background video file not found.")
                return

            person_speaking_clip = VideoFileClip(video_path)
            
            # Load the audio file
            audio_clip = AudioFileClip(audio_path)

            # Calculate the number of times the video needs to be looped
            num_loops = max(1, int(audio_clip.duration / person_speaking_clip.duration))

            # Create a list of video clips by looping the person speaking clip
            video_clips = [person_speaking_clip] * num_loops

            # Concatenate the video clips to create the final video
            final_video_clip = concatenate_videoclips(video_clips, method="compose")

            # Set the audio of the final video clip to the desired audio
            final_video_clip = final_video_clip.set_audio(audio_clip)

            # Export the final video
            video_output_path = os.path.join("static", "output_video.mp4")
            final_video_clip.write_videofile(video_output_path, codec="libx264", audio_codec="aac", remove_temp=True)

            # Display the generated video
            st.video(video_output_path)

if __name__ == "__main__":
    main()

