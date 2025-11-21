# ============================================================================
# FILE: main.py - Main Application with Speech-to-Text Integration
# ============================================================================
import streamlit as st
from app.ui import pdf_uploader, speak_text
from app.pdf_utlis import extract_text_from_pdf
from app.vectorstore_utlis import create_faiss_index, retrive_relevant_docs
from app.chat_utlis import get_chat_model, ask_chat_model
from app.config import EURI_API_KEY
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception as _langchain_err:
    try:
        # Fallback: some langchain versions expose a different splitter name
        from langchain.text_splitter import CharacterTextSplitter as RecursiveCharacterTextSplitter
    except Exception:
        st.error("The 'langchain' package is missing or incompatible in the deployed environment.\nEnsure 'langchain' is listed in `requirements.txt` and pin a compatible version.")
        raise _langchain_err

import time
import uuid

# NEW IMPORTS FOR SPEECH-TO-TEXT
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
try:
    from streamlit_mic_recorder import mic_recorder
except Exception:
    mic_recorder = None
    # Defer warning until after Streamlit is initialized; show a non-blocking message later.

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="JSS Expense Pro - Document Assistant",
    page_icon="💰", # Updated icon as per your latest code
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# STYLING
# ============================================================================
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #2b313e;
        color: white;
    }
    .chat-message.assistant {
        background-color: #f0f2f6;
        color: black;
    }
    .chat-message .timestamp {
        font-size: 0.8rem;
        opacity: 0.7;
        margin-top: 0.5rem;
    }
    .stButton > button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #ff3333;
    }
    .status-success {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
    }
    /* Custom style for the mic recorder button */
    .stMicRecorder button {
        background-color: #007bff; /* A nice blue */
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        width: 100%; /* Make it full width */
        margin-bottom: 1rem; /* Space below it */
    }
    .stMicRecorder button:hover {
        background-color: #0056b3;
    }
    /* Ensure chat input does not have default Streamlit label spacing */
    div.stText input {
        margin-bottom: 0px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_model" not in st.session_state:
    st.session_state.chat_model = None
if "audio_cache" not in st.session_state:
    st.session_state.audio_cache = {}

# NEW SESSION STATE VARS FOR SPEECH-TO-TEXT
if "recorded_audio_bytes" not in st.session_state:
    st.session_state.recorded_audio_bytes = None
if "stt_transcript" not in st.session_state:
    st.session_state.stt_transcript = ""
if "user_input_text_manual" not in st.session_state: # To store manually typed text if mic_recorder is not used
    st.session_state.user_input_text_manual = ""
if "_last_tts" not in st.session_state: # Initialize for speak_text in app/ui.py
    st.session_state._last_tts = None


# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="color: #ff4b4b; font-size: 3rem; margin-bottom: 0.5rem;">💰 Expense Tracker</h1>
    <p style="font-size: 1.2rem; color: #666; margin-bottom: 2rem;">Your Intelligent Expense Document Assistant</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - DOCUMENT UPLOAD
# ============================================================================
with st.sidebar:
    st.markdown("### 📁 Document Upload")
    st.markdown("Upload your expense documents to start chatting!")

    # TTS Test Button
    if st.button("🔊 Test TTS"):
        with st.spinner("Testing audio..."):
            audio_data = speak_text(
                "This is a text to speech test. If you hear this, audio is working correctly.",
                play=False
            )
            if audio_data:
                st.audio(audio_data, format="audio/mp3")
                st.success("✅ TTS test successful! Audio is working.")
            else:
                st.error("❌ TTS test failed. Check error messages above.")
    st.divider()

    # File Uploader
    uploaded_files = pdf_uploader()

    if uploaded_files:
        st.success(f"📄 {len(uploaded_files)} document(s) selected")

        # Process Documents Button
        if st.button("🚀 Process Documents", type="primary", use_container_width=True):
            with st.spinner("⏳ Processing your expense documents..."):
                try:
                    # Extract text from all PDFs
                    all_texts = []
                    progress_bar = st.progress(0)

                    for idx, file in enumerate(uploaded_files):
                        text = extract_text_from_pdf(file)
                        all_texts.append(text)
                        progress_bar.progress((idx + 1) / len(uploaded_files))

                    # Split texts into chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len,
                    )

                    chunks = []
                    for text in all_texts:
                        chunks.extend(text_splitter.split_text(text))

                    st.info(f"📊 Created {len(chunks)} text chunks")

                    # Create FAISS index
                    vectorstore = create_faiss_index(chunks)
                    st.session_state.vectorstore = vectorstore

                    # Initialize chat model
                    chat_model = get_chat_model(EURI_API_KEY)
                    st.session_state.chat_model = chat_model

                    st.success("✅ Documents processed successfully!")
                    st.balloons()

                except Exception as e:
                    st.error(f"❌ Error processing documents: {str(e)}")

# ============================================================================
# MAIN CHAT INTERFACE
# ============================================================================
st.markdown("### 💬 Chat with Your Expense Documents")

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        st.caption(message["timestamp"])

        # Display audio if available
        if message.get("audio_id") and message["audio_id"] in st.session_state.audio_cache:
            try:
                audio_data = st.session_state.audio_cache[message["audio_id"]]
                st.audio(audio_data, format="audio/mp3")
            except Exception as e:
                st.warning(f"⚠️ Could not play audio: {str(e)}")

# --- Speech-to-Text Recorder (Integration Point) ---
st.markdown("<div class='stMicRecorder'>", unsafe_allow_html=True) # Apply custom styling
audio_bytes_from_mic = None
if mic_recorder is not None:
    try:
        audio_bytes_from_mic = mic_recorder(
            start_prompt="Click to Speak",
            stop_prompt="Recording... Click to Stop",
            just_once=True, # Stop recording automatically after first audio segment
            use_container_width=True,
            key='speech_recorder'
        )
    except Exception as e:
        st.warning(f"Microphone recorder is unavailable: {e}")
        audio_bytes_from_mic = None
else:
    st.info("Microphone recorder not available in this environment.")
st.markdown("</div>", unsafe_allow_html=True)


# Process recorded audio (NEW LOGIC)
if audio_bytes_from_mic and st.session_state.recorded_audio_bytes != audio_bytes_from_mic:
    st.session_state.recorded_audio_bytes = audio_bytes_from_mic
    st.session_state.stt_transcript = "" # Clear previous transcript BEFORE transcribing new audio
    st.session_state.user_input_text_manual = "" # Also clear manual input when new speech is detected

    with st.spinner("Transcribing your speech..."):
        try:
            r = sr.Recognizer()
            # Convert raw bytes to AudioSegment, then to a format SpeechRecognition can read
            # You might need to install ffmpeg/libav for pydub to work with AudioSegment
            audio_segment = AudioSegment.from_wav(BytesIO(audio_bytes_from_mic))
            wav_file = BytesIO()
            audio_segment.export(wav_file, format="wav")
            wav_file.seek(0) # Reset stream position to the beginning

            with sr.AudioFile(wav_file) as source:
                audio_data = r.record(source)
                # Use Google Web Speech API for transcription (requires internet connection)
                text = r.recognize_google(audio_data)
                st.session_state.stt_transcript = text # Store the transcript
                st.toast("Speech recognized successfully!", icon="🎤")
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand the audio. Please try again.")
            st.session_state.stt_transcript = "" # Clear transcript on error too
        except sr.RequestError as e:
            st.error(f"Could not request results from speech recognition service; {e}. Check internet connection/API keys if applicable.")
            st.session_state.stt_transcript = "" # Clear transcript on error too
        except Exception as e:
            st.error(f"An unexpected error occurred during transcription: {e}")
            st.session_state.stt_transcript = "" # Clear transcript on error too
    st.experimental_rerun() # Re-run to update chat input with transcript


# Chat input logic
# Determine the value for st.chat_input:
# 1. If there's a fresh STT transcript, use it (highest priority).
# 2. Otherwise, use whatever the user typed manually (if any).
# 3. If neither, it will be an empty string.
current_input_value = ""
if st.session_state.stt_transcript:
    current_input_value = st.session_state.stt_transcript
elif st.session_state.user_input_text_manual:
    current_input_value = st.session_state.user_input_text_manual

# The `on_change` callback for st.chat_input to update user_input_text_manual
def update_input_state_on_change():
    # This function runs when the text in 'chat_input_box' changes (either typed or pre-filled).
    # We want to capture manual edits and clear the transcript if manual editing occurs.
    if st.session_state.chat_input_box != st.session_state.stt_transcript:
        # User has typed something different from the current transcript (or transcript was empty)
        st.session_state.user_input_text_manual = st.session_state.chat_input_box
        if st.session_state.stt_transcript: # If there WAS a transcript, clear it now that user is typing
            st.session_state.stt_transcript = ""
    # If st.session_state.chat_input_box == st.session_state.stt_transcript,
    # it means the input was just populated by STT on a rerun. In this case,
    # we don't clear the transcript yet, and user_input_text_manual should be reset.
    else:
        st.session_state.user_input_text_manual = ""


# Main chat input widget
prompt = st.chat_input(
    "Ask about your Expense documents...",
    value=current_input_value,
    key="chat_input_box", # Unique key for the widget
    on_change=update_input_state_on_change # Callback for handling input changes
)


if prompt: # This block runs when user hits enter or clicks send button
    timestamp = time.strftime("%H:%M")

    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": timestamp
    })

    # IMPORTANT: Clear both transcription and manual input after sending to prepare for next input
    st.session_state.stt_transcript = ""
    st.session_state.user_input_text_manual = ""

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption(timestamp)

    # Generate assistant response
    if st.session_state.vectorstore and st.session_state.chat_model:
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching documents and generating response..."):
                try:
                    # Retrieve relevant documents
                    relevant_docs = retrive_relevant_docs(
                        st.session_state.vectorstore,
                        prompt
                    )

                    # Create context
                    context = "\n\n".join([doc.page_content for doc in relevant_docs])

                    # Create system prompt
                    system_prompt = f"""You are an intelligent expense document assistant specializing in analyzing expense documents.
Based on the following expense documents, provide accurate and helpful answers to user questions.
- If the information is in the documents, provide a detailed answer with specific details.
- If the information is NOT in the documents, clearly state that.
- Always cite which part of the documents you're referencing when possible.
Expense Documents Context:
{context}
User Question: {prompt}
Please provide a comprehensive answer:"""

                    # Get response from LLM
                    response = ask_chat_model(st.session_state.chat_model, system_prompt)

                    # Display response
                    st.markdown(response)
                    st.caption(timestamp)

                    # Generate audio
                    st.info("🎵 Generating audio for this response...")
                    audio_bytes_tts = speak_text(response, play=False) # Renamed to avoid local var conflict if any
                    audio_id = None
                    if audio_bytes_tts:
                        audio_id = f"audio_{uuid.uuid4().hex}"
                        st.session_state.audio_cache[audio_id] = audio_bytes_tts

                        # Display audio player
                        st.audio(audio_bytes_tts, format="audio/mp3")
                        st.success(f"✅ Audio generated successfully ({len(audio_bytes_tts)} bytes)")
                    else:
                        st.warning("⚠️ Audio generation failed, but text response is available")

                    # Add assistant message to history
                    msg_entry = {
                        "role": "assistant",
                        "content": response,
                        "timestamp": timestamp
                    }
                    if audio_id:
                        msg_entry["audio_id"] = audio_id
                    st.session_state.messages.append(msg_entry)

                except Exception as e:
                    st.error(f"❌ Error generating response: {str(e)}")
    else:
        with st.chat_message("assistant"):
            st.error("⚠️ Please upload and process documents first!")
            st.caption(timestamp)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>🤖 Powered by JSSATEB | 💰 Expense Document Intelligence</p>
    <p style="font-size: 0.8rem; margin-top: 1rem;">
        <strong>Troubleshooting Audio:</strong><br>
        • Use the "Test TTS" button to verify audio is working<br>
        • Ensure your browser volume is not muted<br>
        • Check browser console (F12) for JavaScript errors
    </p>
</div>
""", unsafe_allow_html=True)