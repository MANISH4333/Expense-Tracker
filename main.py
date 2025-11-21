# ============================================================================
# FILE: app/ui.py - UI Module with Fixed Audio Playback
# ============================================================================
import streamlit as st
from io import BytesIO

def pdf_uploader():
    """Upload PDF files"""
    return st.file_uploader(
        "Upload a PDF file", 
        type=["pdf"], 
        accept_multiple_files=True,
        help="Upload one or more PDF files to process."
    )

def speak_text(text: str, lang: str = "en", play: bool = False):
    """
    Generate an in-memory MP3 from text using gTTS.
    
    Args:
        text: The text to convert to speech
        lang: Language code (default: "en")
        play: If True, auto-play. If False, just return bytes (default: False)
    
    Returns:
        bytes: MP3 audio data on success, None on failure
    """
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        st.error("❌ Cannot generate audio: text is empty.")
        return None
    
    try:
        from gtts import gTTS
    except ImportError:
        st.error(
            "❌ gTTS is not installed. Please run:\n"
            "```bash\npip install gTTS\n```"
        )
        return None
    
    mp3_fp = BytesIO()
    try:
        # Generate TTS with error handling
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        data = mp3_fp.read()
        
        # Validate audio data
        if not data or len(data) == 0:
            st.error("❌ TTS generated no audio data.")
            return None
        
        # Cache in session for reuse
        st.session_state._last_tts = data
        
        # Only play if explicitly requested
        if play:
            st.audio(data, format="audio/mp3")
        
        return data
        
    except Exception as e:
        st.error(f"❌ TTS Error: {str(e)}")
        return None

# ============================================================================
# FILE: app_main.py - Fixed Main Application
# ============================================================================
import streamlit as st
from app.ui import pdf_uploader, speak_text
from app.pdf_utlis import extract_text_from_pdf
from app.vectorstore_utlis import create_faiss_index, retrive_relevant_docs
from app.chat_utlis import get_chat_model, ask_chat_model
from app.config import EURI_API_KEY
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
import uuid

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="JSS Expense Pro - Document Assistant",
    page_icon="💰",  # Changed to money icon
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

# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="color: #ff4b4b; font-size: 3rem; margin-bottom: 0.5rem;">💰  Expense Tracker</h1>
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

# Chat input
if prompt := st.chat_input("Ask about your Expense documents..."):
    timestamp = time.strftime("%H:%M")
    
    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": timestamp
    })
    
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
                    audio_bytes = speak_text(response, play=False)
                    
                    audio_id = None
                    if audio_bytes:
                        audio_id = f"audio_{uuid.uuid4().hex}"
                        st.session_state.audio_cache[audio_id] = audio_bytes
                        
                        # Display audio player
                        st.audio(audio_bytes, format="audio/mp3")
                        st.success(f"✅ Audio generated successfully ({len(audio_bytes)} bytes)")
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
