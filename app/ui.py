import streamlit as st
from io import BytesIO

def pdf_uploader():
    """
    Renders a file uploader widget for PDF files.
    
    Returns:
        List of uploaded PDF files.
    """
    return st.file_uploader("Upload a PDF file", type=["pdf"], accept_multiple_files=True,
                             help="Upload one or more PDF files to process.")

def speak_text(text: str, lang: str = "en", play: bool = True) -> bytes:
    """
    Generate an in-memory MP3 from `text` and optionally play it via Streamlit's audio player.
    
    Args:
        text (str): The text to convert to speech.
        lang (str): Language of the text (default is "en").
        play (bool): If True, play the audio automatically in the Streamlit app (default is True).

    Returns:
        bytes: MP3 audio data on success, None on failure.
    """
    if not text:
        st.error("Text cannot be empty.")
        return None
    
    try:
        from gtts import gTTS
    except ImportError:
        st.error("Text-to-speech is not available: please install `gTTS` in the Python environment running Streamlit (e.g., `pip install gTTS`).")
        return None
    
    mp3_fp = BytesIO()
    
    try:
        tts = gTTS(text=text, lang=lang)
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        data = mp3_fp.getvalue()  # Get all audio data from BytesIO
        
        if not data:
            st.error("TTS generated no audio data.")
            return None
        
        # Cache last audio in session for reuse
        st.session_state['last_tts'] = data  # Using a specific key
        
        if play:
            st.audio(data, format="audio/mp3")
        
        return data
        
    except Exception as e:
        st.error(f"TTS failed: {str(e)}")
        return None
