import streamlit as st
from io import BytesIO


def pdf_uploader():
    return st.file_uploader("Upload a PDF file", type=["pdf"], accept_multiple_files=True,
                            help="Upload one or more PDF files to process.")


def speak_text(text: str, lang: str = "en", play: bool = True):
    """Generate an in-memory MP3 from `text` and optionally play it via Streamlit's audio player.

    If `play` is False, the function returns the MP3 bytes without calling `st.audio`.
    Returns bytes on success or None on failure.
    """
    if not text:
        return None
    try:
        from gtts import gTTS
    except Exception:
        st.error("Text-to-speech is not available: please install `gTTS` in the Python environment running Streamlit (e.g. `pip install gTTS`) or run Streamlit inside your `medichatbot` conda env.")
        return None

    mp3_fp = BytesIO()
    try:
        tts = gTTS(text=text, lang=lang)
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        data = mp3_fp.read()
        if not data:
            st.error("TTS generated no audio data.")
            return None
        # cache last audio in session so it can be reused if needed
        st.session_state._last_tts = data
        if play:
            st.audio(data, format="audio/mp3")
        return data
    except Exception as e:
        st.error(f"TTS failed: {e}")
        return None
