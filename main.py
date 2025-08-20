# app.py
import io
import os
import string
from collections import Counter

import nltk
import streamlit as st
import speech_recognition as sr

# -----------------------------
# 1) NLTK: ensure required data
# -----------------------------
# We attempt lazy downloads only if missing (avoids re-downloading every run).
def _safe_nltk_download(pkg):
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg.split("/")[-1], quiet=True)

_safe_nltk_download("tokenizers/punkt")
_safe_nltk_download("tokenizers/punkt_tab")  # Added punkt_tab for sentence tokenization
_safe_nltk_download("corpora/wordnet")
_safe_nltk_download("corpora/omw-1.4")
_safe_nltk_download("corpora/stopwords")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()
PUNCT = set(string.punctuation)


# -----------------------------------------
# 2) Preprocessing and chatbot “knowledge”
# -----------------------------------------
def normalize_text(text: str):
    text = text.lower()
    tokens = word_tokenize(text)
    # remove punctuation + stopwords, then lemmatize
    cleaned = [
        LEMMATIZER.lemmatize(t)
        for t in tokens
        if (t not in PUNCT) and (t not in STOPWORDS) and t.isalpha()
    ]
    return cleaned

def build_knowledge_base(raw_text: str):
    """
    Split text into sentences, preprocess each sentence, and build
    token frequency vectors for simple cosine similarity.
    """
    sentences = [s.strip() for s in sent_tokenize(raw_text) if s.strip()]
    bow_vectors = [Counter(normalize_text(s)) for s in sentences]
    return sentences, bow_vectors

def cosine_sim(counts_a: Counter, counts_b: Counter) -> float:
    if not counts_a or not counts_b:
        return 0.0
    # dot product
    keys = set(counts_a.keys()) | set(counts_b.keys())
    dot = sum(counts_a[k] * counts_b[k] for k in keys)
    # magnitudes
    mag_a = sum(v * v for v in counts_a.values()) ** 0.5
    mag_b = sum(v * v for v in counts_b.values()) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)

def retrieve_best_response(user_text: str, sentences, bow_vectors):
    """
    Very simple retrieval: pick the sentence most similar to the user's input.
    """
    user_vec = Counter(normalize_text(user_text))
    best_i, best_score = -1, 0.0
    for i, vec in enumerate(bow_vectors):
        score = cosine_sim(user_vec, vec)
        if score > best_score:
            best_score = score
            best_i = i
    # Threshold to avoid irrelevant echoes
    if best_i == -1 or best_score < 0.05:
        return "I'm not sure about that yet. Could you rephrase the question?"
    return sentences[best_i]

def make_chatbot_response(user_text: str, sentences, bow_vectors):
    """
    Hook for customizing greetings, fallbacks, etc.
    """
    if not user_text.strip():
        return "Say something and I’ll try to help!"
    greetings = ("hello", "hi", "hey", "good morning", "good afternoon", "good evening")
    if any(g in user_text.lower() for g in greetings):
        return "Hello! Ask me anything about the uploaded text."
    return retrieve_best_response(user_text, sentences, bow_vectors)


# ---------------------------------------
# 3) Speech recognition helper (offline)
# ---------------------------------------

# Language support dictionary
SUPPORTED_LANGUAGES = {
    "English (US)": "en-US",
    "English (UK)": "en-GB",
    "French": "fr-FR",
    "Spanish": "es-ES",
    "German": "de-DE",
    "Italian": "it-IT",
    "Portuguese": "pt-PT",
    "Russian": "ru-RU",
    "Japanese": "ja-JP",
    "Chinese (Mandarin)": "zh-CN",
    "Arabic": "ar-SA"
}

def transcribe_speech(timeout=5, phrase_time_limit=10, language="en-US"):
    """
    Use default microphone. Requires a working mic on the machine running Streamlit.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Listening... (speak now)")
        # Adjust for ambient noise briefly
        recognizer.adjust_for_ambient_noise(source, duration=0.6)
        audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
    try:
        # Uses Google Web Speech API (no key needed for basic usage).
        text = recognizer.recognize_google(audio, language=language)
        return text
    except sr.WaitTimeoutError:
        raise RuntimeError("Listening timed out while waiting for phrase to start. Please try speaking again.")
    except sr.UnknownValueError:
        raise RuntimeError("Sorry, I couldn't understand the audio. Please try speaking more clearly or adjusting your microphone settings.")
    except sr.RequestError as e:
        raise RuntimeError(f"Speech service error: {e}. Please check your internet connection and try again.")


# ------------------------
# 4) Streamlit UI / State
# ------------------------
st.set_page_config(page_title="Speech-enabled Chatbot", page_icon="💬", layout="centered")
st.title("💬 Speech-Enabled Chatbot")

# Sidebar: load corpus
st.sidebar.header("Corpus")
uploaded = st.sidebar.file_uploader("Upload a .txt file (recommended)", type=["txt"])
default_corpus_path = "chatbot_corpus.txt"

if "corpus_text" not in st.session_state:
    # Load uploaded file if present; else try local file; else use a tiny demo
    if uploaded is not None:
        st.session_state.corpus_text = uploaded.read().decode("utf-8", errors="ignore")
    elif os.path.exists(default_corpus_path):
        st.session_state.corpus_text = open(default_corpus_path, "r", encoding="utf-8", errors="ignore").read()
    else:
        st.session_state.corpus_text = (
            "This is a small demo corpus. "
            "You can upload your own text file in the sidebar to teach me more. "
            "The chatbot retrieves the most relevant sentence from the corpus to respond."
        )

# Language selection
if "selected_language" not in st.session_state:
    st.session_state.selected_language = "en-US"

st.sidebar.header("Language")
selected_language_name = st.sidebar.selectbox(
    "Select language for speech recognition:",
    list(SUPPORTED_LANGUAGES.keys()),
    index=list(SUPPORTED_LANGUAGES.values()).index(st.session_state.selected_language)
)
st.session_state.selected_language = SUPPORTED_LANGUAGES[selected_language_name]

# Build knowledge base whenever corpus changes (via a key bump)
if "kb_cache" not in st.session_state:
    st.session_state.kb_cache = None
if st.sidebar.button("Rebuild knowledge base"):
    st.session_state.kb_cache = None

if st.session_state.kb_cache is None:
    sentences, bow_vectors = build_knowledge_base(st.session_state.corpus_text)
    st.session_state.kb_cache = (sentences, bow_vectors)
else:
    sentences, bow_vectors = st.session_state.kb_cache

st.sidebar.caption(f"Knowledge sentences: **{len(sentences)}**")

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []  # list of (role, text)

# ------------------------
# 5) Input: text or speech
# ------------------------
st.subheader("Ask by typing or speaking")

col1, col2 = st.columns(2)
with col1:
    user_text = st.text_input("Type your message", value="", placeholder="e.g., What does the text say about X?")
with col2:
    do_speech = st.button("🎙️ Speak instead")

spoken_text = None
if do_speech:
    try:
        spoken_text = transcribe_speech(language=st.session_state.selected_language)
        st.success(f"Transcribed: “{spoken_text}”")
    except Exception as e:
        st.error(str(e))

final_query = user_text.strip() if user_text.strip() else (spoken_text or "").strip()

# Submit
send = st.button("Send")

if send:
    if final_query:
        st.session_state.history.append(("user", final_query))
        bot = make_chatbot_response(final_query, sentences, bow_vectors)
        st.session_state.history.append(("bot", bot))
    else:
        st.warning("Please type or speak a message first.")

# ------------------------
# 6) Display conversation
# ------------------------
if st.session_state.history:
    st.markdown("---")
    for role, text in st.session_state.history:
        if role == "user":
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**Bot:** {text}")

# Footer hint
st.caption(
    "Tip: Upload a richer .txt corpus in the sidebar to improve responses. "
    "For speech, make sure a microphone is connected and accessible."
)