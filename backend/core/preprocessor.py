import re
import string
import os
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize


# ---------------------------------------------------------------------------
# ✅ SAFE NLTK SETUP (NO BREAK)
# ---------------------------------------------------------------------------
def ensure_nltk_data():
    """
    Ensures required NLTK resources are available.

    SAFE BEHAVIOR:
    - If NLTK_READY=true → skip downloads (production)
    - Otherwise → download if missing (your current behavior)
    """

    if os.getenv("NLTK_READY", "false").lower() == "true":
        return

    resources = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
    ]

    for path, pkg in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)

    # optional resource
    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        pass


ensure_nltk_data()

_STOP_WORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()


# ---------------------------------------------------------------------------
# 🔥 STRONG TEXT CLEANING (ANTI-OCR GARBAGE)
# ---------------------------------------------------------------------------
def _clean_text(text: str) -> str:
    text = text.lower()

    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', ' ', text)
    text = re.sub(r'\b(?!a\b|i\b)[a-z]\b', '', text)
    text = re.sub(r'\s+', ' ', text)

    words = text.split()

    words = [
        w for w in words
        if len(w) >= 3 and re.search(r'[aeiou]', w)
    ]

    return " ".join(words)


# ---------------------------------------------------------------------------
# 🔥 ULTRA-STRICT SENTENCE VALIDATION
# ---------------------------------------------------------------------------
def _is_valid_sentence(sentence: str) -> bool:
    words = sentence.split()

    if len(words) < 6:
        return False

    meaningful = [w for w in words if len(w) >= 3]
    if len(meaningful) < len(words) * 0.7:
        return False

    vowel_words = [w for w in words if re.search(r'[aeiou]', w)]
    if len(vowel_words) < len(words) * 0.7:
        return False

    weird = [w for w in words if re.search(r'(.)\1\1', w)]
    if len(weird) > len(words) * 0.1:
        return False

    if len(set(words)) / len(words) < 0.6:
        return False

    avg_len = sum(len(w) for w in words) / len(words)
    if avg_len < 3:
        return False

    return True


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------
def split_sentences(text: str) -> List[str]:
    """Robust sentence splitting with strong OCR noise filtering"""

    text = _clean_text(text)

    sentences = sent_tokenize(text)

    if len(sentences) <= 1:
        sentences = re.split(r'[.\n•\-]', text)

    sentences = [
        s.strip()
        for s in sentences
        if len(s.strip()) > 15 and _is_valid_sentence(s)
    ]

    return sentences


def tokenize(text: str, remove_stops: bool = True) -> List[str]:
    text = _clean_text(text)
    text = _strip_punctuation(text)

    try:
        tokens = word_tokenize(text)
    except LookupError:
        nltk.download("punkt", quiet=True)
        tokens = word_tokenize(text)

    tokens = [
        _LEMMATIZER.lemmatize(t)
        for t in tokens
        if t.isalpha() and len(t) >= 3
    ]

    if remove_stops:
        tokens = [t for t in tokens if t not in _STOP_WORDS]

    return tokens


def build_shingles(tokens: List[str], k: int = 3) -> set:
    if len(tokens) < k:
        return set(zip(*[tokens[i:] for i in range(len(tokens))]))
    return set(zip(*[tokens[i:] for i in range(k)]))


def preprocess_text(text: str, shingle_size: int = 3) -> dict:
    cleaned = _clean_text(text)

    sentences = split_sentences(cleaned)
    tokens = tokenize(cleaned)

    shingles = build_shingles(tokens, k=shingle_size)
    clean_text = " ".join(tokens)

    return {
        "sentences": sentences,
        "tokens": tokens,
        "shingles": shingles,
        "clean_text": clean_text,
    }


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _strip_punctuation(text: str) -> str:
    translator = str.maketrans("", "", string.punctuation.replace("'", ""))
    return text.translate(translator)