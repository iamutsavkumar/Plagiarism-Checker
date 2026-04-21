from __future__ import annotations

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Optional

try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _SEMANTIC_AVAILABLE = True
except ImportError:
    _SEMANTIC_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

from .preprocessor import preprocess_text


# ---------------------------------------------------------------------------
# ✅ LOGGING (SAFE ADD)
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GLOBAL CACHE
# ---------------------------------------------------------------------------
_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.9
)

_semantic_model: Optional["SentenceTransformer"] = None


def _get_semantic_model():
    global _semantic_model
    if _semantic_model is None:
        try:
            _semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("✅ Semantic model loaded")
        except Exception as e:
            logger.exception("❌ Failed to load semantic model: %s", e)
            return None
    return _semantic_model


# ---------------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------------
@dataclass
class MatchedPair:
    sentence_a: str
    sentence_b: str
    score: float
    method: str


@dataclass
class SimilarityReport:
    jaccard_score: float
    tfidf_score: float
    semantic_score: float
    final_score: float
    matched_pairs: List[MatchedPair] = field(default_factory=list)
    weights_used: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 🔥 STRONG TEXT VALIDATION (ANTI-GARBAGE)
# ---------------------------------------------------------------------------
def _is_meaningful(text: str) -> bool:
    import re

    words = text.split()

    if len(words) < 12:
        return False

    meaningful = [w for w in words if len(w) >= 3]
    if len(meaningful) < len(words) * 0.7:
        return False

    vowel_words = [w for w in words if re.search(r'[aeiou]', w)]
    if len(vowel_words) < len(words) * 0.7:
        return False

    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < 0.6:
        return False

    avg_len = sum(len(w) for w in words) / len(words)
    if avg_len < 3:
        return False

    weird = [w for w in words if re.search(r'(.)\1\1', w)]
    if len(weird) > len(words) * 0.1:
        return False

    return True


# ---------------------------------------------------------------------------
# CORE FUNCTIONS
# ---------------------------------------------------------------------------
def jaccard_similarity(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def tfidf_cosine_similarity(clean_a: str, clean_b: str) -> float:
    if not clean_a.strip() or not clean_b.strip():
        return 0.0

    try:
        tfidf = _vectorizer.fit_transform([clean_a, clean_b])

        if tfidf.shape[1] < 8:
            return 0.0

        score = sk_cosine(tfidf[0], tfidf[1])[0][0]
        return float(np.clip(score, 0.0, 1.0))

    except Exception as e:
        logger.exception("TF-IDF failed: %s", e)
        return 0.0


def semantic_similarity(text_a: str, text_b: str) -> float:
    if not _SEMANTIC_AVAILABLE:
        return 0.0

    model = _get_semantic_model()
    if model is None:
        return 0.0

    try:
        emb = model.encode([text_a, text_b], convert_to_tensor=True)
        score = st_util.cos_sim(emb[0], emb[1]).item()
        return float(np.clip(score, 0.0, 1.0))
    except Exception as e:
        logger.exception("Semantic similarity failed: %s", e)
        return 0.0


# ---------------------------------------------------------------------------
# SENTENCE ALIGNMENT
# ---------------------------------------------------------------------------
def _align_sentences(sent_a, sent_b, threshold=0.5):
    pairs = []

    if not sent_a or not sent_b:
        return pairs

    sent_a = [s for s in sent_a if _is_meaningful(s)]
    sent_b = [s for s in sent_b if _is_meaningful(s)]

    if not sent_a or not sent_b:
        return pairs

    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(sent_a + sent_b)
    except Exception:
        return pairs

    n = len(sent_a)
    sim = sk_cosine(matrix[:n], matrix[n:])

    used = set()

    if _SEMANTIC_AVAILABLE:
        model = _get_semantic_model()
        if model:
            emb_a = model.encode(sent_a, convert_to_tensor=True)
            emb_b = model.encode(sent_b, convert_to_tensor=True)
        else:
            emb_a = emb_b = None

    for i in range(len(sent_a)):
        for j in np.argsort(-sim[i]):
            if j in used:
                continue

            tfidf_score = float(sim[i][j])

            if tfidf_score >= threshold:
                pairs.append(
                    MatchedPair(sent_a[i], sent_b[j], round(tfidf_score, 4), "tfidf")
                )
                used.add(j)
                break

            if _SEMANTIC_AVAILABLE and emb_a is not None:
                sem_score = st_util.cos_sim(emb_a[i], emb_b[j]).item()

                if sem_score >= 0.65 and tfidf_score > 0.1:
                    pairs.append(
                        MatchedPair(sent_a[i], sent_b[j], round(sem_score, 4), "semantic")
                    )
                    used.add(j)
                    break

    return sorted(
        [p for p in pairs if p.score >= 0.5],
        key=lambda x: x.score,
        reverse=True
    )


# ---------------------------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------------------------
def compute_similarity(text_a: str, text_b: str) -> SimilarityReport:

    if not text_a.strip() or not text_b.strip():
        return SimilarityReport(0, 0, 0, 0)

    if len(text_a.split()) < 12 or len(text_b.split()) < 12:
        return SimilarityReport(0, 0, 0, 0)

    if not _is_meaningful(text_a) or not _is_meaningful(text_b):
        return SimilarityReport(0, 0, 0, 0)

    prep_a = preprocess_text(text_a)
    prep_b = preprocess_text(text_b)

    clean_a = prep_a["clean_text"]
    clean_b = prep_b["clean_text"]

    if not clean_a.strip() or not clean_b.strip():
        return SimilarityReport(0, 0, 0, 0)

    if len(clean_a.split()) < 15 or len(clean_b.split()) < 15:
        return SimilarityReport(0, 0, 0, 0)

    jac = jaccard_similarity(prep_a["shingles"], prep_b["shingles"])
    tfidf = tfidf_cosine_similarity(clean_a, clean_b)
    sem = semantic_similarity(text_a, text_b) if _SEMANTIC_AVAILABLE else 0.0

    if _SEMANTIC_AVAILABLE:
        weights = {"jaccard": 0.1, "tfidf": 0.2, "semantic": 0.7}
    else:
        weights = {"jaccard": 0.3, "tfidf": 0.7, "semantic": 0.0}

    if sem < 0.3:
        tfidf *= 0.4

    final = (
        weights["jaccard"] * jac +
        weights["tfidf"] * tfidf +
        weights["semantic"] * sem
    )

    if tfidf > 0.95 and sem < 0.5:
        final *= 0.3

    if final > 0.9 and sem < 0.6:
        final *= 0.5

    final = float(np.clip(final, 0.0, 1.0))

    pairs = _align_sentences(prep_a["sentences"], prep_b["sentences"])

    return SimilarityReport(
        round(jac, 4),
        round(tfidf, 4),
        round(sem, 4),
        round(final, 4),
        pairs,
        weights,
    )