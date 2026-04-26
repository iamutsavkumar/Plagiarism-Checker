from __future__ import annotations

import time
import re
import hashlib
import os
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator

from ..core.similarity import SimilarityReport, compute_similarity
from ..utils.file_extractor import extract_text

router = APIRouter(prefix="/api", tags=["plagiarism"])


# ---------------------------------------------------------------------------
# ✅ HEALTH ROUTE
# ---------------------------------------------------------------------------
@router.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# ✅ LIMITS (ENV-BASED, SAFE)
# ---------------------------------------------------------------------------
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", 50_000))
MAX_FILE_BYTES = int(os.getenv("MAX_FILE_BYTES", 5 * 1024 * 1024))


# ---------------------------------------------------------------------------
# SCHEMAS
# ---------------------------------------------------------------------------
class TextCompareRequest(BaseModel):
    text_a: str = Field(..., min_length=20, max_length=MAX_TEXT_CHARS)
    text_b: str = Field(..., min_length=20, max_length=MAX_TEXT_CHARS)

    @field_validator("text_a", "text_b")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Text must not be blank.")
        return v.strip()


class MatchedPairOut(BaseModel):
    sentence_a: str
    sentence_b: str
    score: float
    method: str


class SimilarityResponse(BaseModel):
    similarity_percent: float
    confidence: str
    jaccard_score: float
    tfidf_score: float
    semantic_score: float
    semantic_available: bool
    matched_pairs: list[MatchedPairOut]
    weights_used: dict
    processing_time_ms: float


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def _are_files_identical(bytes_a: bytes, bytes_b: bytes) -> bool:
    return hashlib.md5(bytes_a).hexdigest() == hashlib.md5(bytes_b).hexdigest()


def _report_to_response(report: SimilarityReport, elapsed_ms: float) -> SimilarityResponse:
    from ..core.similarity import _SEMANTIC_AVAILABLE

    pairs = report.matched_pairs[:20]

    pairs_out = [
        MatchedPairOut(
            sentence_a=p.sentence_a,
            sentence_b=p.sentence_b,
            score=p.score,
            method=p.method,
        )
        for p in pairs
    ]

    confidence = (
        "low" if report.final_score < 0.3
        else "medium" if report.final_score < 0.7
        else "high"
    )

    return SimilarityResponse(
        similarity_percent=round(report.final_score * 100, 2),
        confidence=confidence,
        jaccard_score=report.jaccard_score,
        tfidf_score=report.tfidf_score,
        semantic_score=report.semantic_score,
        semantic_available=_SEMANTIC_AVAILABLE,
        matched_pairs=pairs_out,
        weights_used=report.weights_used,
        processing_time_ms=round(elapsed_ms, 1),
    )


# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------
def _is_valid_text(text: str) -> bool:
    words = text.split()

    if len(words) < 12:
        return False

    meaningful = [w for w in words if len(w) >= 3]
    if len(meaningful) < len(words) * 0.7:
        return False

    vowel_words = [w for w in words if re.search(r'[aeiou]', w)]
    if len(vowel_words) < len(words) * 0.7:
        return False

    if len(set(words)) / len(words) < 0.6:
        return False

    if sum(len(w) for w in words) / len(words) < 3:
        return False

    return True


def _validate_text(text: str, label: str) -> None:
    cleaned = text.strip()

    if not cleaned:
        raise HTTPException(422, f"{label} contains no readable text.")

    if len(cleaned.split()) < 15:
        raise HTTPException(422, f"{label} text too weak after OCR.")

    if not _is_valid_text(cleaned):
        raise HTTPException(422, f"{label} text quality too low.")


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------
@router.post("/check-plagiarism", response_model=SimilarityResponse)
async def check_plagiarism(body: TextCompareRequest) -> SimilarityResponse:
    t0 = time.perf_counter()

    _validate_text(body.text_a, "Text A")
    _validate_text(body.text_b, "Text B")

    report = compute_similarity(body.text_a, body.text_b)

    if report is None:
        raise HTTPException(500, "Similarity computation failed.")

    return _report_to_response(report, (time.perf_counter() - t0) * 1000)


@router.post("/check-files", response_model=SimilarityResponse)
async def check_files(
    file_a: Annotated[UploadFile, File()],
    file_b: Annotated[UploadFile, File()],
) -> SimilarityResponse:

    t0 = time.perf_counter()

    bytes_a = await file_a.read()
    bytes_b = await file_b.read()

    if len(bytes_a) > MAX_FILE_BYTES or len(bytes_b) > MAX_FILE_BYTES:
        raise HTTPException(413, "File exceeds size limit.")

    is_identical = _are_files_identical(bytes_a, bytes_b)

    text_a = extract_text(bytes_a, file_a.filename or "file_a")
    text_b = extract_text(bytes_b, file_b.filename or "file_b")

    if not text_a or not text_b:
        raise HTTPException(422, "OCR failed.")

    _validate_text(text_a, "File A")
    _validate_text(text_b, "File B")

    report = compute_similarity(text_a, text_b)

    if report is None:
        raise HTTPException(500, "Similarity failed.")

    # ✅ identical file boost (unchanged logic)
    if is_identical:
        report.final_score = 1.0
        report.jaccard_score = 1.0
        report.tfidf_score = 1.0
        report.semantic_score = max(report.semantic_score, 0.99)

    return _report_to_response(report, (time.perf_counter() - t0) * 1000)
