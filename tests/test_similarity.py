"""
tests/test_similarity.py
------------------------
Unit and integration tests for the similarity engine.

Run with:
    pytest tests/ -v
"""

import pytest

# ── Core imports ──────────────────────────────────────────────
from backend.core.preprocessor import (
    build_shingles,
    split_sentences,
    tokenize,
)
from backend.core.similarity import (
    compute_similarity,
    jaccard_similarity,
    tfidf_cosine_similarity,
)


# ══════════════════════════════════════════════════════════════
# PREPROCESSOR TESTS
# ══════════════════════════════════════════════════════════════

class TestPreprocessor:
    def test_split_sentences_basic(self):
        text = "Hello world. This is a test. Final sentence!"
        sents = split_sentences(text)
        assert len(sents) == 3

    def test_tokenize_removes_stopwords(self):
        text = "The quick brown fox jumps over the lazy dog"
        tokens = tokenize(text, remove_stops=True)
        assert "the" not in tokens
        assert "over" not in tokens
        assert "fox" in tokens or "jump" in tokens  # lemmatized

    def test_tokenize_keeps_stops_when_disabled(self):
        text = "The quick brown fox"
        tokens = tokenize(text, remove_stops=False)
        assert "the" in tokens

    def test_shingles_size(self):
        tokens = ["a", "b", "c", "d", "e"]
        shingles = build_shingles(tokens, k=3)
        # Should be: (a,b,c), (b,c,d), (c,d,e)
        assert len(shingles) == 3

    def test_shingles_short_token_list(self):
        tokens = ["a", "b"]
        shingles = build_shingles(tokens, k=3)
        # fewer tokens than k → returns whatever pairs/triples possible
        assert isinstance(shingles, set)


# ══════════════════════════════════════════════════════════════
# SIMILARITY TESTS
# ══════════════════════════════════════════════════════════════

class TestJaccard:
    def test_identical_sets(self):
        s = {("a", "b", "c"), ("b", "c", "d")}
        assert jaccard_similarity(s, s) == pytest.approx(1.0)

    def test_disjoint_sets(self):
        a = {("x", "y", "z")}
        b = {("p", "q", "r")}
        assert jaccard_similarity(a, b) == pytest.approx(0.0)

    def test_empty_sets(self):
        assert jaccard_similarity(set(), set()) == 0.0

    def test_partial_overlap(self):
        a = {("a", "b"), ("b", "c"), ("c", "d")}
        b = {("b", "c"), ("c", "d"), ("d", "e")}
        score = jaccard_similarity(a, b)
        assert 0.0 < score < 1.0


class TestTfidf:
    def test_identical_texts(self):
        text = "The quick brown fox jumps over the lazy dog"
        score = tfidf_cosine_similarity(text, text)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_completely_different_texts(self):
        a = "quantum physics relativity spacetime"
        b = "chocolate cake baking flour sugar"
        score = tfidf_cosine_similarity(a, b)
        assert score < 0.2

    def test_empty_strings(self):
        score = tfidf_cosine_similarity("", "")
        assert score == 0.0


# ══════════════════════════════════════════════════════════════
# END-TO-END COMPUTE_SIMILARITY
# ══════════════════════════════════════════════════════════════

class TestComputeSimilarity:
    LOREM = (
        "Artificial intelligence is transforming the modern economy. "
        "Machine learning algorithms process vast amounts of data to find patterns. "
        "Companies investing in AI early gain competitive advantages."
    )

    PARAPHRASE = (
        "The modern economy is being reshaped by artificial intelligence. "
        "Advanced algorithms can analyse enormous datasets and uncover hidden trends. "
        "Organisations that adopt AI technology early stand to benefit significantly."
    )

    UNRELATED = (
        "The recipe calls for two cups of flour and a pinch of salt. "
        "Preheat the oven to 180 degrees Celsius before baking. "
        "Allow the cake to cool for thirty minutes before serving."
    )

    def test_high_similarity_for_paraphrase(self):
        report = compute_similarity(self.LOREM, self.PARAPHRASE)
        assert report.final_score >= 0.30, (
            f"Expected paraphrase to score ≥ 0.30, got {report.final_score}"
        )

    def test_low_similarity_for_unrelated(self):
        report = compute_similarity(self.LOREM, self.UNRELATED)
        assert report.final_score < 0.35, (
            f"Expected unrelated texts to score < 0.35, got {report.final_score}"
        )

    def test_identical_texts_score_near_one(self):
        report = compute_similarity(self.LOREM, self.LOREM)
        assert report.final_score >= 0.85

    def test_report_fields_present(self):
        report = compute_similarity(self.LOREM, self.PARAPHRASE)
        assert 0.0 <= report.jaccard_score  <= 1.0
        assert 0.0 <= report.tfidf_score    <= 1.0
        assert 0.0 <= report.semantic_score <= 1.0
        assert 0.0 <= report.final_score    <= 1.0
        assert isinstance(report.matched_pairs, list)

    def test_matched_pairs_have_correct_structure(self):
        report = compute_similarity(self.LOREM, self.PARAPHRASE)
        for pair in report.matched_pairs:
            assert isinstance(pair.sentence_a, str)
            assert isinstance(pair.sentence_b, str)
            assert 0.0 <= pair.score <= 1.0
            assert pair.method in ("tfidf", "semantic")


# ══════════════════════════════════════════════════════════════
# FASTAPI INTEGRATION (requires running server)
# ══════════════════════════════════════════════════════════════

class TestFastAPIIntegration:
    """
    These tests use FastAPI's TestClient and do NOT require a live server.
    """

    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from backend.main import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert "semantic_similarity" in data

    def test_check_plagiarism_success(self, client):
        payload = {
            "text_a": "Artificial intelligence is transforming every sector of the economy. Deep learning models can now recognise images with superhuman accuracy.",
            "text_b": "The economy is being transformed by artificial intelligence. Neural networks achieve better-than-human performance on image recognition tasks.",
        }
        res = client.post("/api/check-plagiarism", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "similarity_percent" in data
        assert 0.0 <= data["similarity_percent"] <= 100.0
        assert isinstance(data["matched_pairs"], list)

    def test_check_plagiarism_short_text_rejected(self, client):
        payload = {"text_a": "Hi", "text_b": "Hello"}
        res = client.post("/api/check-plagiarism", json=payload)
        assert res.status_code == 422

    def test_check_files_txt(self, client):
        import io
        file_a = ("file_a.txt", io.BytesIO(b"The quick brown fox jumps over the lazy dog. This is a test of the file upload endpoint."), "text/plain")
        file_b = ("file_b.txt", io.BytesIO(b"A fast brown fox leaps above the lazy dog. This checks whether the upload feature works correctly."), "text/plain")
        res = client.post(
            "/api/check-files",
            files={"file_a": file_a, "file_b": file_b},
        )
        assert res.status_code == 200
        data = res.json()
        assert "similarity_percent" in data
