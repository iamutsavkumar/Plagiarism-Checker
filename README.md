# ⬡ PlagiarismCheck — Multi-Strategy NLP Similarity Engine

> Detect verbatim copies **and** paraphrased content using a weighted blend of  
> shingling, TF-IDF cosine distance, and semantic sentence embeddings.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Tech Stack](#tech-stack)
4. [Project Structure](#project-structure)
5. [Quick Start](#quick-start)
6. [API Reference](#api-reference)
7. [How It Works](#how-it-works)
8. [Sample Input / Output](#sample-input--output)
9. [Docker Deployment](#docker-deployment)
10. [Cloud Deployment (Render)](#cloud-deployment-render)
11. [Running Tests](#running-tests)
12. [Configuration](#configuration)
13. [Roadmap](#roadmap)
14. [License](#license)

---

## Overview

PlagiarismCheck is a **production-ready plagiarism detection engine** that goes beyond simple keyword matching. It combines three NLP strategies:

| Strategy | What it catches | Weight |
|---|---|---|
| **Jaccard shingling** (k-grams) | Verbatim / near-verbatim copies | 25 % |
| **TF-IDF cosine similarity** | Paraphrasing with shared vocabulary | 40 % |
| **Semantic embeddings** (`all-MiniLM-L6-v2`) | Deep paraphrasing, synonym substitution | 35 % |

The frontend is built in **pure HTML + CSS + vanilla JavaScript** — no framework, no bundler, no build step.

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                         BROWSER                            │
│                                                            │
│  index.html ──▶ styles.css                                 │
│       │                                                    │
│  script.js  ──────────── fetch() ──────────────────────┐  │
└──────────────────────────────────────────────────────── │ ─┘
                                                          │
                         HTTP/JSON                        │
                                                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI  (backend/)                       │
│                                                             │
│  POST /api/check-plagiarism ──▶ routes.py                   │
│  POST /api/check-files      ──▶ routes.py                   │
│  GET  /api/health                                           │
│                                                             │
│  routes.py                                                  │
│    ├── utils/file_extractor.py   (TXT / PDF / DOCX → str)   │
│    └── core/                                                │
│          ├── preprocessor.py    (tokenise, shingle)         │
│          └── similarity.py      (Jaccard + TF-IDF + Sem)    │
└─────────────────────────────────────────────────────────────┘
```

**Data flow for a text comparison request:**

1. User pastes text → JS validates → `fetch POST /api/check-plagiarism`
2. FastAPI validates the JSON body (Pydantic)
3. `preprocessor.py` tokenises, lemmatises, removes stopwords, builds shingles
4. `similarity.py` runs three strategies in parallel and blends the scores
5. Sentence alignment finds matching sentence pairs
6. JSON response → JS renders score card, breakdown, match cards, highlights

---

## Tech Stack

**Backend**
- Python 3.10+
- FastAPI + Uvicorn
- NLTK (tokenisation, lemmatisation, stopwords)
- scikit-learn (TF-IDF, cosine similarity)
- sentence-transformers + `all-MiniLM-L6-v2` (semantic)
- pdfplumber (PDF extraction)
- python-docx (DOCX extraction)

**Frontend**
- HTML5, CSS3 (CSS custom properties, CSS Grid, Flexbox)
- Vanilla JavaScript (ES2022, Fetch API)
- Google Fonts: Syne · Lora · JetBrains Mono

**DevOps**
- Docker + Docker Compose
- pytest for unit & integration tests

---

## Project Structure

```
plagiarism-checker/
├── backend/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app factory
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py            # Endpoint definitions
│   ├── core/
│   │   ├── __init__.py
│   │   ├── preprocessor.py      # NLP preprocessing pipeline
│   │   └── similarity.py        # Multi-strategy similarity engine
│   └── utils/
│       ├── __init__.py
│       └── file_extractor.py    # TXT / PDF / DOCX → plain text
├── frontend/
│   ├── index.html
│   └── assets/
│       ├── css/styles.css
│       └── js/script.js
├── tests/
│   └── test_similarity.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Quick Start

### Prerequisites
- Python 3.10 or higher
- pip

### 1. Clone the repository

```bash
git clone https://github.com/your-username/plagiarism-checker.git
cd plagiarism-checker
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Without semantic similarity** (lighter install, ~200 MB less):  
> Comment out `sentence-transformers` and `torch` in `requirements.txt` before installing.

### 4. Run the server

```bash
uvicorn backend.main:app --reload --port 8000
```

### 5. Open the UI

Navigate to **http://localhost:8000** in your browser.  
API docs are at **http://localhost:8000/api/docs**.

---

## API Reference

### `GET /api/health`

Returns service status and whether the semantic model is loaded.

```json
{ "status": "ok", "semantic_similarity": true }
```

---

### `POST /api/check-plagiarism`

Compare two plain-text strings.

**Request body** (JSON):
```json
{
  "text_a": "Artificial intelligence is transforming...",
  "text_b": "The economy is being reshaped by AI..."
}
```

**Response**:
```json
{
  "similarity_percent": 61.3,
  "jaccard_score": 0.42,
  "tfidf_score": 0.68,
  "semantic_score": 0.74,
  "semantic_available": true,
  "matched_pairs": [
    {
      "sentence_a": "Artificial intelligence is transforming...",
      "sentence_b": "The economy is being reshaped by AI...",
      "score": 0.74,
      "method": "semantic"
    }
  ],
  "weights_used": { "jaccard": 0.25, "tfidf": 0.40, "semantic": 0.35 },
  "processing_time_ms": 312.4
}
```

---

### `POST /api/check-files`

Compare two uploaded files.

**Request**: `multipart/form-data` with fields `file_a` and `file_b`.  
**Accepted formats**: `.txt`, `.pdf`, `.docx` (max 5 MB each).  
**Response**: same schema as `/check-plagiarism`.

---

## How It Works

### 1 — Preprocessing (`preprocessor.py`)

```
Raw text
  → sentence split (NLTK punkt)
  → lowercase + punctuation strip
  → word tokenise
  → lemmatise (WordNetLemmatizer)
  → stopword removal
  → k-shingle (k-gram) set
```

### 2 — Jaccard Similarity

Measures the overlap between two sets of k-shingles (token 3-grams by default):

```
Jaccard(A, B) = |A ∩ B| / |A ∪ B|
```

Fast and effective for catching copy-paste and minor edits.

### 3 — TF-IDF Cosine Similarity

Converts each document to a TF-IDF vector (unigram + bigram) and measures the cosine of the angle between them:

```
cosine(A, B) = (A · B) / (‖A‖ · ‖B‖)
```

Catches paraphrasing with overlapping vocabulary.

### 4 — Semantic Similarity

Encodes each text as a dense sentence embedding using `all-MiniLM-L6-v2` (80 MB, runs on CPU):

```
score = cosine(embed(A), embed(B))
```

Captures meaning equivalence even when no words are shared.

### 5 — Sentence Alignment

For every sentence in Document A, finds the best matching sentence in Document B using TF-IDF cosine, with a semantic fallback for borderline pairs (score 0.30–0.55).

### 6 — Score Blending

```
final = 0.25 × Jaccard + 0.40 × TF-IDF + 0.35 × Semantic
```

Weights adjust automatically when the semantic model is unavailable.

---

## Sample Input / Output

**Document A:**
> Artificial intelligence is transforming every sector of the modern economy. Machine learning algorithms can now process vast amounts of data to identify patterns invisible to humans. Companies investing in AI early gain significant competitive advantages.

**Document B:**
> The modern economy is being reshaped by artificial intelligence. Advanced algorithms analyse enormous datasets, uncovering trends beyond human perception. Organisations adopting AI ahead of competitors stand to benefit greatly.

**Expected output:**
```
Similarity: ~62%  (Moderate Similarity)
Jaccard:    0.38
TF-IDF:     0.65
Semantic:   0.79
Matched pairs: 3 sentence pairs (method: semantic)
```

---

## Docker Deployment

```bash
# Build and run
cd docker
docker compose up --build

# Open http://localhost:8000
```

The container pre-downloads NLTK data and the sentence-transformer model during the build so cold-start is instant.

---

## Cloud Deployment (Render)

1. Push your repository to GitHub.
2. On [render.com](https://render.com), create a new **Web Service**.
3. Set **Build Command**: `pip install -r requirements.txt`
4. Set **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
5. Add environment variable: `CORS_ORIGINS=https://your-app.onrender.com`
6. Deploy — Render's free tier works for CPU-only inference.

> **Note**: First request after cold start may take ~30 s while the semantic model loads into memory. Consider the Starter plan for always-on hosting.

---

## Running Tests

```bash
# Install dev extras (if not already)
pip install pytest httpx

# Run all tests
pytest tests/ -v

# Run only unit tests (no server required)
pytest tests/ -v -k "not FastAPI"
```

---

## Configuration

All configuration is via environment variables:

| Variable | Default | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Bind port |
| `CORS_ORIGINS` | `http://localhost:8000,...` | Allowed CORS origins (comma-separated) |
| `LOG_LEVEL` | `info` | Uvicorn log level |

---

## Roadmap

- [ ] Multi-document comparison (upload 3+ files, detect pairwise similarity)
- [ ] Citation-aware detection (ignore properly quoted text)
- [ ] Database storage of comparison history
- [ ] Export report as PDF
- [ ] REST API key authentication
- [ ] Rate limiting middleware

---

## License

MIT © 2025 — see [LICENSE](LICENSE) for details.
