FROM python:3.10-slim

# Install system dependencies (OCR + OpenCV support)
RUN apt-get update --fix-missing && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project
COPY . .

# Upgrade pip (prevents install issues)
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Force install OpenCV (fixes cv2 issue 100%)
RUN pip install --no-cache-dir opencv-python-headless

# Fix NLTK download (avoid crash)
RUN python -m nltk.downloader punkt stopwords wordnet || true

# Start FastAPI app
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]