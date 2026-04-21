import io
import re
import os

import pytesseract
from PIL import Image
import numpy as np
import cv2
from .ocr_handwriting import extract_handwritten_text

# ✅ SAFE: Use environment variable with fallback (NO BREAK)
pytesseract.pytesseract.tesseract_cmd = os.getenv(
    "TESSERACT_PATH",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# -----------------------------------------------------------
# 🔧 IMAGE PREPROCESSING (IMPROVED)
# -----------------------------------------------------------
def preprocess_image(image: Image.Image) -> Image.Image:
    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 🔥 sharpen
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, kernel)

    # 🔥 denoise
    gray = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)

    # 🔥 threshold (clean binary image)
    thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    return Image.fromarray(thresh)


# -----------------------------------------------------------
# 🔍 OCR VALIDATION (STRONG)
# -----------------------------------------------------------
def _is_valid_ocr(text: str) -> bool:
    words = text.split()

    if len(words) < 10:
        return False

    meaningful = [w for w in words if len(w) >= 3]
    if len(meaningful) < len(words) * 0.7:
        return False

    vowel_words = [w for w in words if re.search(r'[aeiou]', w)]
    if len(vowel_words) < len(words) * 0.7:
        return False

    english_like = [w for w in words if re.search(r'[aeiou]', w) and len(w) >= 3]
    if len(english_like) < len(words) * 0.6:
        return False

    weird = [w for w in words if re.search(r'(.)\1\1', w)]
    if len(weird) > len(words) * 0.1:
        return False

    avg_len = sum(len(w) for w in words) / len(words)
    if avg_len < 3:
        return False

    if len(set(words)) / len(words) < 0.6:
        return False

    return True


# -----------------------------------------------------------
# 🔧 OCR CLEANING (STRONG)
# -----------------------------------------------------------
def clean_ocr_text(text: str) -> str:
    text = text.strip().lower()

    # keep structure, remove noise
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', ' ', text)

    # remove single-letter junk
    text = re.sub(r'\b(?!a\b|i\b)[a-z]\b', '', text)

    # normalize spacing
    text = re.sub(r'\s+', ' ', text)

    words = text.split()

    cleaned_words = [
        w for w in words
        if len(w) >= 3 and re.search(r'[aeiou]', w)
    ]

    if len(cleaned_words) < max(5, len(words) * 0.5):
        return ""

    return " ".join(cleaned_words)


# -----------------------------------------------------------
# 📄 MAIN TEXT EXTRACTOR
# -----------------------------------------------------------
def extract_text(file_bytes: bytes, filename: str) -> str:
    filename = filename.lower()

    # ---------------- TXT ----------------
    if filename.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")

    # ---------------- PDF ----------------
    elif filename.endswith(".pdf"):
        import pdfplumber

        text = ""

        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()

                # fallback to OCR if needed
                if not extracted or len(extracted.strip()) < 20:
                    img = page.to_image(resolution=300).original
                    img = Image.fromarray(np.array(img))

                    print("🔥 USING GOOGLE OCR (PDF)")

                    extracted = extract_handwritten_text(
                        cv2.imencode('.png', np.array(img))[1].tobytes()
                    )

                    if not extracted.strip():
                        print("⚠️ Google OCR failed → fallback to Tesseract")
                        img = preprocess_image(img)
                        extracted = pytesseract.image_to_string(
                            img,
                            config="--oem 3 --psm 4 -l eng"
                        )

                text += (extracted or "") + "\n"

        print("🔎 RAW OCR:", text[:200])

        text = clean_ocr_text(text)

        print("✅ CLEAN OCR:", text[:200])

        if not _is_valid_ocr(text):
            return ""

        return text

    # ---------------- DOCX ----------------
    elif filename.endswith(".docx"):
        from docx import Document

        doc = Document(io.BytesIO(file_bytes))
        text = "\n".join([p.text for p in doc.paragraphs])

        return text.strip()

    # ---------------- IMAGE ----------------
    elif filename.endswith((".png", ".jpg", ".jpeg")):
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

        print("🔥 USING GOOGLE OCR (IMAGE)")

        text = extract_handwritten_text(file_bytes)

        if not text.strip():
            print("⚠️ Google OCR failed → fallback to Tesseract")
            image = preprocess_image(image)
            text = pytesseract.image_to_string(
                image,
                config="--oem 3 --psm 4 -l eng"
            )

        print("🔎 RAW OCR:", text[:200])

        text = clean_ocr_text(text)

        print("✅ CLEAN OCR:", text[:200])

        if not _is_valid_ocr(text):
            return ""

        return text

    # ---------------- UNSUPPORTED ----------------
    else:
        raise ValueError("Unsupported file type")