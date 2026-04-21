import io
import re
import os

# 🔥 SAFE credentials loading (env + fallback)
try:
    from google.cloud import vision
    from google.oauth2 import service_account

    # ✅ Use environment variable, fallback to your current working path
    CREDENTIALS_PATH = os.getenv(
        "GOOGLE_CREDENTIALS_PATH",
        r"C:\PROGRAMMING\Projects\Plagiarism checker\key.json"
    )

    credentials = service_account.Credentials.from_service_account_file(
        CREDENTIALS_PATH
    )

    client = vision.ImageAnnotatorClient(credentials=credentials)
    _VISION_AVAILABLE = True

    print("✅ Google Vision initialized successfully")

except Exception as e:
    print("❌ Google Vision import/init failed:", e)
    client = None
    _VISION_AVAILABLE = False


# -----------------------------------------------------------
# 🔍 BASIC OCR QUALITY CHECK
# -----------------------------------------------------------
def _is_valid_google_ocr(text: str) -> bool:
    words = text.split()

    if len(words) < 10:
        return False

    meaningful = [w for w in words if len(w) >= 3]
    if len(meaningful) < len(words) * 0.6:
        return False

    vowel_words = [w for w in words if re.search(r'[aeiou]', w)]
    if len(vowel_words) < len(words) * 0.6:
        return False

    weird = [w for w in words if re.search(r'(.)\1\1', w)]
    if len(weird) > len(words) * 0.1:
        return False

    return True


# -----------------------------------------------------------
# 📄 MAIN FUNCTION
# -----------------------------------------------------------
def extract_handwritten_text(image_bytes: bytes) -> str:
    if not _VISION_AVAILABLE or client is None:
        print("❌ Google Vision NOT available → fallback to Tesseract")
        return ""

    try:
        print("🔥 USING GOOGLE VISION OCR")

        image = vision.Image(content=image_bytes)
        response = client.document_text_detection(image=image)

        if response.error.message:
            print("❌ GOOGLE API ERROR:", response.error.message)
            return ""

        texts = response.text_annotations

        if not texts:
            print("⚠️ No text detected by Google OCR")
            return ""

        extracted_text = texts[0].description.strip()

        print("🔎 RAW GOOGLE OCR:", extracted_text[:200])

        extracted_text = re.sub(r'\s+', ' ', extracted_text)

        if not _is_valid_google_ocr(extracted_text):
            print("⚠️ Google OCR output too weak → rejected")
            return ""

        print("✅ CLEAN GOOGLE OCR:", extracted_text[:200])

        return extracted_text

    except Exception as e:
        print("❌ GOOGLE OCR EXCEPTION:", e)
        return ""