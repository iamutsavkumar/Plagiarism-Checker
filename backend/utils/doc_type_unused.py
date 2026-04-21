import cv2
import numpy as np
from PIL import Image


def is_handwritten(image: Image.Image) -> bool:
    """
    Improved heuristic to detect handwritten vs printed text.
    """

    # Convert to grayscale
    img = np.array(image.convert("L"))

    # 🔥 Blur to reduce noise
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    # 🔥 Edge detection
    edges = cv2.Canny(blur, 50, 150)
    edge_ratio = np.sum(edges > 0) / edges.size

    # 🔥 Variance (texture irregularity)
    variance = np.var(blur)

    # 🔥 Contour analysis (handwriting = irregular shapes)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_count = len(contours)

    # 🔥 Heuristic rules (balanced)
    handwritten_score = 0

    if edge_ratio > 0.07:
        handwritten_score += 1

    if variance < 800:
        handwritten_score += 1

    if contour_count > 500:
        handwritten_score += 1

    # require at least 2 signals
    return handwritten_score >= 2
