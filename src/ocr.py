import easyocr
import numpy as np
from .utils import convert_to_topleft_and_bottomright, standardize_bbox

reader = easyocr.Reader(["en"])

def detect_text(img: np.ndarray):
    results = reader.readtext(img)
    height, width = img.shape[:2]

    output = []
    for bbox, text, conf in results:
        if conf > 0.4:
            converted = convert_to_topleft_and_bottomright(bbox)
            standardized = standardize_bbox(converted, height, width)

            output.append(
                {
                    "bbox": standardized,
                    "text": text,
                }
            )
    return output

