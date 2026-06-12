import easyocr
import numpy as np
from preprocessing import preprocess

reader = easyocr.Reader(["en"])

def detect_text(img: np.ndarray):
    results = reader.readtext(img)

    output = []
    for bbox, text, conf in results:
        if conf > 0.4:
            output.append({
                "bbox": [[int(x), int(y)] for x, y in bbox],
                "text": text,
                "conf": round(float(conf), 2)
            })
    return output