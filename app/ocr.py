import easyocr
import numpy as np
from preprocessing import preprocess

img= preprocess("samples/original/receipt0.jpg")
reader = easyocr.Reader(["en"])

results = reader.readtext(img)
text = " ".join([res[1] for res in results])
print(text)

#def detect_text(image: np.ndarray):
    