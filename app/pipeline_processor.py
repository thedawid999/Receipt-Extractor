from preprocessing import preprocess
from ocr import detect_text
from pathlib import Path
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
from PIL import Image
import torch
import numpy as np
from utils import merge_bio_tags, extract_date
import json

model_path = "./layoutlm/layoutlmv3-final-v1/"
model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
processor = LayoutLMv3Processor.from_pretrained(model_path)

# check if path includes only one file or mutliple files
# ALWAYS returns a list
def resolve_image_paths(path):
    path = Path(path)

    # if one file only
    if path.is_file():
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            return [path]
        raise ValueError("Unsupported file type")

    # if multiple files
    if path.is_dir():
        return [
            f for f in path.iterdir()
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]

    raise ValueError("Input must be file or directory")

# preprocess
# OCR
# LayoutLM predicition
def predict(path):
    img = Image.open(path) # loading the image
    preprocessed = preprocess(path) # preprocessing the image
    ocr_results = detect_text(preprocessed) # finding text

    text = [item['text'] for item in ocr_results]
    boxes = [item['bbox'] for item in ocr_results]

    # setting up the processor
    encoding = processor(
        images=img,
        text=text,
        boxes=boxes,
        return_tensors="pt",
        truncation=True,
        padding="max_length"
    )

    # save ids to put tokens with the same id into one word (e.g. "ĠSD" and "N" --> "SDN")
    word_ids = encoding.word_ids(batch_index=0)

    # make the prediction
    with torch.no_grad():
        outputs = model(**encoding)

    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    word_predictions = {}

    # assign prediction for each word (using word_ids here to put tokens with same label into one word)
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue

        if word_idx not in word_predictions:
            word_predictions[word_idx] = predictions[token_idx]

    # converting label_id into a string
    words = []
    labels = []
    print(f"[.] Prediction for image: {path}")
    for word_idx, label_id in word_predictions.items():
        words.append(text[word_idx])
        labels.append(model.config.id2label[label_id])

    # merging same bio tags
    entities = merge_bio_tags(words, labels)
        
    # extracting only DATE in following formats dd.mm.yyy OR dd/mm/yyyy
    if "DATE" in entities:
        entities["DATE"] = [extract_date(x) for x in entities["DATE"]]

    print("[✔] Prediction completed")

    return {
        "image": str(path),
        "entities": entities
    }

def save_to_file(filename: str, results):
    with open(f"{filename}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

user_input = resolve_image_paths("./samples/")
results = []
for path in user_input:
    results.append(predict(path))

save_to_file("outputs", results)







