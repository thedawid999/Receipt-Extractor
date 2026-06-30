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

# used only for local main.py
# check if path includes only one file or mutliple files
def process_input(path):
    # if one file only
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            return path
        else:
            raise ValueError("File must be of type .jpg, .jpeg or .png")
    # if multiple files
    elif path.is_dir():
        results = []
        # get a full path of each file
        for file_path in path.iterdir():
            # check if created path is a file and if it's an image
            if file_path.is_file() and file_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                results.append(file_path)
        return results
    else:
        raise ValueError("Input must be a directory or a file of type .jpg, .jpeg or .png")

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

user_input = process_input("./samples/X51005301659.jpg")
results = predict(user_input)
save_to_file("outputs", results)







