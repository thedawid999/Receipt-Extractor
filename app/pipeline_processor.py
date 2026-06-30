from preprocessing import preprocess
from ocr import detect_text
from pathlib import Path
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
from PIL import Image
import torch
import numpy as np
from utils import merge_bio_tags

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
def predict(paths: list[Path]):
    for path in paths:
        img = Image.open(path)
        preprocessed = preprocess(path)
        ocr_results = detect_text(preprocessed)

        text = [item['text'] for item in ocr_results]
        boxes = [item['bbox'] for item in ocr_results]

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

        # converting label_id into a string and merging BIO tags
        words = []
        labels = []
        print(f"============== Predictions for image: {path} ==============")
        for word_idx, label_id in word_predictions.items():
            words.append(text[word_idx])
            labels.append(model.config.id2label[label_id])

        entities = merge_bio_tags(words, labels)
        return entities



user_input = process_input("./samples/")
predict(user_input)





