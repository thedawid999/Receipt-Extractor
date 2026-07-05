from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import shutil
import re
import json

# saves upload image through API to get a PATH as a input for process_receipt()
def save_upload(file: UploadFile, folder="uploads"):
    path = Path(folder)
    # create that folder if it does not exist
    path.mkdir(exist_ok=True)

    # concat folder-path and file to a full path
    file_path = path / file.filename

    # save the file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return str(file_path)

# converts ocr-bbox-format [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] to annotation-format [x_min, y_min, x_max, y_max]
def convert_to_topleft_and_bottomright(bbox: list[list[float, float]]):
    x_coords = [i[0] for i in bbox]
    y_coords = [i[1] for i in bbox]

    x_min = int(min(x_coords))
    y_min = int(min(y_coords))
    x_max = int(max(x_coords))
    y_max = int(max(y_coords))

    return [x_min, y_min, x_max, y_max]

# combines same entities into one (e.g. B-COMPANY and I-COMPANY)
def merge_bio_tags(words, labels):
    entities = {}
    current = None

    for word, label in zip(words, labels):

        if label == "O":
            current = None
            continue

        prefix, entity = label.split("-", 1)

        if prefix == "B":
            entities.setdefault(entity, []).append(word)
            current = entity

        elif prefix == "I" and current == entity:
            entities[entity][-1] += " " + word

    return entities

# extract only DATE, because sometimes the model classifies "22/12/2017 14.03" as DATE
def extract_date(text):
    date_pattern = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b")
    # return nothing if DATE not found
    if not text:
        return None

    match = date_pattern.search(text)

    if match:
        return match.group()

    return None
            
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

# saves prediction to json
def save_to_file(filename: str, results):
    with open(f"{filename}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

# loads config file for scheduler
def load_config():
    with open("src/config.json", "r", encoding="utf-8") as f:
        return json.load(f)