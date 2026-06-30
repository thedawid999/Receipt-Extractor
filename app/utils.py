from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import shutil
import re

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
            