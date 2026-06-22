import kagglehub
from pathlib import Path
import numpy as np
import pandas as pd
import json

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 2000)

base = kagglehub.dataset_download("urbikn/sroie-datasetv2")
base = Path(base)
sroie_path = base / "SROIE2019"

# reads box-file containing bbox coordinates and the word inside
# leaves only top-left and bottom-right bbox coordinates
# returns a DataFrame
def read_bboxes(path: Path):
    bboxes_list = []

    with open(path, "r") as f:
        for line in f.read().splitlines():
            if len(line) == 0:
                continue

            lines = line.split(",")

            bbox = np.array(lines[0:8], dtype = np.int32)
            text = ",".join(lines[8:])

            bboxes_list.append([path.stem, *bbox, text])

        dataframe = pd.DataFrame(bboxes_list, columns=['filename', 'x0', 'y0', 'x1', 'y1', 
        'x2', 'y2', 'x3', 'y3', 'text'])
        dataframe = dataframe.drop(columns=['x1', 'y1', 'x3', 'y3'])

    return dataframe

# reads entities
# return a DataFrame
def read_entities(path: Path):
    with open(path, "r") as f:
        data = json.load(f)

    dataframe = pd.DataFrame([data])
    return dataframe

# combines bbox-file and entities-file into one DataFrame
def assign_labels(bboxes: pd.DataFrame, entities: pd.DataFrame):
    bboxes = bboxes.copy()
    # define the length of labels and set all label to "O" initially
    labels = ["O"] * len(bboxes)

    row = entities.iloc[0]
    company = row["company"]
    address = row["address"]
    date = row["date"]
    total = row["total"]

    total_found = False

    for i, row in bboxes.iterrows():
        line = row["text"]
        y0 = row["y0"]

        # 1. safe-check if ENTITY exists in entities
        # 2. real-check if line is a subset of ENTITY

        # line length at least 3, because according to this logic individual words like "MARKET" that appear in the COMPANY, have been classified as COMPANY which is wrong
        # "line in company" and not "company in line" because company is sometimes separeted into multiple lines
        if company and (line in company) and len(line) > 2:
            labels[i] = "COMPANY"
        # line length at least 3, because according to this logic individual numbers like "1" that appear in the ADDRESS, have been classified as ADDRESS which is wrong
        # "line in address" and not "address in line" because address is mostly separeted into multiple lines
        if address and (line in address) and len(line) > 2: 
            labels[i] = "ADDRESS"
        if date and (date in line):
            labels[i] = "DATE"
        if total and (total in line) and total_found == False:
            # total value might occur more than only once
            # always save the total value which has the highest y-position (so it is always the first one)
            labels[i] = "TOTAL"
            total_found = True

    bboxes["label"] = labels
    return bboxes


# JUST FOR TEST
bboxes_path = sroie_path / "train/box/" / "X51005663297.txt"
bboxes = read_bboxes(bboxes_path)
ent_path = sroie_path / "train/entities/" / "X51005663297.txt"
entities = read_entities(ent_path)

dataset = assign_labels(bboxes, entities)
print(dataset)

