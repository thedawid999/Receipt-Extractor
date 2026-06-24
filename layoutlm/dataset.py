import kagglehub
from pathlib import Path
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from PIL import Image
from datasets import Dataset, DatasetDict

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 2000)

base = kagglehub.dataset_download("urbikn/sroie-datasetv2")
base = Path(base)
sroie_path = base / "SROIE2019"
sroie_path_train = sroie_path / "train"
sroie_path_test = sroie_path / "test"

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
    # set to "" if does not exist in the entities.txt
    company = row.get("company", "")
    address = row.get("address", "")
    date = row.get("date", "")
    total = row.get("total", "")

    y_max = bboxes["y0"].iloc[-1]
    total_found = False

    for i, row in bboxes.iterrows():
        line = row["text"]
        y0 = row["y0"]

        # 1. safe-check if ENTITY exists in entities
        # 2. real-check if line is a subset of ENTITY

        # company must be in the upper 30% of the document
        # "line in company" and not "company in line" because company is sometimes separeted into multiple lines
        if company and (line in company) and (y0 < y_max*0.3):
            labels[i] = "COMPANY"
        # address mus be in the upper 30% of the document
        # "line in address" and not "address in line" because address is mostly separeted into multiple lines
        if address and (line in address) and (y0 < y_max*0.3): 
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

# splits detected text in individual words (important for LayoutLM training)
def split_words(dataframe: pd.DataFrame):
    word_rows = []

    for _, row in dataframe.iterrows():
        text = str(row["text"])
        words = text.split()

        if not words:
            continue

        x0 = int(row["x0"])
        y0 = int(row["y0"])
        x2 = int(row["x2"])
        y2 = int(row["y2"])

        bbox_width = x2 - x0
        total_chars = sum(len(word) for word in words)

        word_x0 = x0

        for i, word in enumerate(words):
            word_width = round(bbox_width * (len(word) / total_chars))

            # if its the last word, set its x2-coordinates to bbox-x2-coordinates
            if i == len(words) - 1:
                word_x2 = x2
            else:
                word_x2 = word_x0 + word_width

            word_rows.append({
                "filename": row["filename"],
                "x0": word_x0,
                "y0": y0,
                "x2": word_x2,
                "y2": y2,
                "text": word,
                "label": row["label"]
            })

            word_x0 = word_x2

    return pd.DataFrame(word_rows)

# adds BIO-tag to previously assigned labels
def add_bio_tags(df: pd.DataFrame):
    bio_labels = []
    prev_label = "O"

    for i, row in df.iterrows():
        current_label = row["label"]

        if current_label == "O":
            bio_labels.append("O")
        elif current_label != prev_label:
            bio_labels.append(f"B-{current_label}")
        else:
            bio_labels.append(f"I-{current_label}")
        
        prev_label = current_label

    df["label"] = bio_labels
    return df

# combines all functions above and creates a dataset that is formatted for LayoutLM training
def create_dataset(folder: Path):
    bbox_folder = folder / "box"
    entities_folder = folder / "entities"
    img_folder = folder / "img"

    # label mapping (see labels.txt)
    labels = ['O', 'B-COMPANY', 'I-COMPANY', 'B-DATE', 'I-DATE', 
              'B-ADDRESS', 'I-ADDRESS', 'B-TOTAL', 'I-TOTAL']
    label2id = {label: i for i, label in enumerate(labels)}

    bbox_files = sorted(bbox_folder.glob("*.txt"))
    entity_files = sorted(entities_folder.glob("*.txt"))
    img_files = sorted(img_folder.glob("*.jpg"))

    data = []

    for bbox_file, entity_file, img_file in tqdm(zip(bbox_files, entity_files, img_files)):
        # load data
        bbox = read_bboxes(bbox_file)
        entities = read_entities(entity_file)
        image = Image.open(img_file)
        width, height = image.size

        # process data
        labeled = assign_labels(bbox, entities)
        splitted = split_words(labeled)
        tagged = add_bio_tags(splitted)

        # normalize bboxes (0-1000)
        def normalize_bbox(row):
            return [
                int(1000 * row['x0'] / width),
                int(1000 * row['y0'] / height),
                int(1000 * row['x2'] / width),
                int(1000 * row['y2'] / height)
            ]
        normalized_bboxes = [normalize_bbox(row) for _, row in tagged.iterrows()]
        
        # map labels with integers
        ner_tags = [label2id.get(label, 0) for label in tagged['label']]

        data.append({
            "id": bbox_file.stem,
            "tokens": tagged["text"].tolist(),
            "bboxes": normalized_bboxes,
            "ner_tags": ner_tags,
            "image": str(img_file)
        })

    return data

# runs create_dataset for training and test data, and saves it to the disk
def save_dataset():
    train_ds = create_dataset(sroie_path_train)
    test_ds = create_dataset(sroie_path_test)

    # convert to hugging face Dataset
    train_ds = Dataset.from_list(train_ds)
    test_ds = Dataset.from_list(test_ds)

    # merge both Datasets into one DatasetDict
    dataset_dict = DatasetDict({
        "train": train_ds,
        "test": test_ds
    })

    dataset_dict.save_to_disk("layoutlm/dataset")

save_dataset()

