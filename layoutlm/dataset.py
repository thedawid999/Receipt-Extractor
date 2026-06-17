import kagglehub
from pathlib import Path
import numpy as np
import pandas as pd
import json

base = kagglehub.dataset_download("urbikn/sroie-datasetv2")
base = Path(base)
sroie_path = base / "SROIE2019"

# reads box-file containing bbox coordinates and the word inside
# leaves only top-left and bottom-right bbox coordinates
# returns a DataFrame
def read_bbox(path: Path):
    bbox_list = []

    with open(path, "r") as f:
        for line in f.read().splitlines():
            if len(line) == 0:
                continue

            lines = line.split(",")

            bbox = np.array(lines[0:8], dtype = np.int32)
            text = ",".join(lines[8:])

            bbox_list.append([path.stem, *bbox, text])

        dataframe = pd.DataFrame(bbox_list, columns=['filename', 'x0', 'y0', 'x1', 'y1', 
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

ent_path = sroie_path / "test/entities/" / "X00016469670.txt"
entities = read_entities(ent_path)
print(entities)
