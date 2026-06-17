from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import shutil

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

# standardizes the bbox coordinates between 0-1000 for LayoutLMv3
def standardize_bbox(bbox, height, width):
    x_min, y_min, x_max, y_max = bbox

    x_min_n = int((x_min / width) * 1000)
    y_min_n = int((y_min / height) * 1000)
    x_max_n = int((x_max / width) * 1000)
    y_max_n = int((y_max / height) * 1000)   

    # to make sure that the coordinates stay between 0-1000 (ocr can create false coordinates like -2 if near to the edge)
    return [
        min(max(x_min_n, 0), 1000),
        min(max(y_min_n, 0), 1000),
        min(max(x_max_n, 0), 1000),
        min(max(y_max_n, 0), 1000)
    ]
