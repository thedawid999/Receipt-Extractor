from preprocessing import preprocess
from ocr import detect_text
from regex_extractor import extract_data
from utils import to_json_labelstudio_format
from pathlib import Path
from PIL import Image
import numpy as np

# used only for local main.py
# check if path includes only one file or mutliple files
def process_input(path):
    # if one file only
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            return process_receipt(path)
        else:
            raise ValueError("File must be of type .jpg, .jpeg or .png")
    # if multiple files
    elif path.is_dir():
        results = []
        # get a full path of each file
        for file_path in path.iterdir():
            # check if created path is a file and if it's an image
            if file_path.is_file() and file_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                results.append(process_receipt(file_path))
        return results
    else:
        raise ValueError("Input must be a directory or a file of type .jpg, .jpeg or .png")


# reads a whole folder of images and creates annotations
def create_annotations_for_labelstudio(folder_path, output_json_path):
    folder = Path(folder_path)

    for img_path in sorted(folder.glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        
        print(f"[{img_path}] PROCESSING...")
        
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        detections = detect_text(img_np)

        to_json_labelstudio_format(img_path, output_json_path, detections)
        print(f"[{img_path}] DONE...")

create_annotations_for_labelstudio("layoutlm-model/train/images", "layoutlm-model/train/labels")

#path = "samples/original/receipt8.jpg"
#def process_receipt(path: str):
    #return None

#preprocessed = preprocess(path)
#detected = detect_text(preprocessed)
#lines = group_lines(detected)
#for i in lines:
#    print(f"{i}\n")
#extract_data(lines)

