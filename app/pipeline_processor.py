import os
from preprocessing import preprocess
from ocr import detect_text
from regex_extractor import extract_data
from utils import group_lines

# used only for local main.py
# check if path includes only one file or mutliple files
def process_input(path):
    # if one file only
    if os.path.isFile:
        return process_receipt(path)
    # if multiple files
    elif os.path.isdir(path):
        results = []
        # get a full path of each file
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            # check if created path is a file and if it's an image
            if os.file.isFile(file_path) and file.lower().endswith(".jpg", ".jpeg", ".png"):
                results.append(process_receipt(file_path))
        return results
    else:
        raise ValueError("Input must be a directory or a file of type .jpg, .jpeg or .png")


path = "samples/original/receipt8.jpg"
#def process_receipt(path: str):
    #return None

preprocessed = preprocess(path)
detected = detect_text(preprocessed)
lines = group_lines(detected)
for i in lines:
    print(f"{i}\n")
#extract_data(lines)

