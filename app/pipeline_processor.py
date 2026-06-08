import os

# used only for local main.py
# check if path includes only one file or mutliple files
def process_input(path):
    if os.path.isFile:
        return process_receipt(path)
    elif os.path.isdir(path):
        results = []

        for file in os.listdir(path)
            full_path = os.path.join(path, file)

            if os.file.isFile(full_path) AND file.lower().endswith(".jpg", ".jpeg", ".png"):
                results.append(process_receipt(full_path))
        return results
    else:
        raise ValueError("Input must be a directory or a file of type .jpg, .jpeg or .png")

def process_receipt(path: str):
    # preprocess
    # ocr 
    # regex
    # output