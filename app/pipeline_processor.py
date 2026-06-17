from preprocessing import preprocess
from pathlib import Path


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


#path = "samples/original/receipt8.jpg"
#def process_receipt(path: str):
    #return None

#preprocessed = preprocess(path)
#detected = detect_text(preprocessed)
#lines = group_lines(detected)
#for i in lines:
#    print(f"{i}\n")
#extract_data(lines)

