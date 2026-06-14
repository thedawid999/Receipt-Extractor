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

# groups text fields with a height difference of 10 to one line string
def group_lines(results: list[dict[str, object]], threshold=10):
    # sort y-position ascending
    results = sorted(results, key=lambda x: x["bbox"][0][1])

    lines = []
    current_line = []
    last_y = None

    for item in results:
        y = item["bbox"][0][1]
        x = item["bbox"][0][0]

        if last_y is None or abs(y - last_y) <= threshold:
            # current_line here contains a dict of bbox and text
            current_line.append(item)
        else:
            # sort also x-position
            current_line = sorted(current_line, key=lambda i: i["bbox"][0][0])

            # add the the whole line
            lines.append(current_line)
            # add the first item of a new line
            current_line = [item]

        last_y = y

    # add the last line if it contains any words    
    if current_line:
        current_line = sorted(current_line, key=lambda i: i["bbox"][0][0])
        lines.append(current_line)
    
    return lines

