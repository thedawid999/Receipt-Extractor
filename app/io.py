from pathlib import Path
import json

# check if path includes only one file or mutliple files
# ALWAYS returns a list
def resolve_image_paths(path):
    path = Path(path)

    # if one file only
    if path.is_file():
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            return [path]
        raise ValueError("Unsupported file type")

    # if multiple files
    if path.is_dir():
        return [
            f for f in path.iterdir()
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]

    raise ValueError("Input must be file or directory")

# saves prediction to json
def save_to_file(filename: str, results):
    with open(f"{filename}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)