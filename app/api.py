from fastapi import FastAPI, UploadFile, File
from layoutlm import process_receipt
from utils import save_upload

# for one file only
@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    file_path = save_upload(file)
    return process_receipt(file_path)

# for multiple files
@app.post("/extract-batch")
async def extract_batch(files: List[UploadFile] = File(...)):
    results = []

    for file in files:
        file_path = save_upload(file)
        results.append(process_receipt(file_path))
    
    return results