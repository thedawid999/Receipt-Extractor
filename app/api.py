from fastapi import FastAPI, UploadFile, File
from pipeline_processor import process_receipt

# for one file only
@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    return process_receipt(file)

# for multiple files
@app.post("/extract-batch")
async def extract_batch(files: List[UploadFile] = File(...)):
    return [process_receipt(file) for file in files]