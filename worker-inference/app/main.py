from fastapi import FastAPI, UploadFile, File
from app.pipeline import PhotoRestorationPipeline
import shutil
import os

app = FastAPI(title="Restoration Inference Worker")
pipeline = PhotoRestorationPipeline()

@app.post("/restore")
async def restore_photo(file: UploadFile = File(...)):
    os.makedirs("/tmp/restoration", exist_ok=True)
    input_path = f"/tmp/restoration/in_{file.filename}"
    output_path = f"/tmp/restoration/out_{file.filename}"

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    pipeline.process(input_path, output_path)

    return {"status": "success", "result_path": output_path}