from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
from model import predict

app = FastAPI()

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    results = predict(image)

    return {"predictions": results}