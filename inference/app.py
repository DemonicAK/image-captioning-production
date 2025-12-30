
from fastapi import FastAPI,UploadFile, File,Form
from PIL import Image
import io
from inference.predict import  prediction_pipeline
import tensorflow as tf
print("TF version:", tf.__version__)
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
async def caption_image(
    file: UploadFile = File(...),
   algoname: str = Form("beam")
):
    image = await file.read()
    image = Image.open(io.BytesIO(image)).convert("RGB")
    caption = prediction_pipeline(image, algoname)
    return {"caption": caption, "algorithm": algoname}