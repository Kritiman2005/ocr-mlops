from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import torch
import io
from PIL import Image

app = FastAPI()

# Mount frontend
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ocr")
async def run_ocr(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    pixel_values = processor(image, return_tensors="pt").pixel_values

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            output_scores=True,
            return_dict_in_generate=True
        )

    text = processor.batch_decode(
        outputs.sequences,
        skip_special_tokens=True
    )[0]

    scores = torch.stack(outputs.scores)
    probs = torch.softmax(scores, dim=-1)
    max_probs = probs.max(dim=-1).values
    confidence = float(max_probs.mean().item())

    return {
        "prediction": text,
        "confidence": confidence
    }
