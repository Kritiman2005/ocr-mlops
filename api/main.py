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

    # Confidence estimation (mean token probability)
    scores = torch.stack(outputs.scores)
    probs = torch.softmax(scores, dim=-1)
    max_probs = probs.max(dim=-1).values
    confidence = float(max_probs.mean().item())

    return {
        "prediction": text,
        "confidence": confidence
    }
