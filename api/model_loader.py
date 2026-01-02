from transformers import TrOCRProcessor, VisionEncoderDecoderModel

MODEL_PATH = "s3://ocr-bucket/models/approved/"

processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)
