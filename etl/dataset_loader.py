import yaml
import s3fs
import pandas as pd
from PIL import Image
from datasets import Dataset
from transformers import TrOCRProcessor

fs = s3fs.S3FileSystem()

def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)

def load_dataset(csv_path, processor):
    df = pd.read_csv(csv_path)

    def preprocess(example):
        with fs.open(example["image_path"], "rb") as f:
            image = Image.open(f).convert("RGB")

        pixel_values = processor(image, return_tensors="pt").pixel_values[0]

        labels = processor.tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        ).input_ids

        return {
            "pixel_values": pixel_values,
            "labels": labels
        }

    return Dataset.from_pandas(df).map(preprocess, remove_columns=df.columns)
