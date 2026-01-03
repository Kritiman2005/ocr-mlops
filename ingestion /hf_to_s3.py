import os
import s3fs
import pandas as pd
from datasets import load_dataset
import yaml

# ----------------------------
# 1️⃣ Load config
# ----------------------------
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config/config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

HF_DATASET_NAME = config["dataset"]["huggingface_name"]
HF_SPLIT = config["dataset"].get("split", "train")  # default to 'train'
S3_BUCKET = config["aws"]["s3_bucket"]
S3_PATH = config["aws"].get("s3_path", "datasets")

# ----------------------------
# 2️⃣ Initialize S3 filesystem
# ----------------------------
fs = s3fs.S3FileSystem(key=config["aws"]["access_key"], secret=config["aws"]["secret_key"])

# ----------------------------
# 3️⃣ Load HuggingFace dataset
# ----------------------------
print(f"Loading HuggingFace dataset: {HF_DATASET_NAME} ({HF_SPLIT})...")
dataset = load_dataset(HF_DATASET_NAME, split=HF_SPLIT)

# ----------------------------
# 4️⃣ Convert to Pandas DataFrame
# ----------------------------
print("Converting dataset to DataFrame...")
df = dataset.to_pandas()

# ----------------------------
# 5️⃣ Save CSV locally
# ----------------------------
local_csv_path = os.path.join(os.path.dirname(__file__), f"{HF_DATASET_NAME.replace('/', '_')}.csv")
df.to_csv(local_csv_path, index=False)
print(f"Dataset saved locally at: {local_csv_path}")

# ----------------------------
# 6️⃣ Upload to S3
# ----------------------------
s3_file_path = f"s3://{S3_BUCKET}/{S3_PATH}/{os.path.basename(local_csv_path)}"
print(f"Uploading dataset to S3: {s3_file_path}")
fs.put(local_csv_path, s3_file_path)
print("Upload complete ✅")
