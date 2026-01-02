import yaml
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Trainer,
    TrainingArguments
)
from etl.dataset_loader import load_dataset, load_config

def train():
    config = load_config()

    processor = TrOCRProcessor.from_pretrained(config["model"]["name"])
    model = VisionEncoderDecoderModel.from_pretrained(config["model"]["name"])

    train_ds = load_dataset(config["dataset"]["train_csv"], processor)
    val_ds = load_dataset(config["dataset"]["val_csv"], processor)

    args = TrainingArguments(
        output_dir="/tmp/model",
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"]["batch_size"],
        num_train_epochs=config["training"]["epochs"],
        learning_rate=config["training"]["learning_rate"],
        fp16=config["training"]["fp16"],
        evaluation_strategy="steps",
        save_steps=500,
        logging_steps=100,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=processor
    )

    trainer.train()
    trainer.save_model("/tmp/model")
    processor.save_pretrained("/tmp/model")

if __name__ == "__main__":
    train()
