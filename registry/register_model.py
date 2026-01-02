import shutil
import yaml

def register_model():
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    shutil.copytree(
        "/tmp/model",
        config["registry"]["approved_model_path"],
        dirs_exist_ok=True
    )

if __name__ == "__main__":
    register_model()
