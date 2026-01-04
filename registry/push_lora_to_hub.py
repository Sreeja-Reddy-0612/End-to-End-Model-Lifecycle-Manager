from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

REPO_ID = "Koppula-Sreeja-Reddy/imdb-bert-lora"
BASE_MODEL = "bert-base-uncased"
LOCAL_LORA_DIR = "outputs/lora_finetune"


def push_lora_model():
    print("🔹 Loading base model...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2
    )

    print("🔹 Loading LoRA adapters...")
    model = PeftModel.from_pretrained(
        base_model,
        LOCAL_LORA_DIR
    )

    print("🔹 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_LORA_DIR)

    print("🔹 Creating HF repo (if not exists)...")
    api = HfApi()
    api.create_repo(
        repo_id=REPO_ID,
        exist_ok=True,
        repo_type="model"
    )

    print("🔹 Pushing model adapters to Hub...")
    model.push_to_hub(REPO_ID)

    print("🔹 Pushing tokenizer to Hub...")
    tokenizer.push_to_hub(REPO_ID)

    print("✅ SUCCESS: LoRA adapters published to Hugging Face Hub")
    print(f"🔗 https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    push_lora_model()
