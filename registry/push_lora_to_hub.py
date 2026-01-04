from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from huggingface_hub import HfApi


def push_lora_model(
    base_model_name: str,
    lora_checkpoint_dir: str,
    repo_id: str
):
    """
    Push LoRA adapters + tokenizer to Hugging Face Hub.
    """

    # Load base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2
    )

    # Load LoRA adapters
    model = PeftModel.from_pretrained(
        base_model,
        lora_checkpoint_dir
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Push model + tokenizer
    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)

    print(f"✅ LoRA model successfully pushed to HF Hub: {repo_id}")


if __name__ == "__main__":
    push_lora_model(
        base_model_name="bert-base-uncased",
        lora_checkpoint_dir="outputs/lora_finetune",
        repo_id="Koppula-Sreeja-Reddy/imdb-bert-lora"
    )
