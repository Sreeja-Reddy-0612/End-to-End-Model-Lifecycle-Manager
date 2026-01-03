from transformers import AutoTokenizer
import numpy as np

def validate_tokenizer(dataset, model_name: str, text_field: str = "text"):
    """
    Validates tokenizer compatibility with dataset text lengths.

    Args:
        dataset: Hugging Face Dataset
        model_name (str): e.g. 'bert-base-uncased'
        text_field (str): column containing text
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    max_length = tokenizer.model_max_length
    token_lengths = []

    for example in dataset:
        tokens = tokenizer(
            example[text_field],
            truncation=False,
            add_special_tokens=True
        )
        token_lengths.append(len(tokens["input_ids"]))

    token_lengths = np.array(token_lengths)

    print(f"\n🔹 Model: {model_name}")
    print(f"🔹 Tokenizer max length: {max_length}")
    print(f"🔹 Avg token length: {int(token_lengths.mean())}")
    print(f"🔹 Max token length in dataset: {token_lengths.max()}")

    overflow_pct = (token_lengths > max_length).mean() * 100
    print(f"⚠️ Samples exceeding max length: {overflow_pct:.2f}%")

    if overflow_pct > 0:
        print("⚠️ WARNING: Truncation WILL occur during training.")
    else:
        print("✅ Safe: No truncation risk detected.")
