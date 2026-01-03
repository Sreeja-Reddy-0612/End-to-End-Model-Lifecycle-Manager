from transformers import AutoTokenizer

def build_preprocessing_fn(
    model_name: str,
    text_field: str = "text",
    label_field: str = "label",
    max_length: int = 512
):
    """
    Builds a tokenizer-aware preprocessing function.

    Returns a function compatible with Hugging Face Dataset.map()
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess(batch):
        tokenized = tokenizer(
            batch[text_field],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

        if label_field in batch:
            tokenized["labels"] = batch[label_field]

        return tokenized

    return preprocess
