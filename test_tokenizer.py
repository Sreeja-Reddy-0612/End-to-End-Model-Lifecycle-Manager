from hf_datasets.loader import load_hf_dataset
from utils.tokenizer_validator import validate_tokenizer

dataset = load_hf_dataset("imdb", split="train")

validate_tokenizer(
    dataset=dataset,
    model_name="bert-base-uncased"
)
