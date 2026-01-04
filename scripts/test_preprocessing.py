from hf_datasets.loader import load_hf_dataset
from training.preprocessing import build_preprocessing_fn

dataset = load_hf_dataset("imdb", split="train")

preprocess_fn = build_preprocessing_fn(
    model_name="bert-base-uncased"
)

tokenized_ds = dataset.map(
    preprocess_fn,
    batched=True,
    remove_columns=dataset.column_names
)

print(tokenized_ds)
print(tokenized_ds[0])

