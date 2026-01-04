from hf_datasets.loader import load_hf_dataset
from training.preprocessing import build_preprocessing_fn
from training.lora_engine import run_lora_finetuning

# Load datasets
train_ds = load_hf_dataset("imdb", split="train")
test_ds = load_hf_dataset("imdb", split="test")

# Preprocess
preprocess_fn = build_preprocessing_fn("bert-base-uncased")

train_tok = train_ds.map(
    preprocess_fn,
    batched=True,
    remove_columns=train_ds.column_names
)

test_tok = test_ds.map(
    preprocess_fn,
    batched=True,
    remove_columns=test_ds.column_names
)

# LoRA training
trainer, model = run_lora_finetuning(
    model_name="bert-base-uncased",
    tokenized_train_ds=train_tok,
    tokenized_eval_ds=test_tok
)
