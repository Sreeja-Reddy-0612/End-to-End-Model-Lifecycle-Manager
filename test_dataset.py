from hf_datasets.loader import load_hf_dataset
from hf_datasets.inspector import inspect_dataset

dataset = load_hf_dataset("imdb", split="train")
inspect_dataset(dataset)
