## Pipeline Validation Scripts

These scripts are used to validate individual stages of the model lifecycle:

- `test_dataset.py` — dataset ingestion & inspection
- `test_tokenizer.py` — tokenizer compatibility validation
- `test_preprocessing.py` — tokenizer-aware preprocessing
- `test_trainer.py` — full fine-tuning Trainer pipeline validation

They are intentionally separated from core modules to keep the system modular and clean.
