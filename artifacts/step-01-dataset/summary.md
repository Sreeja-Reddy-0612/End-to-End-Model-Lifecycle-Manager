## STEP-01: Dataset Ingestion & Inspection

**Dataset:** IMDb  
**Split:** Train  
**Source:** Hugging Face Hub (`load_dataset`)

### Objective
Understand the dataset structure, schema, and label distribution before any tokenization or modeling.

### Observations
- Dataset type: Hugging Face `Dataset`
- Features:
  - `text`: string (movie review)
  - `label`: binary class (`neg`, `pos`)
- Sample text length: highly variable, including long-form reviews
- Label distribution:
  - Label 0 (negative): 12,500
  - Label 1 (positive): 12,500

### Key Findings
- Dataset is **perfectly balanced**, so no resampling is required.
- Text data is **raw and untruncated**, which is ideal for downstream tokenizer validation.
- The dataset is suitable for supervised text classification with transformer models.

### Decision Taken
Proceed with this dataset for the full lifecycle pipeline and validate tokenizer compatibility before training.

### Why This Step Matters
Most ML failures originate from misunderstood data. Explicit inspection ensures:
- Correct loss function choice
- Correct evaluation metrics
- No hidden data leakage or imbalance
