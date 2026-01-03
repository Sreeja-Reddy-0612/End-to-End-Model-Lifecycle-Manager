## STEP-03: Tokenizer-Aware Preprocessing Pipeline

**Model:** bert-base-uncased  
**Max Sequence Length:** 512  
**Padding Strategy:** Fixed (`max_length`)  
**Truncation:** Enabled

### Objective
Convert raw text into model-ready tensors using a controlled, reproducible preprocessing pipeline.

### Preprocessing Decisions
- `truncation=True` to handle overflow safely
- `padding="max_length"` for stable batch shapes
- `max_length=512` to match model context window
- Labels renamed from `label` → `labels` for Trainer compatibility

### Output Features
- `input_ids`
- `attention_mask`
- `token_type_ids`
- `labels`

### Key Findings
- All samples now have uniform sequence length.
- Tokenizer warnings are eliminated by design.
- Dataset is fully compatible with Hugging Face Trainer and PEFT workflows.

### Decision Taken
Proceed with this preprocessing pipeline for:
- Full fine-tuning (Trainer)
- Parameter-efficient fine-tuning (LoRA / PEFT)

### Why This Step Matters
This step transforms validated risks into controlled behavior.
It ensures:
- Reproducibility
- Stable training
- Clear separation between data validation and model training
