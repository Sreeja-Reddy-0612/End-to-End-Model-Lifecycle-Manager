## STEP-04: Full Fine-Tuning (Trainer Baseline)

### Objective
Establish a full fine-tuning baseline using Hugging Face Trainer.

### Outcome
- Trainer pipeline verified end-to-end
- Model, tokenizer, preprocessing, and metrics integrated correctly
- Training initiated successfully

### Observation
- Full BERT fine-tuning on CPU is prohibitively slow
- Expected behavior due to lack of GPU acceleration

### Decision Taken
- Stop full fine-tuning early after pipeline validation
- Use this step as a reference baseline
- Proceed to LoRA / PEFT fine-tuning for practical training

### Why This Is Correct
In production systems, full fine-tuning is GPU-backed.
Local environments are used to validate correctness, not to exhaustively train.
