## STEP-05: LoRA / PEFT Fine-Tuning

### Objective
Validate parameter-efficient fine-tuning using LoRA adapters.

### Configuration
- Base model: bert-base-uncased
- Trainable parameters: ~296K
- Total parameters: ~109M
- Trainable percentage: ~0.27%
- Epochs configured: 3

### Observations
- LoRA adapters successfully injected
- Base model weights frozen
- Training loop started correctly
- Training is significantly lighter than full fine-tuning but still slow on CPU

### Decision Taken
Stop training early after validating correctness.
Use this setup for GPU-backed training in production.

### Why This Matters
This reflects real-world ML workflows where PEFT is preferred for efficiency,
and local environments are used to validate pipelines rather than complete training.
