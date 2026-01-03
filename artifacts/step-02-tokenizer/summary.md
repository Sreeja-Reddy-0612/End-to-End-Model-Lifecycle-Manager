## STEP-02: Tokenizer Compatibility Validation

**Model:** bert-base-uncased  
**Tokenizer:** AutoTokenizer  
**Context Window:** 512 tokens

### Objective
Verify whether the dataset text can be safely consumed by the chosen model without silent truncation or runtime errors.

### Observations
- Average token length: ~313 tokens
- Maximum token length: 3127 tokens
- Percentage of samples exceeding context window: ~14.8%

### Key Findings
- Majority of samples fit within the model’s context window.
- A non-trivial subset of reviews exceeds 512 tokens.
- Without explicit handling, truncation would occur silently during training.

### Decision Taken
- Do **not** change dataset or model.
- Introduce explicit truncation and padding strategy in preprocessing (STEP-03).
- Accept controlled information loss for long samples.

### Why This Step Matters
Transformer models have hard context limits. Validating token lengths:
- Prevents silent bugs
- Makes preprocessing decisions explicit
- Aligns data behavior with model constraints

This is a defensive ML engineering step commonly used in production systems.
