# End-to-End Model Lifecycle Manager

Enterprise-style, lifecycle-focused NLP system designed to **train, manage, register, and distribute models** using modern **parameter-efficient fine-tuning (LoRA / PEFT)** and the **Hugging Face ecosystem**.

This project focuses on **model lifecycle engineering**, not just model training — covering how models move from:

**experiments → artifacts → reusable assets**  
in real-world ML workflows.

---

## Problem Statement

Most machine learning projects stop at:

> “Train a model and save it locally.”

In real-world AI systems, this approach fails because:

- Training is not reproducible  
- Models are tightly coupled to experiments  
- Artifacts are not versioned  
- Fine-tuning is computationally expensive  
- There is no clear model registry  
- Teams cannot easily reuse or deploy models  

These issues make ML systems **fragile, unscalable, and production-unready**.

---

## Solution Overview

This project implements an **end-to-end model lifecycle pipeline** that:

- fine-tunes transformer models efficiently using **LoRA / PEFT**
- persists training artifacts safely, even on interruption
- separates training, preprocessing, registry, and scripts cleanly
- registers models on the **Hugging Face Hub** as reusable assets
- enables downstream inference and deployment workflows

The system behaves like a **production ML training + registry layer**, not a notebook experiment.

---

## Intended Use Cases

- NLP model fine-tuning under compute constraints  
- Enterprise ML / MLOps learning projects  
- Model registry and reuse workflows  
- Sentiment analysis or text classification systems  
- Hugging Face ecosystem exploration  
- Foundation for deployment via Spaces or APIs  

---

## System Architecture

```text
Raw Dataset (IMDb)
        ↓
Dataset Loader
        ↓
Preprocessing Pipeline
 (tokenization, truncation)
        ↓
LoRA / PEFT Fine-Tuning
 (BERT-based classifier)
        ↓
Interrupt-Safe Training
        ↓
Artifact Persistence
 (adapters, tokenizer, configs)
        ↓
Model Registry
 (Hugging Face Hub)
        ↓
Reusable Model Asset
```

## Key Features

- Hugging Face dataset ingestion using the `datasets` library  
- Tokenization via `AutoTokenizer` (WordPiece-based BERT)  
- Parameter-efficient fine-tuning using **LoRA / PEFT**  
- ~**0.27% trainable parameters** (compute-efficient training)  
- Interrupt-safe training with guaranteed adapter persistence  
- Clear separation of concerns:
  - datasets
  - preprocessing
  - training
  - registry
- Script-driven execution (no notebook dependency)  
- Public model registry on the **Hugging Face Hub**

## Tech Stack

Python

Hugging Face Transformers

Hugging Face Datasets

PEFT / LoRA

Trainer API

Hugging Face Hub

PyTorch

## How It Works (High Level)

IMDb dataset is loaded using Hugging Face Datasets

Text is tokenized using a pretrained BERT tokenizer

A BERT-based classifier is wrapped with LoRA adapters

Only adapter parameters are trained (PEFT)

Training is made interrupt-safe

Adapter weights and tokenizer are saved locally

Artifacts are pushed to the Hugging Face Hub

The model becomes a reusable registry asset

## Hugging Face Model Registry

The trained LoRA adapters are published publicly on the Hugging Face Hub.

🔗 Hugging Face Model (Primary)
https://huggingface.co/Koppula-Sreeja-Reddy/imdb-bert-lora

🔗 Hugging Face Model (Short Link)
https://lnkd.in/ghTD3nsJ

## This model can be:

reused without retraining

loaded directly for inference

deployed via Hugging Face Spaces

integrated into APIs and RAG pipelines

## How to Run Locally
git clone https://github.com/Sreeja-Reddy-0612/End-to-End-Model-Lifecycle-Manager
cd End-to-End-Model-Lifecycle-Manager

python -m venv .venv
.venv\Scripts\activate   # Windows

pip install -r requirements.txt

python -m scripts.test_lora_trainer

## Push Adapters to Hugging Face Hub
python -m registry.push_lora_to_hub


Authenticate once:

huggingface-cli login

## Project Status

✅ Dataset loading and preprocessing

✅ Tokenization pipeline

✅ LoRA / PEFT fine-tuning

✅ Interrupt-safe training logic

✅ Local artifact persistence

✅ Hugging Face authentication

✅ Model registry on HF Hub

✅ Hugging Face Spaces deployment

## Design Principles

Lifecycle over experiments

Parameter efficiency over brute force

Reproducibility over ad-hoc training

Clear separation of training and registry

Real-world ML engineering practices

Enterprise realism over demos

## Author

Sreeja Reddy
AI Engineer focused on:

LLM systems

RAG architectures

Model lifecycle engineering

GenAI reliability & evaluation

GitHub
https://github.com/Sreeja-Reddy-0612

LinkedIn
https://www.linkedin.com/in/sreeja-reddy-5ab708288/
