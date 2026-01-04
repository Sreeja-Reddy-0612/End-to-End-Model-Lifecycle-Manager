from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType

from training.metrics import compute_metrics


def run_lora_finetuning(
    model_name,
    tokenized_train_ds,
    tokenized_eval_ds,
):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    from peft import get_peft_model, LoraConfig, TaskType

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["query", "value"]
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
    output_dir="outputs/lora_tmp",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="no",   # ✅ FIXED
    num_train_epochs=1,
    logging_steps=100,
    save_strategy="no",
    report_to="none"
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_ds,
        eval_dataset=tokenized_eval_ds,
        tokenizer=tokenizer
    )

    # trainer.train()

    # # ✅ RETURN TOKENIZER ALSO
    # return trainer, model, tokenizer
    try:
        
        trainer.train()
    except KeyboardInterrupt:
        
        print("⚠️ Training interrupted manually — saving LoRA adapters anyway")
    finally:
        
        # Ensure adapters are ALWAYS saved
        model.save_pretrained("outputs/lora_finetune")
        tokenizer.save_pretrained("outputs/lora_finetune")
        print("✅ LoRA adapters saved to outputs/lora_finetune")

    return trainer, model, tokenizer

