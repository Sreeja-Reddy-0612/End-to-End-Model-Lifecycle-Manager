from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType

from training.metrics import compute_metrics


def run_lora_finetuning(
    model_name: str,
    tokenized_train_ds,
    tokenized_eval_ds,
    output_dir: str = "outputs/lora_finetune"
):
    """
    Runs LoRA / PEFT fine-tuning.
    """

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["query", "value"]
    )

    model = get_peft_model(base_model, lora_config)

    # Print trainable parameters (VERY IMPORTANT)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-4,  # higher LR for adapters
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        logging_dir=f"{output_dir}/logs",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_ds,
        eval_dataset=tokenized_eval_ds,
        compute_metrics=compute_metrics
    )

    trainer.train()

    return trainer, model
