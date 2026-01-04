from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

from training.metrics import compute_metrics


def run_full_finetuning(
    model_name: str,
    tokenized_train_ds,
    tokenized_eval_ds,
    output_dir: str = "outputs/full_finetune"
):
    """
    Runs full fine-tuning using Hugging Face Trainer.
    """

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none"  # no wandb yet
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_ds,
        eval_dataset=tokenized_eval_ds,
        compute_metrics=compute_metrics
    )

    trainer.train()

    return trainer
