import os
import numpy as np
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
import evaluate
from datasets import load_dataset
from typing import Dict
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

ID2LABEL: Dict[int, str] = {0: "正文", 1: "非正文"}
LABEL2ID: Dict[str, int] = {"正文": 0, "非正文": 1}

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

dataset = load_dataset("json", data_files="book_data.json", split="train")
dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)

tokenizer = AutoTokenizer.from_pretrained("Qwen2.5-0.5B")


def preprocess_data(examples, max_length=2048):
    text = examples["text"]
    encoding = tokenizer(text, max_length=max_length, padding=True, truncation=True)
    return encoding


encoder_data = dataset.map(
    preprocess_data, num_proc=12, remove_columns=['text']
)


def model_init():
    model = AutoModelForSequenceClassification.from_pretrained(
        "Qwen2.5-0.5B", num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    )
    model.config.pad_token_id = 151643
    return model


accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy_metric = accuracy.compute(predictions=predictions, references=labels)
    precision_metric = precision.compute(predictions=predictions, references=labels)
    return {**accuracy_metric, **precision_metric}


training_args = TrainingArguments(
    output_dir="book_cls",
    lr_scheduler_type="cosine",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    # weight_decay=0.01,
    logging_strategy="steps",
    logging_steps=0.01,
    eval_steps=0.1,
    eval_strategy="steps",
    save_strategy="epoch",
    use_liger_kernel=True,
    report_to="tensorboard",
    bf16=True,
)

trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=encoder_data["train"],
    eval_dataset=encoder_data["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1.0e-5, 1.0e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.1, step=0.01),
    }


def compute_objective(metrics: Dict[str, float]) -> float:
    return metrics["eval_loss"]


best_trial = trainer.hyperparameter_search(
    direction="minimize",
    backend="optuna",
    n_trials=100,
    hp_space=hp_space,
    compute_objective=compute_objective,
    sampler=TPESampler(n_startup_trials=15),
    pruner=HyperbandPruner(min_resource=5),
    study_name='book_search',
    storage="sqlite:///{}.db".format('book_search'),
    load_if_exists=True
)

print(best_trial)