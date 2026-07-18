# Loadingthe AG News dataset
import pandas as pd
from datasets import Dataset, DatasetDict

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

train_df.columns = ["label", "title", "description"]
test_df.columns = ["label", "title", "description"]

# Labels in CSV are 1-4 -> convert to 0-3 for PyTorch/HF
train_df["label"] = train_df["label"] - 1
test_df["label"] = test_df["label"] - 1

# Fix missing values BEFORE combining (prevents the crash from earlier)
train_df["title"] = train_df["title"].fillna("").astype(str)
train_df["description"] = train_df["description"].fillna("").astype(str)
test_df["title"] = test_df["title"].fillna("").astype(str)
test_df["description"] = test_df["description"].fillna("").astype(str)

# Combine title + description into one text field
train_df["text"] = train_df["title"] + " " + train_df["description"]
test_df["text"] = test_df["title"] + " " + test_df["description"]

label_names = ["World", "Sports", "Business", "Sci/Tech"]

dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False),
    "test": Dataset.from_pandas(test_df[["text", "label"]], preserve_index=False),
})

print(dataset)

# Tokenizing 
from transformers import AutoTokenizer

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_fn(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_fn, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch")

# Fine-tune BERT
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)
    f1 = f1_metric.compute(predictions=preds, references=labels, average="weighted")
    return {"accuracy": acc["accuracy"], "f1": f1["f1"]}

training_args = TrainingArguments(
    output_dir="./bert-agnews",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)
trainer.train()

# Evaluate
metrics = trainer.evaluate()
print("Final metrics:", metrics)

from sklearn.metrics import classification_report

preds_output = trainer.predict(tokenized_dataset["test"])
preds = np.argmax(preds_output.predictions, axis=-1)
labels = preds_output.label_ids
print(classification_report(labels, preds, target_names=label_names))

# STEP 5: Save model for deployment
trainer.save_model("./bert-agnews-final")
tokenizer.save_pretrained("./bert-agnews-final")
print("Model saved to ./bert-agnews-final")